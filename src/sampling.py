import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional

def compute_point_curvature(points: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Computes local surface variation (curvature) score for each point in point cloud
    using k-NN covariance matrix eigenvalue decomposition.
    
    sigma(p_i) = lambda_0 / (lambda_0 + lambda_1 + lambda_2)
    where lambda_0 <= lambda_1 <= lambda_2 are the eigenvalues of the local covariance matrix.
    
    Args:
        points (np.ndarray): Array of shape (N, 3) containing 3D coordinates.
        k (int): Number of nearest neighbors for local surface estimation.
        
    Returns:
        np.ndarray: Curvature scores of shape (N,).
    """
    points = np.asarray(points, dtype=np.float32)
    N = len(points)
    if N == 0:
        return np.zeros(0, dtype=np.float32)
    
    k_actual = min(k, N)
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k_actual)
    
    if k_actual <= 1:
        return np.zeros(N, dtype=np.float32)
        
    curvatures = np.zeros(N, dtype=np.float32)
    for i in range(N):
        neighbors = points[indices[i]]  # [k, 3]
        centered = neighbors - neighbors.mean(axis=0)
        cov = np.dot(centered.T, centered) / k_actual  # [3, 3]
        eigvals = np.linalg.eigvalsh(cov)  # Sorted ascending: lambda_0 <= lambda_1 <= lambda_2
        eigvals = np.maximum(eigvals, 0.0)  # Numerical stability
        total = eigvals.sum()
        if total > 1e-12:
            curvatures[i] = eigvals[0] / total
        else:
            curvatures[i] = 0.0
            
    return curvatures

def farthest_point_sampling(points: np.ndarray, num_samples: int, start_idx: int = 0) -> np.ndarray:
    """
    Vectorized iterative Farthest Point Sampling (FPS) algorithm.
    
    Args:
        points (np.ndarray): (N, 3) or (N, D) array of points.
        num_samples (int): Number of points to sample.
        start_idx (int): Initial point index to seed FPS.
        
    Returns:
        np.ndarray: Array of sampled point indices of shape (num_samples,).
    """
    points = np.asarray(points, dtype=np.float32)
    N = len(points)
    num_samples = min(num_samples, N)
    if N == 0 or num_samples == 0:
        return np.array([], dtype=np.int64)
        
    sampled_indices = np.zeros(num_samples, dtype=np.int64)
    sampled_indices[0] = start_idx
    
    curr_point = points[start_idx]
    min_dists = np.sum((points - curr_point) ** 2, axis=1)
    
    for i in range(1, num_samples):
        next_idx = np.argmax(min_dists)
        sampled_indices[i] = next_idx
        curr_point = points[next_idx]
        dists = np.sum((points - curr_point) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        
    return sampled_indices

def curvature_saliency_sampling(points: np.ndarray, curvatures: np.ndarray, num_samples: int, 
                                temperature: float = 1.0, exclude_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Samples points biased towards high-curvature boundary regions using a softmax saliency probability distribution.
    
    Args:
        points (np.ndarray): (N, 3) point array.
        curvatures (np.ndarray): (N,) curvature scores.
        num_samples (int): Number of boundary points to sample.
        temperature (float): Temperature parameter for probability distribution sharpening.
        exclude_indices (Optional[np.ndarray]): Point indices to exclude (e.g. already selected FPS points).
        
    Returns:
        np.ndarray: Array of sampled point indices of shape (num_samples,).
    """
    N = len(points)
    if N == 0 or num_samples == 0:
        return np.array([], dtype=np.int64)
        
    probs = curvatures.copy().astype(np.float64)
    if exclude_indices is not None and len(exclude_indices) > 0:
        probs[exclude_indices] = -1e9  # Exclude already selected points
        
    max_c = np.max(probs)
    if max_c < -1e8:
        probs = np.ones(N) / N
        if exclude_indices is not None:
            probs[exclude_indices] = 0.0
    else:
        probs = np.exp((probs - max_c) / max(temperature, 1e-5))
        if exclude_indices is not None:
            probs[exclude_indices] = 0.0
            
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs /= prob_sum
    else:
        mask = np.ones(N, dtype=bool)
        if exclude_indices is not None:
            mask[exclude_indices] = False
        if mask.sum() > 0:
            probs[mask] = 1.0 / mask.sum()
            probs[~mask] = 0.0
        else:
            probs = np.ones(N) / N
            
    num_samples = min(num_samples, N)
    replace_sampling = (np.count_nonzero(probs) < num_samples)
    sampled_indices = np.random.choice(N, size=num_samples, replace=replace_sampling, p=probs)
    return sampled_indices

def hybrid_fps_curvature_sampling(points: np.ndarray, normals: np.ndarray, total_points: int = 2048, 
                                  fps_ratio: float = 0.75, k_neighbors: int = 20) -> np.ndarray:
    """
    Main orchestrator for Hybrid 75% FPS + 25% Curvature Point Cloud Sampling.
    
    Args:
        points (np.ndarray): Dense point coordinates of shape (N, 3).
        normals (np.ndarray): Point normal vectors of shape (N, 3).
        total_points (int): Total target sampled points (default: 2048).
        fps_ratio (float): Ratio of uniform FPS points (default: 0.75 -> 1536 points).
        k_neighbors (int): Nearest neighbors for curvature estimation (default: 20).
        
    Returns:
        np.ndarray: Concatenated array of shape (total_points, 6) containing [x, y, z, nx, ny, nz].
    """
    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    
    num_fps = int(round(total_points * fps_ratio))
    num_curv = total_points - num_fps
    
    # 1. 75% FPS for uniform global macro-geometry coverage
    fps_indices = farthest_point_sampling(points, num_samples=num_fps, start_idx=0)
    
    # 2. 25% Curvature Saliency for boundary/high-curvature local features
    curvatures = compute_point_curvature(points, k=k_neighbors)
    curv_indices = curvature_saliency_sampling(points, curvatures, num_samples=num_curv, exclude_indices=fps_indices)
    
    # 3. Concatenate indices and features
    all_indices = np.concatenate([fps_indices, curv_indices])
    sampled_points = points[all_indices]
    sampled_normals = normals[all_indices]
    
    features = np.concatenate([sampled_points, sampled_normals], axis=1)  # shape: (total_points, 6)
    return features
