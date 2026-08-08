import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import trimesh
from pathlib import Path

class VehiclePointCloudDataset(Dataset):
    """
    Custom PyTorch Dataset for loading 3D vehicle point clouds (coordinates and normals)
    and their associated aerodynamic targets (Cd, Drag Area).
    """
    def __init__(self, csv_path="metadata/metadata.csv", scales_path="metadata/target_scales.json", 
                 split=None, num_points=2048, normalize_targets=True):
        """
        Args:
            csv_path (str): Path to master metadata CSV.
            scales_path (str): Path to target scales JSON.
            split (str): One of 'train', 'val', 'test', or None for all.
            num_points (int): Number of points to sample dynamically.
            normalize_targets (bool): Whether to normalize targets using target_scales.json stats.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")
            
        self.df = pd.read_csv(csv_path)
        
        # Filter by split if provided
        if split is not None:
            if split not in ["train", "val", "test"]:
                raise ValueError("split must be one of 'train', 'val', 'test'")
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
            
        self.num_points = num_points
        self.normalize_targets = normalize_targets
        
        # Load targets statistics for normalization
        if self.normalize_targets:
            if not os.path.exists(scales_path):
                raise FileNotFoundError(f"Scales JSON file not found at {scales_path}")
            with open(scales_path, "r") as f:
                self.scales = json.load(f)
                
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load point cloud
        pc_path = row["pointcloud_path"]
        if not os.path.exists(pc_path):
            raise FileNotFoundError(f"Point cloud file not found at {pc_path}")
            
        pcd = trimesh.load(pc_path)
        raw_data = pcd.metadata["_ply_raw"]["vertex"]["data"]
        points = np.stack([raw_data['x'], raw_data['y'], raw_data['z']], axis=-1).astype(np.float32)
        normals = np.stack([raw_data['nx'], raw_data['ny'], raw_data['nz']], axis=-1).astype(np.float32)
        
        # 2. Downsample points dynamically (crucial for CPU training and batching)
        num_pc_points = len(points)
        if num_pc_points >= self.num_points:
            indices = np.random.choice(num_pc_points, self.num_points, replace=False)
        else:
            # Handle fallback if point cloud has fewer points (should not happen for 50k points)
            indices = np.random.choice(num_pc_points, self.num_points, replace=True)
            
        points = points[indices]
        normals = normals[indices]
        
        # 3. Concatenate coordinates [x, y, z] and normals [nx, ny, nz] -> shape: [num_points, 6]
        features = np.concatenate([points, normals], axis=1) # shape: [num_points, 6]
        
        # Transpose to [6, num_points] as expected by PyTorch 1D convolutions (PointNet)
        features_tensor = torch.tensor(features, dtype=torch.float32).t() # shape: [6, num_points]
        
        # 4. Extract targets & body_type class index (0: F, 1: E, 2: N)
        body_type_map = {'F': 0, 'E': 1, 'N': 2}
        if "body_type_idx" in row and not pd.isna(row["body_type_idx"]):
            class_idx = int(row["body_type_idx"])
        elif "body_type" in row and not pd.isna(row["body_type"]):
            class_idx = body_type_map.get(str(row["body_type"])[0].upper(), 0)
        else:
            cfg = str(row.get("config", row["id"]))
            class_idx = body_type_map.get(cfg[0].upper(), 0)
            
        class_idx_tensor = torch.tensor(class_idx, dtype=torch.long)
        
        cd_raw = float(row["cd"])
        drag_area_raw = float(row["drag_area"])
        
        # 5. Apply target normalization if requested
        if self.normalize_targets:
            cd_mean, cd_std = self.scales["cd"]["mean"], self.scales["cd"]["std"]
            drag_area_mean, drag_area_std = self.scales["drag_area"]["mean"], self.scales["drag_area"]["std"]
            
            cd_norm = (cd_raw - cd_mean) / cd_std
            drag_area_norm = (drag_area_raw - drag_area_mean) / drag_area_std
        else:
            cd_norm = cd_raw
            drag_area_norm = drag_area_raw
            
        # Compile inputs and targets
        targets = {
            "cd_raw": torch.tensor(cd_raw, dtype=torch.float32),
            "drag_area_raw": torch.tensor(drag_area_raw, dtype=torch.float32),
            "cd": torch.tensor(cd_norm, dtype=torch.float32),
            "drag_area": torch.tensor(drag_area_norm, dtype=torch.float32),
            "class_idx": class_idx_tensor
        }
        
        return features_tensor, class_idx_tensor, targets

class VehicleOccupancyDataset(Dataset):
    """
    Custom PyTorch Dataset for loading 3D vehicle point clouds (coordinates and normals),
    paired with query 3D points and their occupancy labels (inside/outside),
    along with associated aerodynamic targets.
    """
    def __init__(self, csv_path="metadata/metadata.csv", scales_path="metadata/target_scales.json", 
                 occupancy_dir="occupancy", split=None, num_points=2048, 
                 num_query_points=2048, normalize_targets=True):
        """
        Args:
            csv_path (str): Path to master metadata CSV.
            scales_path (str): Path to target scales JSON.
            occupancy_dir (str): Root directory containing occupancy .npz files.
            split (str): One of 'train', 'val', 'test', or None for all.
            num_points (int): Number of point cloud points to sample dynamically.
            num_query_points (int): Number of occupancy query points to load/sample from the NPZ.
            normalize_targets (bool): Whether to normalize targets.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata file not found at {csv_path}")
            
        self.df = pd.read_csv(csv_path)
        
        # Filter by split if provided
        if split is not None:
            if split not in ["train", "val", "test"]:
                raise ValueError("split must be one of 'train', 'val', 'test'")
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
            
        self.num_points = num_points
        self.num_query_points = num_query_points
        self.occupancy_dir = occupancy_dir
        self.normalize_targets = normalize_targets
        
        # Load targets statistics for normalization
        if self.normalize_targets:
            if not os.path.exists(scales_path):
                raise FileNotFoundError(f"Scales JSON file not found at {scales_path}")
            with open(scales_path, "r") as f:
                self.scales = json.load(f)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load point cloud
        pc_path = row["pointcloud_path"]
        if not os.path.exists(pc_path):
            raise FileNotFoundError(f"Point cloud file not found at {pc_path}")
            
        pcd = trimesh.load(pc_path)
        raw_data = pcd.metadata["_ply_raw"]["vertex"]["data"]
        points = np.stack([raw_data['x'], raw_data['y'], raw_data['z']], axis=-1).astype(np.float32)
        normals = np.stack([raw_data['nx'], raw_data['ny'], raw_data['nz']], axis=-1).astype(np.float32)
        
        # Downsample points dynamically
        num_pc_points = len(points)
        if num_pc_points >= self.num_points:
            indices = np.random.choice(num_pc_points, self.num_points, replace=False)
        else:
            indices = np.random.choice(num_pc_points, self.num_points, replace=True)
            
        points = points[indices]
        normals = normals[indices]
        
        # Concatenate coordinates [x, y, z] and normals [nx, ny, nz] -> shape: [num_points, 6]
        features = np.concatenate([points, normals], axis=1) # shape: [num_points, 6]
        features_tensor = torch.tensor(features, dtype=torch.float32).t() # shape: [6, num_points]
        
        # 2. Load occupancy query points and labels
        norm_stl_path = Path(row["normalized_stl_path"])
        rel_to_norm = norm_stl_path.relative_to("normalized")
        occ_path = Path(self.occupancy_dir) / rel_to_norm.parent / f"{rel_to_norm.stem}_occ.npz"
        
        if not occ_path.exists():
            raise FileNotFoundError(f"Occupancy file not found at {occ_path}. Please run preprocess_occupancy.py first.")
            
        occ_data = np.load(str(occ_path))
        q_pts = occ_data["query_points"].astype(np.float32) # shape: [TotalQueryPoints, 3]
        occ_lbls = occ_data["occupancy"].astype(np.float32) # shape: [TotalQueryPoints]
        
        # Sample query points if requested count is different from what's stored
        num_avail_query = len(q_pts)
        if num_avail_query >= self.num_query_points:
            q_indices = np.random.choice(num_avail_query, self.num_query_points, replace=False)
        else:
            q_indices = np.random.choice(num_avail_query, self.num_query_points, replace=True)
            
        q_pts = q_pts[q_indices]
        occ_lbls = occ_lbls[q_indices]
        
        query_points_tensor = torch.tensor(q_pts, dtype=torch.float32) # shape: [num_query_points, 3]
        occupancy_tensor = torch.tensor(occ_lbls, dtype=torch.float32) # shape: [num_query_points]
        
        # 3. Extract targets & body_type class index
        body_type_map = {'F': 0, 'E': 1, 'N': 2}
        if "body_type_idx" in row and not pd.isna(row["body_type_idx"]):
            class_idx = int(row["body_type_idx"])
        elif "body_type" in row and not pd.isna(row["body_type"]):
            class_idx = body_type_map.get(str(row["body_type"])[0].upper(), 0)
        else:
            cfg = str(row.get("config", row["id"]))
            class_idx = body_type_map.get(cfg[0].upper(), 0)
            
        class_idx_tensor = torch.tensor(class_idx, dtype=torch.long)
        
        cd_raw = float(row["cd"])
        drag_area_raw = float(row["drag_area"])
        
        if self.normalize_targets:
            cd_mean, cd_std = self.scales["cd"]["mean"], self.scales["cd"]["std"]
            drag_area_mean, drag_area_std = self.scales["drag_area"]["mean"], self.scales["drag_area"]["std"]
            
            cd_norm = (cd_raw - cd_mean) / cd_std
            drag_area_norm = (drag_area_raw - drag_area_mean) / drag_area_std
        else:
            cd_norm = cd_raw
            drag_area_norm = drag_area_raw
            
        targets = {
            "cd_raw": torch.tensor(cd_raw, dtype=torch.float32),
            "drag_area_raw": torch.tensor(drag_area_raw, dtype=torch.float32),
            "cd": torch.tensor(cd_norm, dtype=torch.float32),
            "drag_area": torch.tensor(drag_area_norm, dtype=torch.float32),
            "class_idx": class_idx_tensor
        }
        
        return features_tensor, query_points_tensor, occupancy_tensor, class_idx_tensor, targets

# Smoke-test block to verify data loading pipelines
if __name__ == "__main__":
    print("--- Running Dataset Smoke-Tests ---")
    try:
        # Test 1: VehiclePointCloudDataset
        print("\nTesting VehiclePointCloudDataset...")
        dataset_pc = VehiclePointCloudDataset(
            csv_path="metadata/metadata.csv",
            scales_path="metadata/target_scales.json",
            split="train",
            num_points=2048,
            normalize_targets=True
        )
        print(f"PointCloud dataset created. Samples: {len(dataset_pc)}")
        features, class_idx, targets = dataset_pc[0]
        print(f"PointCloud item retrieval: Features shape {features.shape}, Class Index: {class_idx.item()}")
        
        # Test 2: VehicleOccupancyDataset
        print("\nTesting VehicleOccupancyDataset...")
        dataset_occ = VehicleOccupancyDataset(
            csv_path="metadata/metadata.csv",
            scales_path="metadata/target_scales.json",
            split="train",
            num_points=2048,
            num_query_points=2048,
            normalize_targets=True
        )
        print(f"Occupancy dataset created. Samples: {len(dataset_occ)}")
        features, query_points, occupancy, class_idx, targets = dataset_occ[0]
        print("Occupancy item retrieval:")
        print(f"  - PC Features shape: {features.shape} (Expected: [6, 2048])")
        print(f"  - Query points shape: {query_points.shape} (Expected: [2048, 3])")
        print(f"  - Occupancy shape: {occupancy.shape} (Expected: [2048])")
        print(f"  - Class index: {class_idx.item()} (Expected: 0, 1, or 2)")
        print(f"  - Inside points count: {int(occupancy.sum())}")
        
        # Test DataLoader batching
        dataloader = DataLoader(dataset_occ, batch_size=4, shuffle=True)
        b_features, b_q_pts, b_occ, b_class_idx, b_targets = next(iter(dataloader))
        print("\nDataLoader batching test successful!")
        print(f"  - Batch features shape: {b_features.shape} (Expected: [4, 6, 2048])")
        print(f"  - Batch query points shape: {b_q_pts.shape} (Expected: [4, 2048, 3])")
        print(f"  - Batch occupancy shape: {b_occ.shape} (Expected: [4, 2048])")
        print(f"  - Batch class indices: {b_class_idx.tolist()} (Expected: len 4)")
        
        print("\n--- All Dataset Smoke-Tests PASSED! ---")
        
    except Exception as e:
        print("\n--- Smoke-Test FAILED! ---")
        import traceback
        traceback.print_exc()


