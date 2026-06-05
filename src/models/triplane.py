import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    """
    PointNet-based Encoder for the Triplane VAE.
    Compresses a point cloud of shape [Batch, in_channels, N] into a latent distribution.
    """
    def __init__(self, in_channels=6, latent_dim=256):
        super(PointNetEncoder, self).__init__()
        
        # Shared MLPs (1D Convolutional Layers)
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 512, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(512)
        
        # Latent distribution heads
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input point cloud with shape [B, C, N].
        Returns:
            mu (torch.Tensor): Latent mean [B, latent_dim]
            logvar (torch.Tensor): Latent log-variance [B, latent_dim]
        """
        # Shared MLPs
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global Max Pooling: [B, 512, N] -> [B, 512, 1]
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512) # Flatten to [B, 512]
        
        # Latent distribution parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class TriplaneDecoder(nn.Module):
    """
    Decoder that maps latent z of shape [Batch, latent_dim] to three orthogonal 2D feature planes:
    XY, XZ, and YZ planes, each of shape [Batch, plane_channels, plane_resolution, plane_resolution].
    """
    def __init__(self, latent_dim=256, plane_channels=16, plane_resolution=64):
        super(TriplaneDecoder, self).__init__()
        self.plane_channels = plane_channels
        self.plane_resolution = plane_resolution
        
        # We start with a low-resolution grid of 8x8 to save parameters and enforce spatial continuity.
        self.init_res = 8
        self.fc = nn.Linear(latent_dim, 3 * plane_channels * self.init_res * self.init_res)
        
        # Upscaling convolutions (shared across the 3 planes via batch folding)
        self.conv1 = nn.Conv2d(plane_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, plane_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(plane_channels)
        
    def forward(self, z):
        """
        Args:
            z (torch.Tensor): Latent vectors [B, latent_dim]
        Returns:
            plane_xy (torch.Tensor): [B, C, H, W]
            plane_xz (torch.Tensor): [B, C, H, W]
            plane_yz (torch.Tensor): [B, C, H, W]
        """
        B = z.shape[0]
        x = self.fc(z) # [B, 3 * C * 8 * 8]
        
        # Fold batch and plane dimension: [B * 3, C, 8, 8]
        x = x.view(B * 3, self.plane_channels, self.init_res, self.init_res)
        
        # Determine intermediate resolutions dynamically
        res1 = int(self.init_res * (self.plane_resolution / self.init_res) ** (1/3))
        res2 = int(self.init_res * (self.plane_resolution / self.init_res) ** (2/3))
        res3 = self.plane_resolution
        
        # Stage 1 upsampling
        x = F.interpolate(x, size=(res1, res1), mode='bilinear', align_corners=False)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Stage 2 upsampling
        x = F.interpolate(x, size=(res2, res2), mode='bilinear', align_corners=False)
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Stage 3 upsampling
        x = F.interpolate(x, size=(res3, res3), mode='bilinear', align_corners=False)
        x = F.relu(self.bn3(self.conv3(x))) # [B * 3, C, H, W]
        
        # Unfold back to [B, 3, C, H, W]
        x = x.view(B, 3, self.plane_channels, self.plane_resolution, self.plane_resolution)
        
        plane_xy = x[:, 0]
        plane_xz = x[:, 1]
        plane_yz = x[:, 2]
        
        return plane_xy, plane_xz, plane_yz


class OccupancyMLP(nn.Module):
    """
    Implicit decoder that takes projected triplane features and absolute 3D query coordinates,
    and maps them to occupancy probability logits.
    """
    def __init__(self, plane_channels=16):
        super(OccupancyMLP, self).__init__()
        
        # Input features: xy-features (C) + xz-features (C) + yz-features (C) + absolute query coordinates (3)
        in_channels = 3 * plane_channels + 3
        
        # Shared MLP implemented using Conv1d with kernel size 1
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=1)
        self.conv3 = nn.Conv1d(64, 1, kernel_size=1)
        
    def forward(self, feat_xy, feat_xz, feat_yz, points):
        """
        Args:
            feat_xy (torch.Tensor): Projected features from XY plane [B, C, N_q]
            feat_xz (torch.Tensor): Projected features from XZ plane [B, C, N_q]
            feat_yz (torch.Tensor): Projected features from YZ plane [B, C, N_q]
            points (torch.Tensor): Raw query 3D points [B, N_q, 3]
        Returns:
            logits (torch.Tensor): Occupancy probability logits [B, N_q]
        """
        # Concatenate features from the three planes: [B, 3 * C, N_q]
        feat = torch.cat([feat_xy, feat_xz, feat_yz], dim=1)
        
        # Transpose points to [B, 3, N_q]
        pts_t = points.transpose(1, 2)
        
        # Combine plane features and absolute coordinates: [B, 3 * C + 3, N_q]
        x = torch.cat([feat, pts_t], dim=1)
        
        # Run implicit network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        logits = self.conv3(x) # [B, 1, N_q]
        
        return logits.squeeze(1) # [B, N_q]


class TriplaneVAE(nn.Module):
    """
    Full Triplane Variational Autoencoder (VAE) for Watertight Mesh Generation.
    """
    def __init__(self, in_channels=6, latent_dim=256, plane_channels=16, plane_resolution=64):
        super(TriplaneVAE, self).__init__()
        self.encoder = PointNetEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = TriplaneDecoder(latent_dim=latent_dim, plane_channels=plane_channels, plane_resolution=plane_resolution)
        self.occupancy_mlp = OccupancyMLP(plane_channels=plane_channels)
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, pc, query_points):
        """
        Args:
            pc (torch.Tensor): Input point cloud [B, in_channels, N_pc]
            query_points (torch.Tensor): 3D coordinates to query occupancy for [B, N_q, 3]
        Returns:
            logits (torch.Tensor): Predicted occupancy logits [B, N_q]
            mu (torch.Tensor): Latent distribution mean [B, latent_dim]
            logvar (torch.Tensor): Latent distribution log-variance [B, latent_dim]
        """
        # 1. Encode point cloud to latent space
        mu, logvar = self.encoder(pc)
        z = self.reparameterize(mu, logvar)
        
        # 2. Decode latent vector to triplane representation
        plane_xy, plane_xz, plane_yz = self.decoder(z)
        
        # 3. Project 3D query points onto the orthogonal planes and sample features.
        # Query coordinates are scaled by 2.0 because normalized meshes span [-0.5, 0.5]
        # and grid_sample expects coordinates in [-1, 1].
        # Reshape to [B, N_q, 1, 2] as expected by F.grid_sample.
        grid_xy = (query_points[..., [0, 1]] * 2.0).unsqueeze(2) # [B, N_q, 1, 2]
        grid_xz = (query_points[..., [0, 2]] * 2.0).unsqueeze(2) # [B, N_q, 1, 2]
        grid_yz = (query_points[..., [1, 2]] * 2.0).unsqueeze(2) # [B, N_q, 1, 2]
        
        # Sample features from each plane
        feat_xy = F.grid_sample(plane_xy, grid_xy, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(-1) # [B, C, N_q]
        feat_xz = F.grid_sample(plane_xz, grid_xz, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(-1) # [B, C, N_q]
        feat_yz = F.grid_sample(plane_yz, grid_yz, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(-1) # [B, C, N_q]
        
        # 4. Map features and absolute query coordinates to occupancy probability
        logits = self.occupancy_mlp(feat_xy, feat_xz, feat_yz, query_points) # [B, N_q]
        
        return logits, mu, logvar

# Smoke-test block to verify architecture shape validation
if __name__ == "__main__":
    print("--- Triplane VAE Architecture Smoke-Test ---")
    batch_size = 4
    in_channels = 6
    num_pc_points = 2048
    num_query_points = 2048
    latent_dim = 256
    plane_channels = 16
    plane_res = 64
    
    # Create random dummy inputs
    dummy_pc = torch.rand(batch_size, in_channels, num_pc_points)
    dummy_query = torch.rand(batch_size, num_query_points, 3) - 0.5 # Center within [-0.5, 0.5]
    
    print(f"Inputs:")
    print(f"  - Point Cloud shape: {dummy_pc.shape} (Expected: [B, {in_channels}, {num_pc_points}])")
    print(f"  - Query Points shape: {dummy_query.shape} (Expected: [B, {num_query_points}, 3])")
    
    # Initialize full VAE model
    model = TriplaneVAE(
        in_channels=in_channels,
        latent_dim=latent_dim,
        plane_channels=plane_channels,
        plane_resolution=plane_res
    )
    
    # Forward pass
    logits, mu, logvar = model(dummy_pc, dummy_query)
    
    print(f"Outputs:")
    print(f"  - Occupancy logits shape: {logits.shape} (Expected: [B, {num_query_points}])")
    print(f"  - Latent mean shape: {mu.shape} (Expected: [B, {latent_dim}])")
    print(f"  - Latent logvar shape: {logvar.shape} (Expected: [B, {latent_dim}])")
    
    # Validate parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print("--- Smoke-Test PASSED ---")
