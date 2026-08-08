import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    """
    PointNet-based Encoder for the VAE.
    Compresses a point cloud of shape [Batch, in_channels, N] into a latent distribution.
    Optionally accepts a class embedding c_emb of shape [Batch, embed_dim] for C-VAE.
    """
    def __init__(self, in_channels=6, latent_dim=128, embed_dim=16):
        super(PointNetEncoder, self).__init__()
        self.embed_dim = embed_dim
        
        # Shared MLPs (1D Convolutional Layers)
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 512, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(512)
        
        # Latent distribution heads
        in_fc_dim = 512 + embed_dim
        self.fc_mu = nn.Linear(in_fc_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_fc_dim, latent_dim)
        
    def forward(self, x, c_emb=None):
        """
        Args:
            x (torch.Tensor): Input point cloud with shape [B, C, N].
            c_emb (torch.Tensor, optional): Conditioning class embedding [B, embed_dim].
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
        
        if c_emb is not None and self.embed_dim > 0:
            x = torch.cat([x, c_emb], dim=-1)
            
        # Latent distribution parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class PointCloudDecoder(nn.Module):
    """
    MLP Decoder for the VAE.
    Maps a latent vector z of shape [Batch, latent_dim] + condition embedding [Batch, embed_dim]
    to reconstructed coordinates [Batch, 3, N].
    """
    def __init__(self, latent_dim=128, out_points=2048, embed_dim=16):
        super(PointCloudDecoder, self).__init__()
        self.out_points = out_points
        self.embed_dim = embed_dim
        
        in_fc_dim = latent_dim + embed_dim
        self.fc1 = nn.Linear(in_fc_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Output is coordinate reconstruction (3 coordinates: x, y, z per point)
        self.fc_out = nn.Linear(1024, out_points * 3)
        
    def forward(self, z, c_emb=None):
        """
        Args:
            z (torch.Tensor): Latent vectors [B, latent_dim]
            c_emb (torch.Tensor, optional): Conditioning class embedding [B, embed_dim].
        Returns:
            torch.Tensor: Reconstructed coordinates [B, 3, out_points]
        """
        if c_emb is not None and self.embed_dim > 0:
            z_in = torch.cat([z, c_emb], dim=-1)
        else:
            z_in = z
            
        x = F.relu(self.bn1(self.fc1(z_in)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        
        x = self.fc_out(x)
        
        # Reshape to [B, out_points, 3] and then transpose to [B, 3, out_points]
        x = x.view(-1, self.out_points, 3)
        x = x.transpose(1, 2) # [B, 3, out_points]
        
        return x


class PointNetVAE(nn.Module):
    """
    Full 3D Conditional PointNet Variational Autoencoder (C-VAE).
    """
    def __init__(self, in_channels=6, latent_dim=128, num_points=2048, num_classes=3, embed_dim=16):
        super(PointNetVAE, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        if num_classes > 0 and embed_dim > 0:
            self.class_emb = nn.Embedding(num_classes, embed_dim)
        else:
            self.class_emb = None
            self.embed_dim = 0
            
        self.encoder = PointNetEncoder(in_channels=in_channels, latent_dim=latent_dim, embed_dim=self.embed_dim)
        self.decoder = PointCloudDecoder(latent_dim=latent_dim, out_points=num_points, embed_dim=self.embed_dim)
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x, class_idx=None):
        """
        Args:
            x (torch.Tensor): Point cloud inputs [B, C, N]
            class_idx (torch.Tensor, optional): Integer class labels [B]
        Returns:
            recon_x (torch.Tensor): Reconstructed coordinates [B, 3, N]
            mu (torch.Tensor): Latent mean [B, latent_dim]
            logvar (torch.Tensor): Latent log-variance [B, latent_dim]
        """
        c_emb = None
        if self.class_emb is not None and class_idx is not None:
            c_emb = self.class_emb(class_idx)
            
        mu, logvar = self.encoder(x, c_emb=c_emb)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, c_emb=c_emb)
        return recon_x, mu, logvar


def chamfer_distance(p1, p2):
    """
    Computes symmetric Chamfer Distance between two point clouds.
    """
    if p1.shape[1] == 3 and len(p1.shape) == 3:
        p1 = p1.transpose(1, 2)
    if p2.shape[1] == 3 and len(p2.shape) == 3:
        p2 = p2.transpose(1, 2)
        
    B, N, C = p1.shape
    _, M, _ = p2.shape
    
    p1_sq = torch.sum(p1 ** 2, dim=-1, keepdim=True)
    p2_sq = torch.sum(p2 ** 2, dim=-1, keepdim=True).transpose(1, 2)
    
    inner_prod = torch.bmm(p1, p2.transpose(1, 2))
    
    dist = p1_sq + p2_sq - 2.0 * inner_prod
    dist = torch.clamp(dist, min=0.0)
    
    min_dist_p1 = torch.min(dist, dim=2)[0]
    min_dist_p2 = torch.min(dist, dim=1)[0]
    
    chamfer = torch.mean(min_dist_p1) + torch.mean(min_dist_p2)
    return chamfer


# Smoke-test block for VAE shape verification
if __name__ == "__main__":
    print("--- PointNet C-VAE Smoke-Test ---")
    batch_size = 4
    channels = 6
    num_points = 2048
    latent_dim = 128
    num_classes = 3
    embed_dim = 16
    
    dummy_input = torch.rand(batch_size, channels, num_points)
    dummy_class = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    print(f"Input shape: {dummy_input.shape}")
    
    model = PointNetVAE(in_channels=channels, latent_dim=latent_dim, num_points=num_points, 
                        num_classes=num_classes, embed_dim=embed_dim)
    
    recon_x, mu, logvar = model(dummy_input, dummy_class)
    print(f"Reconstructed coordinate shape: {recon_x.shape} (Expected: [{batch_size}, 3, {num_points}])")
    print(f"Latent mu shape: {mu.shape} (Expected: [{batch_size}, {latent_dim}])")
    print(f"Latent logvar shape: {logvar.shape} (Expected: [{batch_size}, {latent_dim}])")
    
    input_coords = dummy_input[:, :3, :]
    loss_cd = chamfer_distance(input_coords, recon_x)
    print(f"Chamfer Distance loss: {loss_cd.item():.4f}")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print("--- Smoke-Test PASSED ---")
