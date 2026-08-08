import torch
import torch.nn as nn

class LatentDragRegressor(nn.Module):
    """
    Multi-Layer Perceptron (MLP) designed to predict aerodynamic drag (Cd or drag_area)
    directly from a 256-dimensional latent vector z and optional category embedding.
    
    Architecture:
      (latent_dim + embed_dim) -> 128 -> 64 -> 1
    Uses LeakyReLU for non-zero gradients and LayerNorm/Dropout for regularization.
    """
    def __init__(self, latent_dim=256, num_classes=3, embed_dim=16):
        super(LatentDragRegressor, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        if num_classes > 0 and embed_dim > 0:
            self.class_emb = nn.Embedding(num_classes, embed_dim)
            in_dim = latent_dim + embed_dim
        else:
            self.class_emb = None
            in_dim = latent_dim
            
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(64, 1),
            nn.Softplus()  # Guarantees positive drag prediction
        )
        
    def forward(self, z, class_idx=None):
        """
        Args:
            z (torch.Tensor): Latent vectors of shape (B, latent_dim)
            class_idx (torch.Tensor, optional): Integer class labels [B]
        Returns:
            torch.Tensor: Predicted drag values of shape (B, 1)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            
        if self.class_emb is not None and class_idx is not None:
            c_emb = self.class_emb(class_idx) # [B, embed_dim]
            z_in = torch.cat([z, c_emb], dim=-1)
        else:
            z_in = z
            
        return self.net(z_in)

if __name__ == "__main__":
    print("--- Testing LatentDragRegressor ---")
    model = LatentDragRegressor(latent_dim=256, num_classes=3, embed_dim=16)
    dummy_z = torch.randn(4, 256)
    dummy_c = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    preds = model(dummy_z, dummy_c)
    print(f"Input shape: {dummy_z.shape}, Class shape: {dummy_c.shape}")
    print(f"Output shape: {preds.shape}")
    print("Conditional LatentDragRegressor initialized successfully.")
