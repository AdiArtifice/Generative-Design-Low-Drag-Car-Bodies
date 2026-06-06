import torch
import torch.nn as nn

class LatentDragRegressor(nn.Module):
    """
    A small Multi-Layer Perceptron (MLP) designed to predict the aerodynamic drag
    (e.g., drag_area) directly from a 256-dimensional latent vector z.
    
    Architecture:
      256 -> 128 -> 64 -> 1
    Uses LeakyReLU for non-zero gradients and BatchNorm/Dropout for regularization.
    """
    def __init__(self, latent_dim=256):
        super(LatentDragRegressor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
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
        
    def forward(self, z):
        """
        Args:
            z (torch.Tensor): Latent vectors of shape (B, latent_dim)
        Returns:
            torch.Tensor: Predicted drag values of shape (B, 1)
        """
        # Ensure z is 2D
        if z.dim() == 1:
            z = z.unsqueeze(0)
            
        return self.net(z)

if __name__ == "__main__":
    model = LatentDragRegressor(256)
    dummy_z = torch.randn(4, 256) # Batch of 4
    preds = model(dummy_z)
    print(f"Input shape: {dummy_z.shape}")
    print(f"Output shape: {preds.shape}")
    print("LatentDragRegressor initialized successfully.")
