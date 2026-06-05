import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetRegressor(nn.Module):
    """
    3D PointNet architecture adapted for regression tasks.
    Takes point clouds of shape [Batch, Channels, NumPoints] and outputs a scalar target.
    
    Expected input Channels: 6 (x, y, z, nx, ny, nz)
    """
    def __init__(self, in_channels=6, dropout_prob=0.3):
        super(PointNetRegressor, self).__init__()
        
        # 1D Convolutional Layers (acting independently on each point)
        # Input shape: [Batch, in_channels, N]
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 512, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(512)
        
        # Multi-Layer Perceptron (MLP) Regression Head
        self.fc1 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        self.fc2 = nn.Linear(256, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        self.fc3 = nn.Linear(64, 1) # Single scalar output (e.g., drag_area)
        
    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input point cloud of shape [B, C, N].
        Returns:
            torch.Tensor: Predicted continuous value of shape [B].
        """
        # --- Shared MLPs (1D Convolutions) ---
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # --- Global Feature Extraction ---
        # Global Max Pooling across the N dimension (points)
        # Input: [B, 512, N] -> Output: [B, 512, 1]
        x = torch.max(x, 2, keepdim=True)[0]
        
        # Flatten to [B, 512]
        x = x.view(-1, 512)
        
        # --- Regression Head ---
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        # Squeeze the output from [B, 1] to [B]
        return x.squeeze(1)

# Smoke-test block for architecture shape validation
if __name__ == "__main__":
    print("--- PointNet Architecture Smoke-Test ---")
    batch_size = 8
    channels = 6
    num_points = 2048
    
    # Create random dummy input tensor
    dummy_input = torch.rand(batch_size, channels, num_points)
    print(f"Input shape: {dummy_input.shape}")
    
    # Initialize model
    model = PointNetRegressor(in_channels=channels)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape} (Expected: [{batch_size}])")
    
    # Verify parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print("--- Smoke-Test PASSED ---")
