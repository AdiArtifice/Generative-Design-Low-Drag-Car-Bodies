import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

class ClosedLoopSurrogate(nn.Module):
    """
    Differentiable wrapper around LatentDragRegressor for Phase 8.
    
    1. Optimizes relative Delta CdA rather than absolute surrogate output:
       Delta CdA(z) = Regressor(z, c) - Regressor(z_0, c)
       
    2. Enforces active directional CFD feedback constraints:
       Penalizes displacement in directions where CFD proved the surrogate's
       predicted drag reduction was an illusion (Delta CdA_CFD > Delta CdA_surr).
    """
    def __init__(
        self,
        base_regressor: nn.Module,
        z_initial: torch.Tensor,
        class_idx: Optional[torch.Tensor] = None,
        cfd_constraints: Optional[List[Dict[str, Any]]] = None,
        cfd_penalty_weight: float = 2.0
    ):
        super(ClosedLoopSurrogate, self).__init__()
        self.regressor = base_regressor
        self.register_buffer("z_initial", z_initial.detach().clone())
        self.class_idx = class_idx
        self.cfd_penalty_weight = cfd_penalty_weight
        
        # Calculate baseline predicted drag
        with torch.no_grad():
            self.baseline_pred = self.regressor(self.z_initial, class_idx=self.class_idx).item()
            
        # Register directional constraint vectors
        self.constraint_directions = []
        self.constraint_errors = []
        self.constraint_norms = []
        
        if cfd_constraints:
            for c in cfd_constraints:
                v = c["v"].to(z_initial.device).float()
                err = float(c["error_delta"])
                norm_v = float(c["norm_v"])
                self.constraint_directions.append(v)
                self.constraint_errors.append(err)
                self.constraint_norms.append(norm_v)
                
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Returns the physics-corrected relative objective:
        Loss(z) = Delta CdA(z) + CFD_Repulsive_Barrier(z)
        """
        pred_abs = self.regressor(z, class_idx=self.class_idx)
        delta_pred = pred_abs - self.baseline_pred
        
        # Add directional CFD penalty if movement aligns with falsified gradient directions
        if not self.constraint_directions:
            return delta_pred
            
        disp = z - self.z_initial
        cfd_penalty = torch.tensor(0.0, device=z.device)
        
        for v, err, norm_v in zip(self.constraint_directions, self.constraint_errors, self.constraint_norms):
            # Projection of displacement along falsified direction v
            proj = torch.sum(disp * v) / (norm_v + 1e-8)
            # Quadratic penalty if moving along the falsified direction
            alignment = torch.relu(proj)
            # Normalized penalty scaled by the empirical CFD error discrepancy
            cfd_penalty = cfd_penalty + (alignment / (norm_v + 1e-8)) ** 2 * err
            
        return delta_pred + self.cfd_penalty_weight * cfd_penalty

    def get_raw_prediction(self, z: torch.Tensor) -> float:
        with torch.no_grad():
            return self.regressor(z, class_idx=self.class_idx).item()
