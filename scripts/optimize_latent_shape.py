#!/usr/bin/env python
import os
import json
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from skimage.measure import marching_cubes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.triplane import TriplaneVAE
from src.models.latent_regressor import LatentDragRegressor
from src.dataset import VehiclePointCloudDataset
from src.cfd_evidence_store import CFDEvidenceStore
from src.surrogate_correction import ClosedLoopSurrogate
from scripts.denormalize_mesh import denormalize

def extract_mesh(vae, z, output_path, device, grid_res=64, threshold=0.5, c_emb=None):
    # Generates dense grid coordinates
    min_val, max_val = -0.6, 0.6
    x = np.linspace(min_val, max_val, grid_res, dtype=np.float32)
    y = np.linspace(min_val, max_val, grid_res, dtype=np.float32)
    z_coords = np.linspace(min_val, max_val, grid_res, dtype=np.float32)
    
    xv, yv, zv = np.meshgrid(x, y, z_coords, indexing='ij')
    grid_coords = np.stack([xv, yv, zv], axis=-1)
    flat_coords = grid_coords.reshape(-1, 3)
    
    # Inference
    with torch.no_grad():
        plane_xy, plane_xz, plane_yz = vae.decoder(z, c_emb=c_emb)
        
        batch_size = 16384
        occupancies = []
        for i in range(0, len(flat_coords), batch_size):
            batch_coords = flat_coords[i : i + batch_size]
            batch_tensor = torch.tensor(batch_coords, dtype=torch.float32).unsqueeze(0).to(device)
            
            grid_xy = (batch_tensor[..., [0, 1]] * 2.0).unsqueeze(2)
            grid_xz = (batch_tensor[..., [0, 2]] * 2.0).unsqueeze(2)
            grid_yz = (batch_tensor[..., [1, 2]] * 2.0).unsqueeze(2)
            
            feat_xy = F.grid_sample(plane_xy, grid_xy, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(-1)
            feat_xz = F.grid_sample(plane_xz, grid_xz, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(-1)
            feat_yz = F.grid_sample(plane_yz, grid_yz, mode='bilinear', padding_mode='zeros', align_corners=True).squeeze(-1)
            
            logits = vae.occupancy_mlp(feat_xy, feat_xz, feat_yz, batch_tensor)
            probs = torch.sigmoid(logits).squeeze(0)
            occupancies.append(probs.cpu().numpy())
            
    occupancy_flat = np.concatenate(occupancies, axis=0)
    occupancy_grid = occupancy_flat.reshape(grid_res, grid_res, grid_res)
    
    # Mask out-of-bounds
    mask_x = (x >= -0.5) & (x <= 0.5)
    mask_y = (y >= -0.25) & (y <= 0.25)
    mask_z = (z_coords >= -0.18) & (z_coords <= 0.18)
    mask_3d = mask_x[:, None, None] & mask_y[None, :, None] & mask_z[None, None, :]
    occupancy_grid[~mask_3d] = 0.0
    
    if occupancy_grid.max() < threshold:
        print(f"[Warning] Max occupancy {occupancy_grid.max():.4f} is less than threshold.")
        return False
        
    padded_grid = np.pad(occupancy_grid, pad_width=1, mode='constant', constant_values=0.0)
    verts, faces, normals, values = marching_cubes(volume=padded_grid, level=threshold)
    
    spacing = (max_val - min_val) / (grid_res - 1)
    verts_physical = (verts - 1.0) * spacing + min_val
    
    mesh = trimesh.Trimesh(vertices=verts_physical, faces=faces, vertex_normals=normals)
    if mesh.volume < 0:
        mesh.invert()
        
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mesh.export(output_path)
    return True

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def optimize(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")
    
    # 1. Load VAE
    print(f"Loading Triplane VAE from {args.vae_path}...")
    if not os.path.exists(args.vae_path):
        print(f"Error: {args.vae_path} not found.")
        sys.exit(1)
        
    vae_state = torch.load(args.vae_path, map_location=device)
    has_vae_class_emb = "class_emb.weight" in vae_state
    num_classes_vae = args.num_classes if has_vae_class_emb else 0
    embed_dim_vae = args.embed_dim if has_vae_class_emb else 0
    
    vae = TriplaneVAE(
        in_channels=6, 
        latent_dim=256, 
        plane_channels=16, 
        plane_resolution=args.plane_res, 
        num_classes=num_classes_vae, 
        embed_dim=embed_dim_vae
    ).to(device)
    vae.load_state_dict(vae_state)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
        
    # 2. Load Latent Regressor
    print(f"Loading Latent Drag Regressor from {args.regressor_path}...")
    if not os.path.exists(args.regressor_path):
        # Fallback to smoke model if default file not found and smoke is available
        if args.regressor_path == "models/latent_regressor_best.pth" and os.path.exists("models/latent_regressor_smoke.pth"):
            print("Warning: Using smoke model as latent_regressor_best.pth is not found.")
            args.regressor_path = "models/latent_regressor_smoke.pth"
        else:
            print(f"Error: {args.regressor_path} not found.")
            sys.exit(1)
            
    reg_state = torch.load(args.regressor_path, map_location=device)
    has_reg_class_emb = "class_emb.weight" in reg_state
    num_classes_reg = args.num_classes if has_reg_class_emb else 0
    embed_dim_reg = args.embed_dim if has_reg_class_emb else 0
    
    regressor = LatentDragRegressor(
        latent_dim=256, 
        num_classes=num_classes_reg, 
        embed_dim=embed_dim_reg
    ).to(device)
    regressor.load_state_dict(reg_state)
    regressor.eval()
    for param in regressor.parameters():
        param.requires_grad = False
        
    # 3. Load baseline car
    print("Loading dataset to find a baseline car...")
    dataset = VehiclePointCloudDataset(
        csv_path="metadata/metadata.csv",
        scales_path="metadata/target_scales.json",
        split=None,
        num_points=2048,
        pc_dir=args.pc_dir,
        normalize_targets=False
    )
    
    # Filter for cars that actually exist locally
    valid_mask = dataset.df['pointcloud_path'].apply(os.path.exists)
    valid_df = dataset.df[valid_mask]
    
    if len(valid_df) == 0:
        print("Error: No cars found locally! Cannot perform local optimization.")
        sys.exit(1)
        
    if args.car_id:
        target_car_id = args.car_id
    else:
        # Check if an STL file exists in temp_raw_stl
        raw_stl_dir = "temp_raw_stl"
        stl_files = [f for f in os.listdir(raw_stl_dir) if f.endswith(".stl")] if os.path.exists(raw_stl_dir) else []
        if len(stl_files) > 0:
            target_car_id = os.path.splitext(stl_files[0])[0]
            print(f"Auto-detected baseline STL car in '{raw_stl_dir}': {stl_files[0]} -> Car ID: {target_car_id}")
        else:
            target_car_id = None
            
    if target_car_id:
        target_rows = valid_df[valid_df['id'] == target_car_id]
        if len(target_rows) == 0:
            print(f"Error: Specified or detected car_id '{target_car_id}' not found locally or in metadata.")
            sys.exit(1)
        baseline_idx = target_rows.index[0]
    else:
        # Find highest drag car among the valid local cars
        baseline_idx = valid_df['drag_area'].idxmax()
        
    row = valid_df.loc[baseline_idx]
    
    # Get the integer index for the dataset loader
    dataset_idx = dataset.df.index.get_loc(baseline_idx)
    baseline_id = row['id']
    
    print(f"Selected baseline car: {baseline_id} with original drag_area = {row['drag_area']:.4f} m^2")
    
    features, class_idx, targets = dataset[dataset_idx]
    features = features.unsqueeze(0).to(device)
    class_idx = class_idx.unsqueeze(0).to(device)
    
    # Extract baseline latent vector (explicitly detached & cloned)
    with torch.no_grad():
        c_emb = vae.class_emb(class_idx) if vae.class_emb is not None else None
        z_initial, _ = vae.encoder(features, c_emb=c_emb)
        z_initial = z_initial.detach().clone()
        
    # Initial prediction
    with torch.no_grad():
        initial_pred_drag = regressor(z_initial, class_idx=class_idx).item()
        
    print(f"Baseline Predicted Drag Area: {initial_pred_drag:.4f} m^2")
    
    # Compute drag floor from dataset (lowest drag the regressor was trained on)
    drag_floor = valid_df['drag_area'].min()
    print(f"Dataset drag_area floor: {drag_floor:.4f} m^2 (optimizer will not target below this)")
    
    # 4. Optimization Loop
    z_opt = torch.nn.Parameter(z_initial.clone())
    optimizer = torch.optim.Adam([z_opt], lr=args.lr)
    
    version_dir = f"optimization_output_{args.version}" if args.version != "v1" else "optimization_output"
    out_dir = args.out_dir if args.out_dir else (f"{version_dir}/{baseline_id}" if args.car_id else version_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    # Check Phase 8 Closed-Loop CFD feedback
    body_map = {'F': 'Fastback', 'E': 'Estateback', 'N': 'Notchback'}
    body_name = body_map.get(str(row.get('body_type', 'F')), 'Fastback')
    use_cfd = args.use_cfd_feedback or (args.version == "v3")
    
    if use_cfd:
        store = CFDEvidenceStore(args.evidence_store)
        filter_car_id = None if args.cross_vehicle_cfd else baseline_id
        constraints = store.get_directional_constraints(body_name, baseline_id=filter_car_id)
        print(f"\n[Phase 8 Closed-Loop Mode]")
        print(f"Loaded {len(constraints)} active directional CFD constraint(s) for {body_name} (baseline: {filter_car_id}).")
        closed_loop_model = ClosedLoopSurrogate(
            base_regressor=regressor,
            z_initial=z_initial,
            class_idx=class_idx,
            cfd_constraints=constraints,
            cfd_penalty_weight=args.cfd_penalty_weight
        )
    else:
        closed_loop_model = None
    
    # Export step 0 (baseline)
    print(f"Exporting initial mesh (Step 0) to {out_dir}...")
    success = extract_mesh(vae, z_opt, f"{out_dir}/optimized_car_step_0.stl", device, grid_res=args.grid_res, c_emb=c_emb)
    if not success:
        print("[Warning] Initial mesh reconstruction failed.")
    
    drag_floor_tensor = torch.tensor(drag_floor, dtype=torch.float32, device=device)
    
    print("\nStarting Latent Space Optimization...")
    print(f"Explicit Latent Trust Region: R_trust = {args.trust_radius if args.trust_radius else 'None (unconstrained)'}")
    if closed_loop_model is not None:
        print("Objective: Relative Delta CdA + Directional CFD Repulsive Barrier")
    else:
        print("Objective: Absolute Surrogate CdA Minimization")
        
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        
        if closed_loop_model is not None:
            drag_loss = closed_loop_model(z_opt)
            pred_drag = regressor(z_opt, class_idx=class_idx)
        else:
            pred_drag = regressor(z_opt, class_idx=class_idx)
            pred_drag_clamped = torch.clamp(pred_drag, min=drag_floor_tensor)
            drag_loss = pred_drag_clamped
            
        similarity_penalty = torch.sum((z_opt - z_initial) ** 2)  # Squared L2
        
        # Total loss formula: drag objective + proximity penalty
        loss = drag_loss + args.lambda_reg * similarity_penalty
        loss.backward()
        optimizer.step()
        
        # 1. Explicit Latent Trust Region: Project onto hard L2 ball B(z_initial, trust_radius)
        if args.trust_radius is not None and args.trust_radius > 0:
            with torch.no_grad():
                delta = z_opt.data - z_initial
                dist = torch.norm(delta)
                if dist > args.trust_radius:
                    z_opt.data.copy_(z_initial + delta * (args.trust_radius / dist))
                    
        # 2. Coordinate-wise clamp to preserve latent manifold bounds
        with torch.no_grad():
            z_opt.data.clamp_(-args.z_clamp, args.z_clamp)
            
        latent_dist = torch.norm(z_opt.data - z_initial).item()
        
        if step % 10 == 0 or step == 1:
            trust_str = f"{args.trust_radius:.2f}" if args.trust_radius is not None else "inf"
            delta_val = pred_drag.item() - initial_pred_drag
            print(f"Step {step:03d} | Loss: {loss.item():.4f} | Drag: {pred_drag.item():.4f} m^2 (Delta: {delta_val:+.4f}) | Dist: {latent_dist:.4f}/{trust_str} | Penalty: {similarity_penalty.item():.4f}")
            
        if step % 50 == 0 or step == args.steps:
            output_path = f"{out_dir}/optimized_car_step_{step}.stl"
            print(f"  -> Exporting intermediate mesh: {output_path}")
            success = extract_mesh(vae, z_opt, output_path, device, grid_res=args.grid_res, threshold=0.5, c_emb=c_emb)
            if not success:
                print(f"[Warning] Mesh reconstruction failed at step {step}.")
            
    # Calculate final reduction in raw physical units
    with torch.no_grad():
        final_pred_drag = regressor(z_opt, class_idx=class_idx).item()
        reduction = (initial_pred_drag - final_pred_drag) / initial_pred_drag * 100 if initial_pred_drag != 0 else 0
        
    print("\nOptimization Complete!")
    print(f"Final Predicted Drag Area: {final_pred_drag:.4f} m^2 (Baseline: {initial_pred_drag:.4f} m^2)")
    print(f"Theoretical Drag Reduction: {reduction:.2f}%")
    print(f"Final Latent Distance from Baseline: {latent_dist:.4f} (Trust Radius: {args.trust_radius})")
    
    if args.max_reduction_guardrail is not None and reduction > args.max_reduction_guardrail:
        print(f"[Warning] Predicted reduction of {reduction:.2f}% exceeds the {args.max_reduction_guardrail:.1f}% engineering guardrail!")
        
    print(f"Check the '{out_dir}' folder for STL files.")
    
    # Save optimization summary for downstream tools (e.g., visualizer)
    summary = {
        "version": args.version,
        "baseline_id": baseline_id,
        "baseline_body_type": str(row.get('body_type', 'unknown')),
        "baseline_ground_truth_drag_area": float(row['drag_area']),
        "baseline_predicted_drag_area": float(initial_pred_drag),
        "final_predicted_drag_area": float(final_pred_drag),
        "reduction_percent": float(reduction),
        "steps": args.steps,
        "lambda_reg": args.lambda_reg,
        "trust_radius": float(args.trust_radius) if args.trust_radius is not None else None,
        "cfd_feedback_enabled": bool(closed_loop_model is not None),
        "cfd_constraints_count": len(closed_loop_model.constraint_directions) if closed_loop_model is not None else 0,
        "final_latent_distance": float(torch.norm(z_opt - z_initial).item()),
        "final_latent_norm": float(torch.norm(z_opt).item()),
        "lr": args.lr,
        "z_clamp": args.z_clamp,
    }
    z_init_file = os.path.abspath(f"{out_dir}/z_initial.pt")
    z_opt_file = os.path.abspath(f"{out_dir}/z_opt_{args.steps}.pt")
    summary["z_initial_path"] = z_init_file
    summary["z_opt_path"] = z_opt_file

    # Save latent tensors for evidence tracking
    torch.save(z_initial.cpu(), z_init_file)
    torch.save(z_opt.data.cpu(), z_opt_file)
    
    summary_path = f"{out_dir}/optimization_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved optimization summary to {summary_path}")

    if args.stage_cfd:
        print("\n--- CFD Mesh Staging ---")
        final_stl = f"{out_dir}/optimized_car_step_{args.steps}.stl"
        step0_stl = f"{out_dir}/optimized_car_step_0.stl"
        ref_stl = f"temp_raw_stl/{baseline_id}.stl"
        staged_stl = f"{out_dir}/optimized_car_step_{args.steps}_1to1_smooth.stl"
        step0_staged = f"{out_dir}/optimized_car_step_0_1to1_smooth.stl"
        
        if not os.path.exists(ref_stl):
            print(f"[Error] Reference CAD {ref_stl} not found! Cannot automatically denormalize.")
        else:
            if os.path.exists(step0_stl) and not os.path.exists(step0_staged):
                try:
                    print("Denormalizing and smoothing baseline Step 0 mesh...")
                    denormalize(
                        input_stl=step0_stl,
                        ref_stl=ref_stl,
                        output_stl=step0_staged,
                        subdivide_levels=args.subdivide,
                        taubin_iters=args.smooth_taubin
                    )
                    print(f"Staged baseline step 0 mesh: {step0_staged}")
                except Exception as e:
                    print(f"[Warning] Baseline step 0 staging failed: {e}")

            if not os.path.exists(final_stl):
                print(f"[Error] Final mesh {final_stl} not found! Did extraction fail?")
            else:
                print(f"Denormalizing and smoothing AI champion mesh...")
                try:
                    denormalize(
                        input_stl=final_stl,
                        ref_stl=ref_stl,
                        output_stl=staged_stl,
                        subdivide_levels=args.subdivide,
                        taubin_iters=args.smooth_taubin
                    )
                    print(f"Successfully staged physical CFD-ready mesh: {staged_stl}")
                except Exception as e:
                    print(f"[Error] Denormalization failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--car_id", type=str, default=None, help="ID of baseline car to optimize (e.g. E_S_WWC_WM_014)")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save output meshes and summary")
    parser.add_argument("--version", type=str, default="v3", help="Optimization version tag (v1, v2, v3)")
    parser.add_argument("--use_cfd_feedback", action="store_true", help="Enable Phase 8 active CFD directional feedback constraints")
    parser.add_argument("--cross_vehicle_cfd", action="store_true", help="Allow cross-vehicle CFD constraints within the same body category")
    parser.add_argument("--evidence_store", type=str, default=None, help="Path to cfd_evidence_store.json (default: metadata/cfd_evidence_store.json)")
    parser.add_argument("--cfd_penalty_weight", type=float, default=1.5, help="Weight for directional CFD constraint repulsive barrier (default: 1.5)")
    parser.add_argument("--steps", type=int, default=250, help="Number of optimization steps")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for Adam optimizer")
    parser.add_argument("--lambda_reg", type=float, default=0.01, help="Squared-L2 penalty weight to preserve core structure")
    parser.add_argument("--trust_radius", type=float, default=0.75, help="Explicit hard trust region radius in latent space around baseline (default: 0.75)")
    parser.add_argument("--max_reduction_guardrail", type=float, default=15.0, help="Engineering guardrail threshold in percent (default: 15.0)")
    parser.add_argument("--vae_path", type=str, default="models/triplane_vae_best_128.pth", help="Path to pre-trained VAE weights")
    parser.add_argument("--regressor_path", type=str, default="models/latent_regressor_best_128.pth", help="Path to trained regressor weights")
    parser.add_argument("--plane_res", type=int, default=128, help="Triplane resolution of VAE")
    parser.add_argument("--grid_res", type=int, default=64, help="Marching Cubes grid resolution for mesh extraction")
    parser.add_argument("--pc_dir", type=str, default="pointclouds_hybrid", help="Directory containing point clouds")
    parser.add_argument("--num_classes", type=int, default=3, help="Number of vehicle classes for C-VAE")
    parser.add_argument("--embed_dim", type=int, default=16, help="Category embedding dimension")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--z_clamp", type=float, default=3.0, help="Clamp radius for latent vector (keeps z within training manifold)")
    parser.add_argument("--stage_cfd", action="store_true", help="Automatically denormalize and smooth the final mesh for CFD")
    parser.add_argument("--subdivide", type=int, default=1, help="Subdivision levels for CFD staging (default: 1)")
    parser.add_argument("--smooth_taubin", type=int, default=4, help="Taubin smoothing iterations for CFD staging (default: 4)")
    args = parser.parse_args()
    
    optimize(args)
