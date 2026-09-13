import sys
import argparse
import trimesh
import numpy as np

def postprocess_mesh(mesh, subdivide_levels=0, taubin_iters=0):
    """
    Applies volume-preserving subdivision and Taubin low-pass smoothing
    to eliminate Marching Cubes facet staircasing and reduce form drag.
    """
    m = mesh.copy()
    initial_vol = m.volume
    for level in range(subdivide_levels):
        print(f"Applying subdivision level {level + 1} (faces: {len(m.faces)} -> {len(m.faces)*4})...")
        m = m.subdivide()
    if taubin_iters > 0:
        print(f"Applying Taubin low-pass smoothing ({taubin_iters} iterations, lamb=0.3, nu=-0.31)...")
        trimesh.smoothing.filter_taubin(m, lamb=0.3, nu=-0.31, iterations=taubin_iters)
        vol_change = (m.volume - initial_vol) / (initial_vol + 1e-8) * 100
        print(f"Taubin smoothing complete. Volume change: {vol_change:+.2f}%")
    return m

def denormalize(input_stl, ref_stl, output_stl, subdivide_levels=0, taubin_iters=0):
    print(f"Loading AI mesh: {input_stl}")
    mesh_ai = trimesh.load(input_stl)
    
    print(f"Loading reference mesh: {ref_stl}")
    mesh_ref = trimesh.load(ref_stl)
    
    # Calculate extents
    ai_extents = mesh_ai.extents
    ref_extents = mesh_ref.extents
    
    print(f"AI mesh extents: {ai_extents}")
    print(f"Ref mesh extents: {ref_extents}")
    
    # Scale factors
    scale = ref_extents / ai_extents
    print(f"Applying scale factors: {scale}")
    
    # Create transform matrix
    transform = np.eye(4)
    transform[0, 0] = scale[0]
    transform[1, 1] = scale[1]
    transform[2, 2] = scale[2]
    
    mesh_ai.apply_transform(transform)
    
    # Shift to match the reference bounding box min
    shift = mesh_ref.bounds[0] - mesh_ai.bounds[0]
    print(f"Applying translation shift: {shift}")
    
    translation = np.eye(4)
    translation[:3, 3] = shift
    mesh_ai.apply_transform(translation)
    
    if subdivide_levels > 0 or taubin_iters > 0:
        mesh_ai = postprocess_mesh(mesh_ai, subdivide_levels, taubin_iters)
        # Re-verify alignment with reference ground
        z_shift = mesh_ref.bounds[0][2] - mesh_ai.bounds[0][2]
        if abs(z_shift) > 1e-4:
            ground_align = np.eye(4)
            ground_align[2, 3] = z_shift
            mesh_ai.apply_transform(ground_align)
    
    print(f"Final AI mesh bounds:\n{mesh_ai.bounds}")
    print(f"Reference bounds:\n{mesh_ref.bounds}")
    
    print(f"Saving to {output_stl}")
    mesh_ai.export(output_stl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denormalize and post-process 3D meshes for OpenFOAM CFD")
    parser.add_argument("--input", required=True, help="Path to input normalized STL")
    parser.add_argument("--ref", required=True, help="Path to reference CAD STL for bounding box scaling")
    parser.add_argument("--output", required=True, help="Path to output 1:1 physical STL")
    parser.add_argument("--subdivide", type=int, default=0, help="Number of surface subdivision levels (default: 0)")
    parser.add_argument("--smooth_taubin", type=int, default=0, help="Taubin smoothing iterations to reduce facet kinks (default: 0)")
    args = parser.parse_args()
    
    denormalize(args.input, args.ref, args.output, args.subdivide, args.smooth_taubin)
