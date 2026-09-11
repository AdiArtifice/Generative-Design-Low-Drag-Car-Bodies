import sys
import argparse
import trimesh
import numpy as np

def denormalize(input_stl, ref_stl, output_stl):
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
    
    print(f"Final AI mesh bounds:\n{mesh_ai.bounds}")
    print(f"Reference bounds:\n{mesh_ref.bounds}")
    
    print(f"Saving to {output_stl}")
    mesh_ai.export(output_stl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    denormalize(args.input, args.ref, args.output)
