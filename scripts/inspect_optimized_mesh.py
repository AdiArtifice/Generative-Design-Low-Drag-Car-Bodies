import trimesh
import os

print("--- Inspecting Optimized Meshes ---")
for step in [0, 50, 100, 150, 200, 250]:
    path = f"optimization_output/optimized_car_step_{step}.stl"
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        continue
    mesh = trimesh.load(path)
    print(f"\nStep {step}:")
    print(f"  - Watertight: {mesh.is_watertight}")
    print(f"  - Vertices: {len(mesh.vertices)}")
    print(f"  - Faces: {len(mesh.faces)}")
    print(f"  - Volume: {mesh.volume:.6f}")
    print(f"  - Area: {mesh.area:.6f}")
