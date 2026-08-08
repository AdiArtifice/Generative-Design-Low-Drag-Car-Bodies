import os
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    step_0_path = "optimization_output/optimized_car_step_0.stl"
    step_250_path = "optimization_output/optimized_car_step_250.stl"
    
    if not os.path.exists(step_0_path) or not os.path.exists(step_250_path):
        print("Error: Optimized STL files not found. Run the optimization script first.")
        return
        
    print("Loading meshes...")
    mesh_0 = trimesh.load(step_0_path)
    mesh_250 = trimesh.load(step_250_path)
    
    v0 = mesh_0.vertices
    v250 = mesh_250.vertices
    
    print(f"Loaded Step 0 (vertices: {v0.shape[0]})")
    print(f"Loaded Step 250 (vertices: {v250.shape[0]})")
    
    print("Generating side-by-side Matplotlib visualization...")
    fig = plt.figure(figsize=(15, 6))
    
    # 1. Plot Step 0 (Baseline)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sc1 = ax1.scatter(v0[:, 0], v0[:, 1], v0[:, 2], c=v0[:, 2], cmap='viridis', s=1, alpha=0.5)
    ax1.set_title("Step 0: Baseline Car Body", fontsize=14)
    ax1.set_xlim(-0.6, 0.6)
    ax1.set_ylim(-0.6, 0.6)
    ax1.set_zlim(-0.6, 0.6)
    ax1.view_init(elev=20, azim=-60)
    ax1.axis('off')
    
    # 2. Plot Step 250 (Optimized)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    sc2 = ax2.scatter(v250[:, 0], v250[:, 1], v250[:, 2], c=v250[:, 2], cmap='viridis', s=1, alpha=0.5)
    ax2.set_title("Step 250: Optimized Car Body (-7.27% Drag)", fontsize=14)
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_zlim(-0.6, 0.6)
    ax2.view_init(elev=20, azim=-60)
    ax2.axis('off')
    
    plt.suptitle("Aerodynamic Shape Optimization: Initial vs. Optimized Mesh", fontsize=16, y=0.95)
    plt.tight_layout()
    
    print("Displaying visualization window...")
    plt.show()

if __name__ == "__main__":
    main()
