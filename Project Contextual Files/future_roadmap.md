# Future Project Roadmap: From Local Sandbox to Final CFD

Based on our current implementations and the successful completion of the **Phase 5** scripts, here is the definitive, chronological roadmap for the rest of the project. This answers exactly *when* to scale, *when* to integrate advanced physics, and *when* GPUs are required.

---

## 🟢 Completed: Phase 4: Triplane VAE Training
**Goal:** Learn a continuous shape representation of the car bodies on the full 692-car subset.
* **What we did:** Triggered and resolved GPU jobs on Camber Cloud to train the Triplane VAE on the 692 point clouds and occupancy grids for 80 epochs.
* **Outcome:** Achieved a validation accuracy of **85.49%** (Recon: 0.2882, KL: 2.2451), establishing a smooth, highly generalized latent representation.
* **Compute Used:** Camber Cloud GPU (`gpu-t4` node).

## 🟢 Next: Phase 5: Latent Shape Optimization (Staged Cloud-to-Local Execution)
**Goal:** Verify our optimization mathematics actually yield a physically sound car shape.
* **What we do:** 
  1. Train the `LatentDragRegressor` fully on the **692-car** dataset using **Camber Cloud** (where all 692 point clouds and our trained `triplane_vae_best_80.pth` are located).
  2. Download the VAE checkpoint (`triplane_vae_best_80.pth`) and the trained regressor checkpoint (`latent_regressor_best_80.pth`) locally.
  3. Run the 250-step shape optimization loop (`optimize_latent_shape.py`) **locally** on CPU using a baseline car from our 100 local files to construct the morphed `.stl` geometries.
* **Outcome:** Generate an `optimized_car_step_250.stl` and visually inspect it. If it looks aerodynamic and realistic (no spikes, no collapsed roofs), we have proven our AI architecture works.
* **Compute Required:** **Cloud (for regressor training) / Local (for optimization and marching cubes)**. 

---

## 🟡 Phase 6: The Great Scaling (Multi-Config Dataset Integration)
**Goal:** Expand the AI's "knowledge" from 692 fastbacks (`F_S_WWC_WM`) to thousands of vehicles across all other body configurations (Sedans, Hatchbacks, SUVs, Notchbacks).
* **When:** Immediately after Phase 5 shape optimization is visually verified. 
* **Why wait until now?** Running full preprocessing and VAE training on thousands of multi-config meshes before proving the optimization mathematics could lead to massive computation waste.
* **What we do:**
  1. Ingest the full, multi-config dataset.
  2. Retrain the **Triplane VAE** on thousands of cars to build a massive, universal vehicle latent space.
  3. Retrain the **Latent Drag Regressor** on this new space.
* **Compute Required:** **Cloud GPUs (e.g., A100 or V100)**. 

---

## 🟠 Phase 7: Physics-Informed AI Integration (NVIDIA Modulus)
**Goal:** Upgrade our "fitness judge" from a simple scalar MLP to a full 3D airflow predictor.
* **When:** After the dataset is fully scaled (Phase 6).
* **What we do:** Replace `LatentDragRegressor` with a Modulus PINN (Physics-Informed Neural Network). Instead of just predicting a single drag number, the optimizer will "see" where the high-pressure zones are on the car's surface and morph the latent shape to relieve that pressure.
* **Compute Required:** **Multi-GPU Cloud Cluster**. Modulus is computationally demanding during training.

---

## 🔴 Phase 8: Final Ground-Truth Validation (OpenFOAM)
**Goal:** Prove to the engineering world that our AI actually works using industry-standard simulation.
* **When:** At the very end of the project, once the Modulus-powered optimizer generates its "Champion" car design.
* **What we do:** Take the final `.stl` file of the AI-designed "Champion" car and run a rigorous, high-fidelity OpenFOAM wind-tunnel simulation.
* **Compute Required:** **HPC CPU Cluster**. OpenFOAM scales exceptionally well across dozens of CPU cores (MPI).

---

### User Review Required
Does this timeline map to your expectations for the project? If you approve this progression, our immediate next action is to officially execute Phase 5 (Option 1) today.
