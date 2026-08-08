# Upgrade to Conditional VAE (C-VAE) Pipeline

This plan outlines the steps required to transition the generative AI pipeline from a standard Triplane VAE to a Conditional Triplane VAE (C-VAE), which will enable the system to handle multiple vehicle topologies (e.g., Fastback, Estateback, Notchback) without suffering from geometric blur or mode collapse. 

## User Review Required

> [!IMPORTANT]
> Please review the proposed architecture changes below. Using a learned `nn.Embedding` layer is recommended over raw one-hot encoding because it offers greater flexibility if we decide to add more diverse metadata later (such as specific OEM styles or smooth vs non-smooth configurations).
>
> Please confirm if you approve this approach so we can proceed with execution.

## Proposed Changes

### Project Documentation
We will update the project's foundational documents to reflect the C-VAE scaling strategy.

#### [MODIFY] [README.md](file:///c:/Aditya%20Files/Projects/Generative%20Design%20for%20Low-Drag%20Car%20Bodies/local%20subset/README.md)
*   Update the `Phase 4: Triplane VAE Generative Model` section to describe the transition to a Conditional VAE.
*   Update the `Future Roadmap` section to explain how the C-VAE handles the 4,000+ car scaling.

#### [MODIFY] [future_roadmap.md](file:///c:/Aditya%20Files/Projects/Generative%20Design%20for%20Low-Drag%20Car%20Bodies/local%20subset/Project%20Contextual%20Files/future_roadmap.md)
*   Modify `Phase 6: The Great Scaling` to explicitly mention that the C-VAE will condition the latent space on vehicle body style labels (e.g., F, E, N) using an `nn.Embedding` layer.

---

### Dataset Upgrades
We need to extract the categorical condition and feed it into the models.

#### [MODIFY] [dataset.py](file:///c:/Aditya%20Files/Projects/Generative%20Design%20for%20Low-Drag%20Car%20Bodies/local%20subset/src/dataset.py)
*   Modify `VehicleOccupancyDataset` and `VehiclePointCloudDataset` to parse the `config` column from `metadata.csv`.
*   Extract the primary body shape character (e.g., 'F' from `F_S_WWC_WM`).
*   Map the string label to an integer class index (e.g., `{'F': 0, 'E': 1, 'N': 2, 'S': 3, 'H': 4}`).
*   Return this `class_idx` tensor as part of the dataset tuple for each sample.

---

### Model Architecture Upgrades
We will modify the VAE to conditionally accept the class embeddings.

#### [MODIFY] [triplane.py](file:///c:/Aditya%20Files/Projects/Generative%20Design%20for%20Low-Drag%20Car%20Bodies/local%20subset/src/models/triplane.py)
*   **TriplaneVAE:**
    *   Add an `nn.Embedding(num_classes, embed_dim)` layer.
*   **PointNetEncoder:**
    *   Accept the conditional embedding vector.
    *   Concatenate it with the global max-pooled features (`[B, 512]`) before the `fc_mu` and `fc_logvar` linear layers.
*   **TriplaneDecoder:**
    *   Accept the conditional embedding vector alongside the latent vector `z`.
    *   Concatenate them (`z_cond = torch.cat([z, emb], dim=1)`) before mapping to the initial `8x8` planes.

#### [MODIFY] [vae.py](file:///c:/Aditya%20Files/Projects/Generative%20Design%20for%20Low-Drag%20Car%20Bodies/local%20subset/src/models/vae.py)
*   Apply the same `nn.Embedding` and concatenation modifications to the standard `PointNetVAE` to keep it synchronized with the Triplane VAE approach.

---

### Training Script Upgrades
We will pass the new condition tensor during training.

#### [MODIFY] [train_triplane.py](file:///c:/Aditya%20Files/Projects/Generative%20Design%20for%20Low-Drag%20Car%20Bodies/local%20subset/scripts/train_triplane.py)
*   Update the training and validation loops to unpack the `class_idx` from the dataloader.
*   Pass `class_idx` into the `model(pc, query_pts, class_idx)` forward pass.
*   Add a `--num_classes` and `--embed_dim` argument to `parse_args()`.

## Verification Plan

### Automated Tests
*   Run the smoke-test blocks built into `src/dataset.py`, `src/models/triplane.py`, and `src/models/vae.py` using `python src/dataset.py`, etc.
*   Verify that the input and output dimensions correctly account for the new concatenated condition sizes.

### Manual Verification
*   Run the fast CPU smoke test for training: `python scripts/train_triplane.py --smoke_test`. 
*   Verify that the loss successfully computes and backpropagates without shape mismatch errors.
