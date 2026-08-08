# 3D Modeling Roadmap & Implementation Plan

We are transitioning from the Data Preprocessing Phase to the **Modeling & Deep Learning Phase**. We follow a rigorous, 3-Phase "Baseline-First" strategy that proves the superiority of 3D spatial learning.

---

## Current Status

* **Phase 1: Tabular Baseline Benchmark** — **[COMPLETED]**
  * Train and test sets verified. Linear/classical models hit a hard wall of **49% - 57% $R^2$** on test data.
* **Phase 2: PyTorch Dataset & Smoke-Test** — **[COMPLETED]**
  * Built `src/dataset.py` with dynamic 2,048-point downsampling for CPU laptops and target Z-score scaling. Fully tested via pytest.
* **Phase 3: 3D PointNet Regressor & Training** — **[ACTIVE / NEXT STEP]**
  * We will construct the neural network model and its training script to train and evaluate the spatial baseline.

---

## Detailed Modeling Phases

### Phase 1: Tabular Baseline Benchmark (Completed)
* **Goal:** Prove the "performance floor" and extract feature importance using classical ML.
* **Script:** [tabular_baseline.py](file:///c:/Aditya%20Files/Projects/Generative%20Design%20for%20Low-Drag%20Car%20Bodies/local%20subset/scripts/tabular_baseline.py)
* **Findings:** RandomForest and GradientBoosting regressors overfit training data but score ~53% and ~57% $R^2$ on the test set, verifying the spatial feature gap.

### Phase 2: PyTorch Dataset & Smoke-Test (Completed)
* **Goal:** Build the deep learning data engine to stream 3D point clouds.
* **Script:** [dataset.py](file:///c:/Aditya%20Files/Projects/Generative%20Design%20for%20Low-Drag%20Car%20Bodies/local%20subset/src/dataset.py)
* **Status:** Passed automated pytest checks. Returns inputs of shape `[Batch, 6, NumPoints]`.

### Phase 3: 3D PointNet Regressor & Training Loop (Next Step)
* **Goal:** Implement the Deep Learning architecture capable of "seeing" the 3D car and training it.
* **Architecture File:** #### [NEW] [src/models/pointnet.py](file:///c:/Aditya%20Files/Projects/Generative%20Design%20for%20Low-Drag%20Car%20Bodies/local%20subset/src/models/pointnet.py)
  * **1D Convolutions:** Shared-weight 1D CNNs mapping inputs `[6, N]` $\rightarrow$ `[64, N]` $\rightarrow$ `[128, N]` $\rightarrow$ `[512, N]`.
  * **Global Max Pooling:** Collapses dimension `N` to get a global feature vector of shape `[512]`.
  * **MLP Regression Head:** Fully connected layers (`512` $\rightarrow$ `256` $\rightarrow$ `64` $\rightarrow$ `1`) with ReLU activations and dropout to predict normalized `drag_area` or `cd`.
* **Training Script:** #### [NEW] [scripts/train_pointnet.py](file:///c:/Aditya%20Files/Projects/Generative%20Design%20for%20Low-Drag%20Car%20Bodies/local%20subset/scripts/train_pointnet.py)
  * **Lightweight CPU Training Loop:** A clean script using Adam optimizer and MSE loss.
  * **Configurable Parameters:** Allows setting number of epochs (e.g. 5 epochs for smoke-testing, 50 epochs for full CPU training) and learning rate.
  * **Model Checkpointing:** Saves the best performing model state (`.pth`) to `models/pointnet_best.pth`.
  * **Evaluation:** Plots validation/test curves and compares final test $R^2$ and MAE directly against Phase 1 metrics to prove PointNet's spatial advantage.

---

## User Review Required

> [!NOTE]
> Does the active Phase 3 detailed plan for writing both `src/models/pointnet.py` and `scripts/train_pointnet.py` look correct? If approved, I will begin implementing them to build and run our first deep learning 3D model!
