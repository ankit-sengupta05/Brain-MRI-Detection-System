# Brain MRI Tumor Classification — Three-Model Comparison

A PyTorch benchmark comparing three architectures on brain MRI tumor classification: a CNN trained from scratch, and two transfer-learning models (ResNet18, MobileNetV3-Small) with frozen backbones. All three are trained under identical conditions (same data splits, optimizer, schedule, and epoch budget) so their results are directly comparable.

## Classes

The dataset is a 4-class brain MRI classification task:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

## Models compared

| Model | Type | Trainable params |
|---|---|---|
| **Custom CNN** (`BrainCNN`) | Trained from scratch — 4 conv blocks (32→64→128→256 channels) with BatchNorm, ReLU, MaxPool, global average pool, dropout(0.4) head | All layers |
| **ResNet18** | Transfer learning — ImageNet-pretrained backbone frozen, final `fc` layer replaced and fine-tuned | Final layer only |
| **MobileNetV3-Small** | Transfer learning — ImageNet-pretrained backbone frozen, final classifier layer replaced and fine-tuned | Final layer only |

## Results (50 epochs, held-out test set)

| Model | Test Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Mean Confidence |
|---|---|---|---|---|---|
| **Custom CNN** | **89.88%** | 90.81% | 89.88% | 89.61% | 0.853 |
| ResNet18 | 85.81% | 86.14% | 85.81% | 85.48% | 0.782 |
| MobileNetV3-Small | 84.62% | 85.72% | 84.62% | 84.10% | 0.811 |

The from-scratch CNN outperformed both frozen-backbone transfer-learning models on this run, likely because only the final layer was fine-tuned for ResNet18/MobileNetV3 — the frozen ImageNet features aren't fully adapted to MRI-specific textures. Unfreezing more layers (or fine-tuning the full backbone at a lower learning rate) is a natural next experiment; see [Notes & possible improvements](#notes--possible-improvements).

Per-epoch metrics, confusion matrices, and dashboard plots are generated for every model — see [Outputs](#outputs).

## Project structure

```
.
├── Three_Model_Testing_for_comparison.ipynb   # main notebook
├── Training/                                  # ImageFolder-structured training data
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── Testing/                                    # ImageFolder-structured test data
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── saved_models/                               # created at runtime, see Outputs
    └── run_<timestamp>_<uuid>/
```

`Training/` and `Testing/` must each contain one subfolder per class (standard `torchvision.datasets.ImageFolder` layout), and both must expose the same four class folders.

## Requirements

- Python 3.9+
- CUDA-capable GPU (the notebook raises an error if CUDA isn't visible — see [GPU memory management](#gpu-memory-management) for why)
- PyTorch with CUDA support, matching your installed CUDA toolkit
- `torchvision`, `numpy`, `matplotlib`, `Pillow`

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # match to your CUDA version
pip install numpy matplotlib pillow
```

> The notebook was developed against a Windows virtualenv (`DL Project (PyTorch cu130)` kernel); the CUDA build/version isn't load-bearing for the code itself — just point the kernel at any Python environment with a matching CUDA-enabled PyTorch install.

## Running

1. Place `Training/` and `Testing/` folders (ImageFolder layout, see above) beside the notebook.
2. Run cells in this order:
   1. **Setup cell** — imports, seeding, CUDA check, dataset/dataloader creation.
   2. **Preprocessing inspection cell** — prints the augmentation pipeline and class distribution, shows a sample augmented batch.
   3. **Model definitions cell** — defines `BrainCNN` and builds all three models (`models_to_train` dict), keeps everything on CPU until training.
   4. **Shared training-functions cell** — defines `metrics_from_predictions`, `evaluate_model`, `show_live_dashboard`, `save_metrics_json`, and a baseline `train_one_model`.
   5. **GPU-safe trainer override cell** — replaces `train_one_model` with a version that keeps only the active model on the GPU and uses mixed precision. **Must run after step 4 and before the three algorithm cells below.**
   6. **Algorithm 1 — Custom CNN** (50 epochs)
   7. **Algorithm 2 — ResNet18** (50 epochs)
   8. **Algorithm 3 — MobileNetV3** (50 epochs)
3. Run the **evaluation/comparison cell** to print classification reports, plot confusion matrices, per-model training curves, and a 3-model comparison grid across accuracy, precision, recall, F1, confidence, uncertainty, and learning rate.
4. Use `predict_brain_mri(image_path, model_name)` to run inference on a single image with any of the three trained models still held in memory.

## Training configuration

| Setting | Value |
|---|---|
| Epochs | 50 |
| Optimizer | Adam, weight_decay=1e-4 |
| Learning rate | 1e-3, with `ReduceLROnPlateau` (factor 0.5, patience 3, min 1e-6) |
| Loss | Cross-entropy with label smoothing (0.1) |
| Gradient clipping | max norm 1.0 |
| Batch size | 16 |
| Image size | 224×224 |
| Train augmentation | resize → random horizontal flip → random rotation (±10°) → color jitter (brightness/contrast 0.15) → normalize (ImageNet mean/std) |
| Test augmentation | resize → normalize only |
| Seed | 42 (Python, NumPy, PyTorch, CUDA) |
| Precision | Mixed precision (`torch.amp`) on CUDA |

## GPU memory management

The notebook is written for an **8 GB GPU** and trains three models in sequence rather than in parallel: before each model's training loop starts, every *other* model in `models_to_train` is moved to CPU, and only the active model is moved to CUDA, followed by `gc.collect()` + `torch.cuda.empty_cache()`. This keeps peak VRAM usage to one model at a time even though all three live in memory for the later comparison/inference cells.

## Outputs

Each run creates a unique folder under `saved_models/run_<timestamp>_<8-char-uuid>/` containing, per model:

- `<model>_epoch_<NNN>.png` — a 4-panel live dashboard (loss, accuracy, precision/recall/F1, confusion matrices) saved after every epoch
- `<model>_<run_id>_metrics.json` — full per-epoch history (all metrics, both confusion matrices, learning rate) after every epoch
- `<model>_<run_id>_best.pth` — the model checkpoint with the best test accuracy seen during training (`model_state_dict`, `class_names`, `image_size`, `model_name`, `run_id`)
- `<model>_<run_id>_complete_history.png` — final 6-panel training-curve summary (from the comparison cell)

Plus, once all three models finish:

- `all_models_<run_id>_metrics.json` — combined history for all three models
- `all_models_<run_id>_comparison.png` — side-by-side comparison across all three models for every tracked metric

## Inference

```python
prediction, confidence = predict_brain_mri(
    "path/to/some_mri_image.jpg",
    model_name="ResNet18",  # or "Custom CNN" / "MobileNetV3"
)
```

Displays the image with the predicted class and confidence, and returns `(predicted_class: str, confidence: float)`. Any of the three trained models can be selected by name as long as they're still loaded in `models_to_train`.

## Local dashboard

The React dashboard loads the three trained checkpoints in `saved_models/`, reads their saved per-epoch metrics and test-set confusion matrices, and exposes image inference through a local FastAPI service. The Predictions page accepts JPEG, PNG, and WebP images up to 10 MB. Uploaded images are processed in memory and are not saved by the API. Prediction reports include the selected model, predicted training label, all four class probabilities, the run identifier, and its held-out test report.

The results are a research demonstration only. They are not a medical diagnosis or a substitute for evaluation by a qualified healthcare professional.

### One-time setup

- Install Node.js (including npm) and Python 3.10 or newer.
- From the project directory, create/activate a Python environment and install the backend requirements:

  ```powershell
  py -3 -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  ```

  `torch` and `torchvision` can be large downloads. If you need a CUDA-enabled PyTorch build, install the matching PyTorch packages for your CUDA version before installing the remaining requirements.
- Install the frontend packages once with `npm install` (the launcher also does this when `node_modules/` is absent).

### Start

Double-click `start_dashboard.bat` or run it from the project directory. It checks the Python backend imports and installs `requirements.txt` if a dependency is missing, then starts the API at `http://127.0.0.1:8000`, starts the Vite frontend at `http://127.0.0.1:5173`, and opens the dashboard in your browser. The first install may take several minutes because PyTorch is a large package. Keep both terminal windows open.

Running `npm run dev` alone starts only the frontend; it does not start the prediction API. Use `start_dashboard.bat` to start both services. Alternatively, start the backend separately with `.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000` before running `npm run dev`.

The API loads the latest complete run containing all three `*_best.pth` checkpoints and its `all_models_*_metrics.json` report; a missing or incompatible checkpoint is reported by the backend startup log rather than silently substituting a model.

### Local API

- `GET /api/health` — backend and loaded-model status.
- `GET /api/models` — run metadata, epoch histories, confusion matrices, and per-class test reports for all models.
- `POST /api/predict` — multipart fields `model_name` (`Custom CNN`, `ResNet18`, or `MobileNetV3`) and `image`.

The browser dashboard proxies these routes to the local API. A prediction JSON report can be downloaded from the Predictions page, and all saved model reports can be downloaded from the sidebar.

## Notes & possible improvements

- Only the final layer of ResNet18 and MobileNetV3 is fine-tuned (backbone frozen) — unfreezing the last few backbone blocks, or the whole network at a lower LR, is a likely way to close the gap with the from-scratch CNN.
- `NUM_WORKERS = 0` for the dataloaders — increasing this can speed up data loading on multi-core machines.
- Class imbalance isn't explicitly corrected for (no weighted sampler / weighted loss) beyond label smoothing; worth checking per-class support in the classification report if a specific class underperforms.
- Reported metrics (precision/recall/F1) are macro-averaged across all four classes.
