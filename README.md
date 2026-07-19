# Wrist-TDCNN Interaction

#### Lightweight wearable-sensor classification and real-time BLE cursor interaction

## Overview

This repository provides the code used to train and evaluate a compact one-dimensional convolutional neural network for wearable-sensor gesture classification, together with a Bluetooth Low Energy (BLE) demonstration that converts real-time predictions into cursor actions.

Two related, but **not interchangeable**, model variants are included:

| Workflow | Model input | Output | Default checkpoint |
|---|---:|---:|---|
| Offline flexion/extension classification | 5 channels × 12 time steps | 10 classes | `checkpoints/released/best_model_classification.pth` |
| Real-time cursor control | 2 channels × 12 time steps | 4 classes: rest, left, right, pinch | `checkpoints/released/best_model_cursor_control.pth` |

The 10-class checkpoint cannot be loaded into the 4-class cursor model because their input and output layer dimensions differ.

Released checkpoints associated with the manuscript are kept under `checkpoints/released/`. Newly trained classification checkpoints are written separately to `checkpoints/trained/`, so training does not modify the released manuscript checkpoint.

## Repository structure

```text
wrist-tdcnn-interaction/
├── README.md
├── environments.yaml
├── LICENSE
├── assets/
│   ├── background.png
│   ├── chrome.png
│   ├── cursor.png
│   └── model_architecture.png
├── checkpoints/
│   ├── released/
│   │   ├── best_model_classification.pth
│   │   └── best_model_cursor_control.pth
│   └── trained/                     # generated classification checkpoints
├── configs/
│   ├── flexion_extension_classification_parameters.py
│   └── cursor_control_parameters.py
├── models/
│   ├── flexion_extension_classification_model.py
│   └── ultralight_model.py
├── preprocessing/
│   ├── flexion_extension_classification_preprocessing.py
│   └── cursor_control_preprocessing.py
├── pipelines/
│   ├── flexion_extension_classification_train.py
│   ├── flexion_extension_classification_offline_inference.py
│   └── cursor_control.py
└── examples/
    ├── data_classification/
    │   ├── train/
    │   ├── valid/
    │   └── test/
    └── raw_data_cursor_control.csv
```

## Setup

Clone the repository, enter its root directory, and create the provided environment.

```bash
git clone <REPOSITORY_URL>
cd wrist-tdcnn-interaction
conda env create -f environments.yaml
conda activate <ENVIRONMENT_NAME>
```

Run all commands below from the repository root. The pipeline scripts import `models`, `preprocessing`, and `configs` from that root, so expose it on `PYTHONPATH` before using direct `.py` execution.

Linux or macOS:

```bash
export PYTHONPATH="$(pwd)"
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

The environment must include PyTorch, NumPy, pandas, Bleak, Pillow, and Tkinter support. A CUDA-capable GPU is used automatically when available; CPU execution is also supported.

## Data preparation

Place the classification CSV files in the following directories:

```text
examples/data_classification/train/
examples/data_classification/valid/
examples/data_classification/test/
```

Each CSV file must contain these columns:

```text
Counts, GestureIdentifier, Thumb, Index, Middle, Ring, Little
```

The default preprocessing performs per-channel min–max normalization, creates 120-sample windows with stride 1, divides each window into 10-sample blocks, and computes a trimmed mean after discarding the three smallest and three largest values in each block. Consequently, each five-channel window is reduced to an input tensor of shape `5 × 12`.

Raw gesture identifiers are mapped to class indices as follows:

```text
51→0, 52→1, 61→2, 62→3, 71→4,
72→5, 81→6, 82→7, 91→8, 92→9
```

Dataset paths and preprocessing/training parameters can be changed in `configs/flexion_extension_classification_parameters.py`.

## Training the classification model

The training pipeline writes newly generated checkpoints to `checkpoints/trained/`. The directory is created automatically if it does not exist.

```bash
python pipelines/flexion_extension_classification_train.py
```

The principal output is:

```text
checkpoints/trained/best_model_classification.pth
```

<!-- Upload the manuscript figure to assets/model_architecture.png. -->
<p align="center">
  <img src="assets/model_architecture.png" width="900" alt="Architecture of the compact 1D-CNN classification model">
</p>
<p align="center">
  <em>Figure 1. Architecture of the compact 1D-CNN classification model.</em>
</p>

The checkpoint is updated whenever validation accuracy matches or exceeds the previous best value. If `checkpoints/trained/best_model_classification.pth` already exists, a new training run overwrites it. Copy the generated checkpoint elsewhere before rerunning training when the previous result must be retained.

With the default configuration, training can also save up to five models whose validation accuracy reaches the early-stopping threshold:

```text
checkpoints/trained/top_model_<accuracy>_<epoch>.pth
```

The released manuscript checkpoint under `checkpoints/released/` is never used as a training output and is therefore not overwritten by this pipeline.

The current training script does not print an epoch-by-epoch log. For a manuscript-reproduction run, record the exact configuration, random seed, validation trajectory, and software environment together with the generated checkpoint.

## Offline evaluation

By default, offline inference loads the released manuscript checkpoint:

```text
checkpoints/released/best_model_classification.pth
```

It evaluates all files in `examples/data_classification/test/`.

```bash
python pipelines/flexion_extension_classification_offline_inference.py
```

To evaluate the checkpoint produced by a new training run, select it explicitly:

```bash
python pipelines/flexion_extension_classification_offline_inference.py --checkpoint checkpoints/trained/best_model_classification.pth
```

Any checkpoint supplied through `--checkpoint` must be compatible with the five-channel, ten-class classification model.

## Real-time cursor-control demonstration

The cursor-control pipeline requires a BLE sensor device and a **separate two-channel, four-class checkpoint**. The classification training script above does not generate this cursor-control checkpoint.

1. Set `ble_characteristic_uuid`, `ble_device_address`, and `selected_channels` in `configs/cursor_control_parameters.py`.
2. Place the released cursor-control checkpoint at `checkpoints/released/best_model_cursor_control.pth`.
   The cursor-control pipeline always loads this checkpoint through `params.model_path`; it does not accept a command-line checkpoint override.
3. Ensure the UI assets are accessible. The current code opens `background.png`, `chrome.png`, and `cursor.png` relative to the working directory. For the repository structure shown above, update those paths to `assets/background.png`, `assets/chrome.png`, and `assets/cursor.png` before release.
4. Start the demonstration from the repository root:

```bash
python pipelines/cursor_control.py
```

At startup, the application connects to the BLE device, displays a 45-second waiting screen, and then collects 15 seconds of data to estimate per-channel normalization ranges. Live predictions are subsequently interpreted as rest (`0`), left (`1`), right (`2`), or pinch (`3`). Selected BLE channels are also written asynchronously to the directory specified by `CSV_LOG_DIR` in `pipelines/cursor_control.py`.

## Model architecture

Both variants use the same compact 1D-CNN backbone:

```text
Input
  → Conv1d
  → BatchNorm1d
  → ReLU
  → MaxPool1d
  → Conv1d
  → BatchNorm1d
  → ReLU
  → AdaptiveAvgPool1d(1)
  → Flatten
  → Linear classifier
```

The offline model uses `5→16→32` convolutional channels and a 10-class output layer (2,250 trainable parameters). The cursor model uses `2→16→32` convolutional channels and a 4-class output layer (1,908 trainable parameters).

## Reproducibility notes

- Checkpoints under `checkpoints/released/` are the fixed artifacts associated with the manuscript. Training writes only to `checkpoints/trained/`.
- `checkpoints/trained/best_model_classification.pth` is a generated working artifact and is overwritten when training is run again. Preserve it separately when multiple training results must be compared.
- The current preprocessing code computes min–max values independently for the training, validation, and test splits. For strict deployment-style evaluation, estimate normalization statistics on the training set and reuse them for validation and test data.
- Files within each split are concatenated before window generation. Use continuous recordings only, or modify the loader to prevent windows from crossing file boundaries.
- No global random seed is set in the current training pipeline. Exact numerical reproduction requires deterministic seed and backend settings.
- A complete real-time reproduction requires publication of the compatible cursor-control checkpoint or the corresponding training pipeline.

## Related code release

The organization of this reproducibility guide follows the concise overview–setup–training–evaluation structure used in the public repository for *A simplified wearable device powered by a generative EMG network for hand-gesture recognition and gait prediction*:

- Paper: https://www.nature.com/articles/s44460-025-00002-2
- Code: https://github.com/nature-sensors/GenENet

## License

See `LICENSE` for the terms governing use and redistribution of this code.

## Citation

Citation metadata will be added when the associated manuscript becomes publicly available.

```bibtex
@article{<CITATION_KEY>,
  title   = {<MANUSCRIPT_TITLE>},
  author  = {<AUTHORS>},
  journal = {Nature Sensors},
  year    = {<YEAR>}
}
```

## Contact

**[CORRESPONDING AUTHOR]** — [EMAIL ADDRESS]
