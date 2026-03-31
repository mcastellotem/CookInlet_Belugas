# Adaptive acoustic monitoring for endangered Cook Inlet beluga whales in complex soundscapes

This repository contains the code and experiment pipelines to reproduce the results in "Adaptive acoustic monitoring for endangered Cook Inlet beluga whales in complex soundscapes." It implements an end-to-end workflow for passive acoustic monitoring (PAM), including spectrogram generation, a two-stage deep learning architecture for cetacean signal detection and species classification (beluga, humpback, killer whale), and an active-learning loop for domain adaptation to new soundscapes. The repository is organized to support training, evaluation, inference on long-duration recordings, and replication of the final framework proposed in the manuscript.

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Audio data and annotations are available upon request to the corresponding author. Place all audio files and the annotations JSON under the `data/` directory.

The preparation pipeline is controlled by a single YAML config file (`data/data_config.yaml`) that defines audio parameters, spectrogram settings, and split ratios. It runs four steps in order:

| Step | Description |
|------|-------------|
| `stats` | Print dataset statistics (number of sounds, total duration, annotation counts per category). |
| `windows` | Slide fixed-size windows over each audio file and assign labels based on annotation overlap. Saves a JSON mapping. |
| `spectrograms` | Compute GPU-accelerated mel spectrograms for every window and save them as `.npy` files. |
| `splits` | Create grouped train/val/test splits (stratified by label, grouped by sound). It generates `splits_4class/`, `splits_3class/` (positive classes only, remapped), and `splits_binary/` (all positive classes mapped to 1) variants. |

```bash
# Run the full pipeline
python prepare_dataset.py --config data/data_config.yaml

# Run specific steps only (e.g., statistics and windows)
python prepare_dataset.py --config data/data_config.yaml --steps stats windows

# Compute spectrograms only (windows must already exist)
python prepare_dataset.py --config data/data_config.yaml --steps spectrograms

# Create splits only (windows and spectrograms must already exist)
python prepare_dataset.py --config data/data_config.yaml --steps splits
```

After running the full pipeline the `data/` directory will contain:

```
data/
├── data_config.yaml
├── annotations_combined.json
├── windows_mapping_0.4overlap.json
├── mel_spectrograms_multiclass/   # .npy spectrograms
├── splits_4class/                 # 4-class CSVs (noise + 3 species)
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── splits_3class/                 # 3-class CSVs (species only)
│   └── ...
└── splits_binary/                 # Binary CSVs (whale vs. no-whale)
    └── ...
```

### 3. Train Base Models

All model and training parameters are defined in YAML configs under `configs/`. Only runtime arguments (split paths, checkpoint path) are passed via CLI. Config files can be referenced by filename alone — the script resolves them from the `configs/` directory automatically.

```bash
# Binary classification — Stage 1 (whale / no-whale), ResNet18
python train.py --config config_binary.yaml \
    --train_csv data/base_splits/splits_binary/train_split.csv \
    --val_csv data/base_splits/splits_binary/val_split.csv \
    --test_csv data/base_splits/splits_binary/test_split.csv
```

```bash
# 3-class species classification — Stage 2 (humpback / orca / beluga), ResNet34
python train.py --config config_3class.yaml \
    --train_csv data/base_splits/splits_3class/train_split.csv \
    --val_csv data/base_splits/splits_3class/val_split.csv \
    --test_csv data/base_splits/splits_3class/test_split.csv
```

```bash
# 4-class - Stage 1 (no-whale + 3 species), ResNet34
python train.py --config config_4class_75.yaml \
    --train_csv data/base_splits/splits_4class/train_split.csv \
    --val_csv data/base_splits/splits_4class/val_split.csv \
    --test_csv data/base_splits/splits_4class/test_split.csv
```

```bash
# 4-class — Stage 2 (no-whale + 3 species), ResNet34
python train.py --config config_4class_25.yaml \
    --train_csv data/base_splits/splits_4class_25/train_split.csv \
    --val_csv data/base_splits/splits_4class_25/val_split.csv \
    --test_csv data/base_splits/splits_4class_25/test_split.csv
```

**Evaluate from a saved checkpoint (no training):**

```bash
python train.py --config config_binary.yaml \
    --test_csv data/splits_binary/test_split.csv \
    --ckpt_path checkpoints/binary/best.ckpt
```

Checkpoints and test prediction CSVs are saved to `checkpoints/<model_name>/`. The `--predict_only` flag skips metric computation and only exports a predictions CSV, which is useful when the test label space differs from the model's label space.

### 4. Active Learning Training and Testing

```bash
python train.py --config config_binary.yaml \
    --train_csv data/tuxedni_splits/splits_binary/train_split.csv \
    --val_csv data/tuxedni_splits/splits_binary/val_split.csv \
    --test_csv data/tuxedni_splits/splits_binary/test_split.csv \
    --ckpt_path checkpoints/binary/best.ckpt --finetune
```

```bash
python train.py --config config_binary.yaml \
    --train_csv data/johnson_splits/splits_binary/train_split.csv \
    --val_csv data/johnson_splits/splits_binary/val_split.csv \
    --test_csv data/johnson_splits/splits_binary/test_split.csv \
    --ckpt_path checkpoints/binary/best.ckpt --finetune
```

### 5. Inference on Long-Duration Recordings

`inference.py` builds sliding windows over raw audio files, computes mel spectrograms on the fly, runs the loaded checkpoint, and exports per-window predictions as a CSV. It supports binary and multiclass modes.

**Audio source options:**
- **Folder**: scans for all audio files, builds windows automatically, saves a `<dataset>_windows.json`
- **JSON file**: loads pre-built windows from a previous run
- **CSV file**: loads windows from a CSV with spectrogram paths

```bash
# Binary inference from an audio folder
python inference.py \
    --checkpoint checkpoints/binary/best.ckpt \
    --audios_source /path/to/audio \
    --dataset my_recording

# Multiclass inference
python inference.py \
    --checkpoint checkpoints/3class/best.ckpt \
    --audios_source /path/to/audio \
    --num_classes 3 --class_names Humpback Orca Beluga \
    --dataset my_recording

# Config-driven inference (reads audio/spectrogram params from YAML)
python inference.py \
    --config configs/config_binary.yaml \
    --checkpoint checkpoints/binary/best.ckpt \
    --audios_source /path/to/audio
```

Key parameters (all have defaults; use `--config` to inherit them from a training config):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--window_size_sec` | 5.0 | Sliding window length in seconds |
| `--overlap_sec` | 4.0 | Overlap between consecutive windows |
| `--sample_rate` | 48000 | Audio sample rate |
| `--n_fft` | 2048 | FFT size |
| `--hop_length` | 512 | Spectrogram hop length |
| `--n_mels` | 224 | Number of mel bands |
| `--temperature` | 1.0 | Temperature for confidence scaling |
| `--batch_size` | 64 | Inference batch size |

Results are saved to `inference/<dataset>/`:
- `<binary|multiclass>_inference_results.csv` — per-window predictions with probabilities
- `per_second_results.csv` — overlap-weighted aggregation to 1-second resolution (binary only)

### 6. Model Comparison

`compare_models.py` reads the `test_split_with_predictions.csv` files written by `train.py` and evaluates three cascade strategies without reloading any model.

| Mode | Description |
|------|-------------|
| **4-Class** | Single 4-class model |
| **Binary+3-Class** | Binary gate → 3-class species classifier |
| **Binary+4-Class** | Binary gate → 4-class model |

```bash
# Cascade comparison (reads from default checkpoint paths)
python compare_models.py

# Custom prediction CSV paths
python compare_models.py \
    --pred_binary        checkpoints/binary/test_split_with_predictions.csv \
    --pred_3class        checkpoints/3class/test_split_with_predictions.csv \
    --pred_4class        checkpoints/4class/test_split_with_predictions.csv \
    --pred_4class_2stage checkpoints/4class/test_split_with_predictions.csv

# Compare two runs of the same model type
python compare_models.py \
    --compare checkpoints/binary/test_split_with_predictions_v1.csv \
              checkpoints/binary/test_split_with_predictions_v2.csv \
    --names v1 v2

# Save comparison table to CSV
python compare_models.py --output comparison_results.csv
```
