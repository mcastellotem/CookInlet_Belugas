# Adaptive acoustic monitoring for endangered Cook Inlet beluga whales in complex soundscapes

This repository contains the code and experiment pipelines to reproduce the results in “Adaptive acoustic monitoring for endangered Cook Inlet beluga whales in complex soundscapes.” It implements an end-to-end workflow for passive acoustic monitoring (PAM), including spectrogram generation, a two-stage deep learning architecture for cetacean signal detection and species classification (beluga, humpback, killer whale), and an active-learning loop for domain adaptation to new soundscapes. The repository is organized to support training, evaluation, inference on long-duration recordings, and replication of the final framework proposed in the manuscript.

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

```bash
# Binary classification (1st Stage)
python train.py --config config_binary.yaml \
    --train_csv splits_binary/train_split.csv \
    --val_csv splits_binary/val_split.csv \
    --test_csv splits_binary/test_split.csv \
```

```bash
# 3class classification (2nd Stage)
python train.py --config config_3class.yaml \
    --train_csv splits_3class/train_split.csv \
    --val_csv splits_3class/val_split.csv \
    --test_csv splits_3class/test_split.csv \
```

### 4. Active learning training and testing

### 5. Inference in complete Johnson River dataset
