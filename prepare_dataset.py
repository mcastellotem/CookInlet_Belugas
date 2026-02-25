"""
Dataset preparation script for Cook Inlet Belugas bioacoustics project.

Usage:
    # Full pipeline
    python prepare_dataset.py --config data/data_config.yaml

    # Run specific steps only
    python prepare_dataset.py --config data/data_config.yaml --steps stats windows

    # Available steps: stats, windows, spectrograms, splits
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Import from PytorchWildlife core library
from PytorchWildlife.utils.bioacoustics_configs import load_config, DomainConfig
from PytorchWildlife.data.bioacoustics_windows import build_windows, count_window_labels


def run_stats(config: DomainConfig) -> None:
    """Load and display dataset statistics."""
    print(f"\n{'='*60}")
    print(f"Step: Dataset Statistics")
    print(f"{'='*60}")

    annotation_path = config.paths.annotations_path
    print(f"Loading annotations from: {annotation_path}")

    if not os.path.exists(annotation_path):
        print(f"Warning: Annotations file not found: {annotation_path}")
        return

    with open(annotation_path, 'r') as f:
        data = json.load(f)

    # Dataset info
    if 'info' in data:
        print(f"\nDataset Info:")
        for key, value in data['info'].items():
            print(f"  - {key}: {value}")

    # Sound statistics
    sounds = data.get('sounds', [])
    print(f"\nSounds: {len(sounds)}")
    if sounds:
        durations = [s.get('duration', 0) for s in sounds]
        print(f"  - Total duration: {sum(durations):.1f}s ({sum(durations)/3600:.2f}h)")
        print(f"  - Mean duration: {sum(durations)/len(durations):.1f}s")
        print(f"  - Min duration: {min(durations):.1f}s")
        print(f"  - Max duration: {max(durations):.1f}s")

    # Annotation statistics
    annotations = data.get('annotations', [])
    print(f"\nAnnotations: {len(annotations)}")
    if annotations:
        categories = {}
        for ann in annotations:
            cat_id = ann.get('category_id', 0)
            categories[cat_id] = categories.get(cat_id, 0) + 1
        print(f"  - By category: {categories}")

    # Category names
    if 'categories' in data:
        print(f"\nCategories:")
        for cat in data['categories']:
            print(f"  - {cat.get('id', '?')}: {cat.get('name', 'Unknown')}")


def run_windows(config: DomainConfig) -> List[dict]:
    """Build windows from annotations."""
    print(f"\n{'='*60}")
    print(f"Step: Build Windows")
    print(f"{'='*60}")

    annotation_path = config.paths.annotations_path
    output_dir = config.paths.data_root
    os.makedirs(output_dir, exist_ok=True)

    windows_output_path = os.path.join(
        output_dir,
        config.paths.windows_file,
    )

    if os.path.exists(windows_output_path):
        print(f"Loading existing windows from: {windows_output_path}")
        with open(windows_output_path, 'r') as f:
            windows = json.load(f)
        print(f"Loaded {len(windows)} windows")
    else:
        strategy = config.audio.window_strategy
        print(f"Building windows with:")
        print(f"  - strategy: {strategy}")
        print(f"  - window_size: {config.audio.window_size_sec}s")
        print(f"  - overlap: {config.audio.overlap_sec}s")
        print(f"  - sample_rate: {config.audio.sample_rate}")
        print(f"  - datasets: {config.datasets}")
        if strategy == "balanced":
            print(f"  - negative_proportion: {config.audio.negative_proportion}")
        print(f"  - min_overlap_sec: {config.audio.min_overlap_sec}")

        windows = build_windows(
            annotation_file=annotation_path,
            window_size_sec=config.audio.window_size_sec,
            overlap_sec=config.audio.overlap_sec,
            sample_rate=config.audio.sample_rate,
            datasets_names=config.datasets,
            strategy=strategy,
            negative_proportion=config.audio.negative_proportion,
            multiclass=config.audio.multiclass,
            min_overlap_sec=config.audio.min_overlap_sec,
        )

        with open(windows_output_path, 'w') as f:
            json.dump(windows, f, indent=2)
        print(f"Saved {len(windows)} windows to: {windows_output_path}")

    # Show label distribution
    counts = count_window_labels(windows)
    print(f"\nLabel distribution: {counts}")

    return windows


def run_spectrograms(config: DomainConfig, windows: List[dict]) -> None:
    """Compute mel spectrograms using GPU."""
    # Import here to avoid loading torch unnecessarily
    from inference import compute_mel_spectrograms_gpu

    print(f"\n{'='*60}")
    print(f"Step: Compute Mel Spectrograms (GPU)")
    print(f"{'='*60}")

    spectrograms_dir = config.paths.spectrograms_dir
    os.makedirs(spectrograms_dir, exist_ok=True)

    print(f"Output directory: {spectrograms_dir}")
    print(f"Spectrogram parameters:")
    print(f"  - n_fft: {config.spectrogram.n_fft}")
    print(f"  - hop_length: {config.spectrogram.hop_length}")
    print(f"  - n_mels: {config.spectrogram.n_mels}")
    print(f"  - top_db: {config.spectrogram.top_db}")
    print(f"  - fill_noise: {config.spectrogram.fill_noise}")

    # Load annotations to get audio file paths
    with open(config.paths.annotations_path, 'r') as f:
        annotations = json.load(f)

    sounds = {s['id']: s for s in annotations['sounds']}

    # Convert windows format to include sound_path (keep legacy keys for lookup)
    inference_windows = []
    for win in windows:
        sound = sounds.get(win['sound_id'])
        if sound:
            inference_windows.append({
                'window_id': win['window_id'],
                'sound_id': win['sound_id'],
                'sound_path': sound['file_name_path'],
                'start': win['start'],
                'end': win['end'],
                'label': win.get('label'),
            })

    compute_mel_spectrograms_gpu(
        windows=inference_windows,
        sample_rate=config.audio.sample_rate,
        n_fft=config.spectrogram.n_fft,
        hop_length=config.spectrogram.hop_length,
        n_mels=config.spectrogram.n_mels,
        top_db=config.spectrogram.top_db,
        spectrograms_path=spectrograms_dir,
        save_npy=True,
        fill_noise=config.spectrogram.fill_noise,
        noise_db_std=config.spectrogram.noise_db_std,
        storage_dtype=config.spectrogram.storage_dtype,
    )

    print("Spectrogram computation complete!")


def run_splits(config: DomainConfig, windows: List[dict]) -> None:
    """Create grouped train/val/test splits, with optional binary conversion.

    Steps:
        1. Build a DataFrame from the windows mapping.
        2. Generate the spectrogram file name expected on disk and filter out
           windows whose spectrogram has not been computed yet.
        3. Split into train+val / test (grouped by ``sound_id``), then
           train / val (stratified + grouped).
        4. Save CSVs under ``<data_root>/splits/``.
    """
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

    print(f"\n{'='*60}")
    print(f"Step: Create Data Splits")
    print(f"{'='*60}")

    spectrograms_dir = config.paths.spectrograms_dir
    output_dir = config.paths.data_root

    print(f"Spectrograms directory: {spectrograms_dir}")
    print(f"Splits output directory: {output_dir}")
    print(f"Split parameters:")
    print(f"  - test_size: {config.splits.test_size}")
    print(f"  - val_size: {config.splits.val_size}")
    print(f"  - random_state: {config.splits.random_state}")

    # --- Build DataFrame from windows ---
    df = pd.DataFrame(windows)

    # Load annotations to map sound_id -> file path
    with open(config.paths.annotations_path, 'r') as f:
        annotations = json.load(f)
    sounds = {s['id']: s for s in annotations['sounds']}

    # Build spectrogram file name column
    # Try standard naming first; fall back to legacy naming per row
    df['sound_filename'] = df['sound_id'].map(
        lambda sid: os.path.splitext(
            os.path.basename(sounds[sid]['file_name_path'])
        )[0]
    )

    def _resolve_spec_name(row):
        """Return the spectrogram filename that exists on disk."""
        standard = f"{row['sound_filename']}_{row['start']}_{row['end']}.npy"
        if os.path.exists(os.path.join(spectrograms_dir, standard)):
            return standard
        legacy = (
            f"sid{row['sound_id']}_idx{row['window_id']}"
            f"_start{row['start']}_end{row['end']}_lab{row['label']}.npy"
        )
        if os.path.exists(os.path.join(spectrograms_dir, legacy)):
            return legacy
        # Default to standard (will be filtered out later)
        return standard

    df['spec_name'] = df.apply(_resolve_spec_name, axis=1)

    # Filter to rows where the spectrogram .npy exists on disk
    df['spec_exists'] = df['spec_name'].apply(
        lambda x: os.path.exists(os.path.join(spectrograms_dir, x))
    )
    print(f"\nTotal windows: {len(df)}")
    print(f"Existing spectrograms: {df['spec_exists'].sum()}")
    df = df[df['spec_exists']].drop(columns=['spec_exists'])

    if len(df) == 0:
        print("Error: No spectrograms found. Run 'spectrograms' step first.")
        return

    # --- Train+Val / Test split (grouped by sound_id) ---
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=config.splits.test_size,
        random_state=config.splits.random_state,
    )
    trainval_idx, test_idx = next(
        gss.split(df, df['label'], groups=df['sound_id'])
    )
    trainval_df = df.iloc[trainval_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # --- Train / Val split (stratified + grouped by sound_id) ---
    sgkf = StratifiedGroupKFold(
        n_splits=config.splits.n_splits, shuffle=True,
        random_state=config.splits.random_state,
    )
    train_idx, val_idx = next(
        sgkf.split(trainval_df, trainval_df['label'], trainval_df['sound_id'])
    )
    train_df = trainval_df.iloc[train_idx].copy()
    val_df = trainval_df.iloc[val_idx].copy()

    # --- Save 4-class splits ---
    splits_dir = os.path.join(output_dir, "splits_4class")
    os.makedirs(splits_dir, exist_ok=True)

    train_df.to_csv(os.path.join(splits_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test_split.csv"), index=False)

    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        label_counts = split_df['label'].value_counts().to_dict()
        print(f"  {name:5s} (n={len(split_df)}): {label_counts}")

    print(f"Saved splits to: {splits_dir}")

    # --- Derived splits ---
    # --- 3-class splits: drop noise, remap {1→0, 2→1, 3→2} ---
    three_class_dir = os.path.join(output_dir, "splits_3class")
    os.makedirs(three_class_dir, exist_ok=True)
    remap_3class = {1: 0, 2: 1, 3: 2}
    print(f"\nCreating 3-class splits (drop class 0, remap {remap_3class})")

    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        three_df = _remap_labels(
            split_df, label_map=remap_3class, drop_unmapped=True
        )
        three_df.to_csv(
            os.path.join(three_class_dir, f"{name}_split.csv"), index=False
        )
        label_counts = three_df['label'].value_counts().to_dict()
        print(f"  {name.capitalize():5s} 3-class (n={len(three_df)}): {label_counts}")

    print(f"Saved 3-class splits to: {three_class_dir}")

    # --- Binary splits: all positive classes → 1 ---
    binary_dir = os.path.join(output_dir, "splits_binary")
    os.makedirs(binary_dir, exist_ok=True)

    positive_classes = [c for c in sorted(df['label'].unique()) if c != 0]
    print(f"\nCreating binary splits (positive classes: {positive_classes})")

    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        binary_df = _convert_to_binary(split_df, positive_classes)
        binary_df.to_csv(
            os.path.join(binary_dir, f"{name}_split.csv"), index=False
        )
        label_counts = binary_df['label'].value_counts().to_dict()
        print(f"  {name.capitalize():5s} binary (n={len(binary_df)}): {label_counts}")

    print(f"Saved binary splits to: {binary_dir}")


def _convert_to_binary(
    df: pd.DataFrame,
    positive_classes: List[int],
    label_col: str = "label",
) -> pd.DataFrame:
    """Return a copy of *df* with *label_col* mapped to binary (0/1).

    Any label in *positive_classes* becomes 1; everything else becomes 0.
    """
    binary_df = df.copy()
    binary_df[label_col] = binary_df[label_col].apply(
        lambda x: 1 if x in positive_classes else 0
    )
    return binary_df


def _remap_labels(
    df: pd.DataFrame,
    label_map: dict,
    label_col: str = "label",
    drop_unmapped: bool = True,
) -> pd.DataFrame:
    """Return a copy of *df* with labels remapped according to *label_map*.

    Args:
        df: Input DataFrame.
        label_map: Mapping ``{old_label: new_label}``.
        label_col: Column containing the labels.
        drop_unmapped: If True, rows whose label is not in *label_map* are
            dropped.  Otherwise they are kept unchanged.
    """
    out = df.copy()
    if drop_unmapped:
        out = out[out[label_col].isin(label_map)]
    out[label_col] = out[label_col].map(label_map).fillna(out[label_col]).astype(int)
    return out


def load_windows_if_exists(config: DomainConfig) -> Optional[List[dict]]:
    """Load windows from file if they exist."""
    output_dir = config.paths.data_root
    windows_output_path = os.path.join(
        output_dir,
        config.paths.windows_file,
    )

    if os.path.exists(windows_output_path):
        with open(windows_output_path, 'r') as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g., config/template.yaml)"
    )
    parser.add_argument(
        "--steps", type=str, nargs="+",
        default=["stats", "windows", "spectrograms", "splits"],
        choices=["stats", "windows", "spectrograms", "splits"],
        help="Steps to run (default: all)"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Track windows (needed for some steps)
    windows = None

    # Run requested steps
    if "stats" in args.steps:
        run_stats(config)

    if "windows" in args.steps:
        windows = run_windows(config)
    elif "spectrograms" in args.steps or "splits" in args.steps:
        windows = load_windows_if_exists(config)
        if windows is None:
            print("\nError: Windows not found. Run 'windows' step first.")
            return

    if "spectrograms" in args.steps:
        if windows is None:
            windows = load_windows_if_exists(config)
        if windows is None:
            print("\nError: Windows not found. Run 'windows' step first.")
            return
        run_spectrograms(config, windows)

    if "splits" in args.steps:
        if windows is None:
            windows = load_windows_if_exists(config)
        if windows is None:
            print("\nError: Windows not found. Run 'windows' step first.")
            return
        run_splits(config, windows)

    print(f"\n{'='*60}")
    print("Dataset preparation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
