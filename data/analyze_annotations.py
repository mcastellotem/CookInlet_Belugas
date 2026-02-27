"""
Comprehensive annotation analysis script for whale datasets.

This script provides three analysis modes:
1. Basic summary: Quick overview of dataset statistics
2. Detailed stats: Annotation duration statistics per dataset with invalid detection
3. Visual analysis: Comprehensive visualizations saved as PNG files

Usage:
    # Basic summary
    python analyze_annotations.py --mode summary --annotations NOAA_Whales/Beluga_annotations.json

    # Detailed statistics with CSV export of invalid annotations
    python analyze_annotations.py --mode stats --annotations NOAA_Whales/Beluga_annotations.json

    # Visual analysis with plots (saves to <dataset>_plots/ directory)
    python analyze_annotations.py --mode visual --annotations NOAA_Whales/Beluga_annotations.json
"""

import argparse
import collections
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CATEGORY_MAP = {
    1: "Humpback",
    2: "Orca",
    3: "Beluga",
}


# ============================================================================
# Basic Summary Functions
# ============================================================================

def load_dataset_summary(output_path):
    """Loads the dataset and displays basic summary statistics."""
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    sounds = data["sounds"]
    annotations = data["annotations"]
    
    total_duration = sum(sound['duration'] for sound in sounds)
    total_hours = total_duration / 3600
    
    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Total species: {len(categories)}")
    print(f"Total audio recordings: {len(sounds)}")
    print(f"Total annotations: {len(annotations)}")
    print(f"Total duration: {total_hours:.2f} hours")
    print("=" * 60)


# ============================================================================
# Visual Analysis Functions
# ============================================================================

def load_dataset_with_stats(output_path, save_plots=True):
    """Loads the dataset and generates statistical summaries and histograms."""
    # Create output directory for plots
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    output_dir = os.path.join(os.path.dirname(output_path), f"{base_name}_plots")
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n📁 Saving plots to: {output_dir}\n")
    
    # Load JSON data
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare mappings and lists
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    sounds = data["sounds"]
    annotations = data["annotations"]
    
    # Sound statistics
    durations = [int(s['duration']) for s in sounds]
    sample_rates = [int(s['sample_rate']) for s in sounds]

    print("\n" + "=" * 60)
    print("Sound File Statistics")
    print("=" * 60)
    print(f"Maximum duration: {max(durations)} seconds")
    print(f"Maximum sample rate: {max(sample_rates)} Hz")
    print(f"Minimum duration: {min(durations)} seconds")
    print(f"Minimum sample rate: {min(sample_rates)} Hz")
    print(f"Unique durations: {sorted(set(durations))}")
    print(f"Unique sample rates: {sorted(set(sample_rates))}")
    print("=" * 60)

    df_anno = pd.DataFrame(annotations)
    df_anno['category_name'] = df_anno['category']
    
    # ✅ Pie Chart 1: Exact Unique Durations
    duration_counts = collections.Counter(durations)
    sorted_items = sorted(duration_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [f"{dur}" for dur, _ in sorted_items]
    sizes = [count for _, count in sorted_items]
    total = sum(sizes)
    legend_labels = [f"{label}: {count} ({count/total:.1%})"
                    for label, count in zip(labels, sizes)]
    plt.figure()
    wedges, texts = plt.pie(sizes, startangle=90)
    plt.legend(
        wedges[:5], 
        legend_labels[:5], 
        title="Duration(s)", 
        loc="lower center", 
        bbox_to_anchor=(1, 0.5))
    plt.title('Recording Duration Distribution')
    plt.axis('equal')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'recording_duration_distribution.png'), dpi=300, bbox_inches='tight')
        print("✓ Saved: recording_duration_distribution.png")
    plt.close()
    
    # ✅ Pie Chart 2: Exact Unique Sample Rates
    sample_rate_counts = collections.Counter(sample_rates)
    sorted_sr = sorted(sample_rate_counts.items(), key=lambda x: x[1], reverse=True)
    labels_sr = [f"{sr}" for sr, _ in sorted_sr]
    sizes_sr = [count for _, count in sorted_sr]
    total_sr = sum(sizes_sr)
    legend_labels_sr = [f"{label} Hz: {count} ({count/total_sr:.1%})"
                        for label, count in zip(labels_sr, sizes_sr)]

    plt.figure()
    wedges, texts = plt.pie(sizes_sr, startangle=90)
    plt.legend(wedges, legend_labels_sr, title="Sample Rate", loc="lower center", bbox_to_anchor=(1, 0.5))
    plt.title('Sample Rate Distribution')
    plt.axis('equal')
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'sample_rate_distribution.png'), dpi=300, bbox_inches='tight')
        print("✓ Saved: sample_rate_distribution.png")
    plt.close()
    
    # Calcular ancho y alto de las cajas
    df_anno['bbox_width'] = df_anno['t_max'] - df_anno['t_min']
    df_anno['bbox_height'] = df_anno['f_max'] - df_anno['f_min']
    
    # Colores personalizados para cada histograma
    colors = ['cornflowerblue', 'darkorange', 'seagreen',
            'royalblue', 'tomato', 'mediumvioletred']

    # Crear figura y ejes: 2 filas, 3 columnas
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))

    # -------- Primera fila: tiempo --------
    axs[0, 0].hist(df_anno['t_min'], bins=50, color=colors[0])
    axs[0, 0].set_title('t_min Distribution')
    axs[0, 0].set_xlabel('t_min (s)')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].set_yscale('log')

    axs[0, 1].hist(df_anno['t_max'], bins=50, color=colors[1])
    axs[0, 1].set_title('t_max Distribution')
    axs[0, 1].set_xlabel('t_max (s)')
    axs[0, 1].set_ylabel('Count')
    axs[0, 1].set_yscale('log')

    axs[0, 2].hist(df_anno['bbox_width'], bins=50, color=colors[2])
    axs[0, 2].set_title('BBox Width Distribution')
    axs[0, 2].set_xlabel('Width (s)')
    axs[0, 2].set_ylabel('Count')
    axs[0, 2].set_yscale('log')

    # -------- Segunda fila: frecuencia --------
    axs[1, 0].hist(df_anno['f_min'], bins=50, color=colors[3])
    axs[1, 0].set_title('f_min Distribution')
    axs[1, 0].set_xlabel('f_min (Hz)')
    axs[1, 0].set_ylabel('Count')

    axs[1, 1].hist(df_anno['f_max'], bins=50, color=colors[4])
    axs[1, 1].set_title('f_max Distribution')
    axs[1, 1].set_xlabel('f_max (Hz)')
    axs[1, 1].set_ylabel('Count')

    axs[1, 2].hist(df_anno['bbox_height'], bins=50, color=colors[5])
    axs[1, 2].set_title('BBox Height Distribution')
    axs[1, 2].set_xlabel('Height (Hz)')
    axs[1, 2].set_ylabel('Count')

    # Ajustar layout
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(output_dir, 'annotation_distributions.png'), dpi=300, bbox_inches='tight')
        print("✓ Saved: annotation_distributions.png")
    plt.close()

    print("\n" + "=" * 60)
    print("Annotation Frequency Statistics")
    print("=" * 60)
    print(f"Maximum f_max: {df_anno['f_max'].max():.1f} Hz")
    print(f"Minimum f_min: {df_anno['f_min'].min():.1f} Hz")
    
    mean_min = df_anno.groupby('category_name')['f_min'].mean().reset_index()
    mean_max = df_anno.groupby('category_name')['f_max'].mean().reset_index()
    top_mean_max = mean_max.nlargest(1, 'f_max')
    print(f"Category with highest avg f_max: {top_mean_max['category_name'].iloc[0]}, "
          f"avg f_max: {top_mean_max['f_max'].iloc[0]:.1f} Hz")
    top_mean_min = mean_min.nsmallest(1, 'f_min')
    print(f"Category with lowest avg f_min: {top_mean_min['category_name'].iloc[0]}, "
          f"avg f_min: {top_mean_min['f_min'].iloc[0]:.1f} Hz")
    
    top_max = df_anno.nlargest(1, 'f_max')
    print(f"Annotation with max f_max: {top_max['category'].iloc[0]}, "
          f"max f_max: {top_max['f_max'].iloc[0]:.1f} Hz")
    top_min = df_anno.nsmallest(1, 'f_min')
    print(f"Annotation with min f_min: {top_min['category'].iloc[0]}, "
          f"min f_min: {top_min['f_min'].iloc[0]:.1f} Hz")
    print("=" * 60)
    
    if save_plots:
        print(f"\n✅ All plots saved to: {output_dir}\n")


# ============================================================================
# Detailed Statistics Functions
# ============================================================================

def compute_annotation_stats(annotations_path: str) -> tuple:
    """Compute per-dataset annotation duration and frequency statistics.

    Returns a tuple of:
        - stats: dict keyed by dataset name with duration statistics
        - durations: dict of all durations per dataset
        - durations_by_location: nested dict of durations per dataset per location
        - invalid_by_location: nested dict of invalid counts per dataset per location
        - invalid_annotations: list of invalid annotation dicts
        - freq_stats: dict keyed by dataset name with frequency statistics
        - frequencies: dict of frequency data per dataset
        - frequencies_by_location: nested dict of frequencies per dataset per location
    """
    with open(annotations_path, "r") as f:
        data = json.load(f)

    annotations = data["annotations"]
    sounds = {s["id"]: s for s in data["sounds"]}

    # Collect durations and frequencies per category and detect invalid annotations
    durations: dict[str, list[float]] = defaultdict(list)
    f_mins: dict[str, list[float]] = defaultdict(list)
    f_maxs: dict[str, list[float]] = defaultdict(list)
    f_ranges: dict[str, list[float]] = defaultdict(list)
    invalid_count: dict[str, int] = defaultdict(int)
    invalid_annotations: list[dict] = []
    
    for ann in annotations:
        cat_id = ann["category_id"]
        dataset = CATEGORY_MAP.get(cat_id, f"Unknown ({cat_id})")
        sound = sounds.get(ann["sound_id"], {})
        sound_duration = sound.get("duration", 0)
        dur = ann["t_max"] - ann["t_min"]
        f_min = ann["f_min"]
        f_max = ann["f_max"]
        f_range = f_max - f_min
        
        if ann["t_min"] > sound_duration:
            invalid_count[dataset] += 1
            invalid_annotations.append({
                "startSeconds": ann["t_min"],
                "sound_duration": sound_duration,
                "sound_path": sound.get("file_name_path", ""),
                "location": sound.get("location", ""),
            })
        durations[dataset].append(dur)
        f_mins[dataset].append(f_min)
        f_maxs[dataset].append(f_max)
        f_ranges[dataset].append(f_range)

    # Collect per-location breakdowns
    durations_by_location: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    f_mins_by_location: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    f_maxs_by_location: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    f_ranges_by_location: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    invalid_by_location: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    
    for ann in annotations:
        cat_id = ann["category_id"]
        dataset = CATEGORY_MAP.get(cat_id, f"Unknown ({cat_id})")
        dur = ann["t_max"] - ann["t_min"]
        f_min = ann["f_min"]
        f_max = ann["f_max"]
        f_range = f_max - f_min
        sound = sounds.get(ann["sound_id"], {})
        location = sound.get("location", "Unknown")
        durations_by_location[dataset][location].append(dur)
        f_mins_by_location[dataset][location].append(f_min)
        f_maxs_by_location[dataset][location].append(f_max)
        f_ranges_by_location[dataset][location].append(f_range)
        if ann["t_min"] > sound.get("duration", 0):
            invalid_by_location[dataset][location] += 1

    stats = {}
    freq_stats = {}
    for dataset in sorted(durations.keys()):
        durs = np.array(durations[dataset])
        # Compute stats on valid (non-negative) durations only
        valid = durs[durs >= 0]
        stats[dataset] = {
            "count": len(durs),
            "invalid_count": int(invalid_count.get(dataset, 0)),
            "valid_count": len(valid),
            "total_sec": float(valid.sum()) if len(valid) > 0 else 0.0,
            "mean_sec": float(valid.mean()) if len(valid) > 0 else 0.0,
            "median_sec": float(np.median(valid)) if len(valid) > 0 else 0.0,
            "std_sec": float(valid.std()) if len(valid) > 0 else 0.0,
            "min_sec": float(valid.min()) if len(valid) > 0 else 0.0,
            "max_sec": float(valid.max()) if len(valid) > 0 else 0.0,
            "q25_sec": float(np.percentile(valid, 25)) if len(valid) > 0 else 0.0,
            "q75_sec": float(np.percentile(valid, 75)) if len(valid) > 0 else 0.0,
        }
        
        # Compute frequency statistics
        fmin_arr = np.array(f_mins[dataset])
        fmax_arr = np.array(f_maxs[dataset])
        frange_arr = np.array(f_ranges[dataset])
        freq_stats[dataset] = {
            "count": len(fmin_arr),
            "fmin_mean": float(fmin_arr.mean()) if len(fmin_arr) > 0 else 0.0,
            "fmin_median": float(np.median(fmin_arr)) if len(fmin_arr) > 0 else 0.0,
            "fmin_std": float(fmin_arr.std()) if len(fmin_arr) > 0 else 0.0,
            "fmin_min": float(fmin_arr.min()) if len(fmin_arr) > 0 else 0.0,
            "fmin_max": float(fmin_arr.max()) if len(fmin_arr) > 0 else 0.0,
            "fmax_mean": float(fmax_arr.mean()) if len(fmax_arr) > 0 else 0.0,
            "fmax_median": float(np.median(fmax_arr)) if len(fmax_arr) > 0 else 0.0,
            "fmax_std": float(fmax_arr.std()) if len(fmax_arr) > 0 else 0.0,
            "fmax_min": float(fmax_arr.min()) if len(fmax_arr) > 0 else 0.0,
            "fmax_max": float(fmax_arr.max()) if len(fmax_arr) > 0 else 0.0,
            "frange_mean": float(frange_arr.mean()) if len(frange_arr) > 0 else 0.0,
            "frange_median": float(np.median(frange_arr)) if len(frange_arr) > 0 else 0.0,
            "frange_std": float(frange_arr.std()) if len(frange_arr) > 0 else 0.0,
            "frange_min": float(frange_arr.min()) if len(frange_arr) > 0 else 0.0,
            "frange_max": float(frange_arr.max()) if len(frange_arr) > 0 else 0.0,
        }

    # Package frequency data for per-location analysis
    frequencies = {
        "f_min": f_mins,
        "f_max": f_maxs,
        "f_range": f_ranges,
    }
    frequencies_by_location = {
        "f_min": f_mins_by_location,
        "f_max": f_maxs_by_location,
        "f_range": f_ranges_by_location,
    }

    return (stats, durations, durations_by_location, invalid_by_location, 
            invalid_annotations, freq_stats, frequencies, frequencies_by_location)


def export_invalid_annotations(
    invalid_annotations: list[dict],
    source_csv_path: str,
    output_path: str,
) -> None:
    """Match invalid annotations to source CSV and export with original columns."""
    # Build a lookup from startSeconds -> durationSeconds from source CSV
    start_to_duration: dict[str, str] = {}
    if source_csv_path and os.path.exists(source_csv_path):
        with open(source_csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                start_to_duration[row["startSeconds"]] = row["durationSeconds"]
    else:
        if source_csv_path:
            print(f"  Source CSV not found: {source_csv_path}")
            print("  durationSeconds column will be empty in the output.")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["startSeconds", "durationSeconds", "sound_duration", "sound_path", "location"],
        )
        writer.writeheader()
        for ann in invalid_annotations:
            # Match startSeconds to source CSV (float precision match)
            start_str = str(ann["startSeconds"])
            duration_seconds = start_to_duration.get(start_str, "")
            # If exact string match fails, try float comparison
            if not duration_seconds:
                for csv_start, csv_dur in start_to_duration.items():
                    if abs(float(csv_start) - ann["startSeconds"]) < 1e-6:
                        duration_seconds = csv_dur
                        break
            writer.writerow({
                "startSeconds": ann["startSeconds"],
                "durationSeconds": duration_seconds,
                "sound_duration": ann["sound_duration"],
                "sound_path": ann["sound_path"],
                "location": ann.get("location", ""),
            })
    print(f"\nExported {len(invalid_annotations)} invalid annotations to {output_path}")


def print_stats(
    stats: dict, durations: dict, durations_by_location: dict,
    invalid_by_location: dict, freq_stats: dict, frequencies: dict,
    frequencies_by_location: dict,
) -> None:
    """Pretty-print the duration and frequency statistics."""

    # Warn about invalid annotations first
    has_invalid = any(s["invalid_count"] > 0 for s in stats.values())
    if has_invalid:
        print(
            "\n⚠  WARNING: Some annotations are invalid because their "
            "startSeconds (t_min) exceeds the audio file duration."
        )
        print(
            "   After clipping t_max to sound_duration, t_max < t_min, "
            "producing negative durations."
        )
        print("   These are EXCLUDED from the statistics below.\n")
        for dataset, s in stats.items():
            if s["invalid_count"] > 0:
                print(
                    f"   {dataset}: {s['invalid_count']} invalid annotations "
                    f"(t_min > sound_duration)"
                )
                for loc, cnt in sorted(invalid_by_location.get(dataset, {}).items()):
                    print(f"     - {loc}: {cnt}")
        print()

    header = (
        f"{'Dataset':<12} {'Count':>7} {'Invalid':>7} {'Valid':>7} {'Total(s)':>10} {'Mean(s)':>8} "
        f"{'Median(s)':>9} {'Std(s)':>7} {'Min(s)':>7} {'Max(s)':>8} "
        f"{'Q25(s)':>7} {'Q75(s)':>7}"
    )
    print("=" * len(header))
    print("Annotation Duration Statistics per Dataset (valid annotations only)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    total_count = 0
    total_valid = 0
    all_valid_durs = []
    for dataset, s in stats.items():
        total_count += s["count"]
        total_valid += s["valid_count"]
        valid = [d for d in durations[dataset] if d >= 0]
        all_valid_durs.extend(valid)
        print(
            f"{dataset:<12} {s['count']:>7,} {s['invalid_count']:>7,} {s['valid_count']:>7,} "
            f"{s['total_sec']:>10.1f} "
            f"{s['mean_sec']:>8.3f} {s['median_sec']:>9.3f} {s['std_sec']:>7.3f} "
            f"{s['min_sec']:>7.3f} {s['max_sec']:>8.3f} {s['q25_sec']:>7.3f} "
            f"{s['q75_sec']:>7.3f}"
        )

    # Overall row
    total_invalid = sum(s["invalid_count"] for s in stats.values())
    all_arr = np.array(all_valid_durs)
    print("-" * len(header))
    print(
        f"{'TOTAL':<12} {total_count:>7,} {total_invalid:>7,} {total_valid:>7,} "
        f"{all_arr.sum():>10.1f} "
        f"{all_arr.mean():>8.3f} {np.median(all_arr):>9.3f} {all_arr.std():>7.3f} "
        f"{all_arr.min():>7.3f} {all_arr.max():>8.3f} "
        f"{np.percentile(all_arr, 25):>7.3f} {np.percentile(all_arr, 75):>7.3f}"
    )
    print("=" * len(header))

    # Per-location breakdown
    for dataset in sorted(durations_by_location.keys()):
        locations = durations_by_location[dataset]
        print(f"\n--- {dataset} by location ---")
        loc_header = (
            f"  {'Location':<25} {'Count':>7} {'Valid':>7} "
            f"{'Total(s)':>10} {'Mean(s)':>8} {'Median(s)':>9}"
        )
        print(loc_header)
        print("  " + "-" * (len(loc_header) - 2))
        for loc in sorted(locations.keys()):
            durs = np.array(locations[loc])
            valid = durs[durs >= 0]
            neg = int((durs < 0).sum())
            suffix = f"  ⚠ {neg} invalid (t_min > sound_duration)" if neg > 0 else ""
            print(
                f"  {loc:<25} {len(durs):>7,} {len(valid):>7,} "
                f"{valid.sum():>10.1f} "
                f"{valid.mean():>8.3f} {np.median(valid):>9.3f}{suffix}"
            )

    # ========================================================================
    # Frequency Statistics
    # ========================================================================
    print("\n\n")
    freq_header = (
        f"{'Dataset':<12} {'Count':>7} {'fmin_mean':>10} {'fmin_med':>10} {'fmin_std':>9} "
        f"{'fmax_mean':>10} {'fmax_med':>10} {'fmax_std':>9} "
        f"{'range_mean':>11} {'range_med':>10}"
    )
    print("=" * len(freq_header))
    print("Annotation Frequency Statistics per Dataset (Hz)")
    print("=" * len(freq_header))
    print(freq_header)
    print("-" * len(freq_header))

    all_fmin = []
    all_fmax = []
    all_frange = []
    for dataset in sorted(freq_stats.keys()):
        s = freq_stats[dataset]
        all_fmin.extend(frequencies["f_min"][dataset])
        all_fmax.extend(frequencies["f_max"][dataset])
        all_frange.extend(frequencies["f_range"][dataset])
        print(
            f"{dataset:<12} {s['count']:>7,} "
            f"{s['fmin_mean']:>10.1f} {s['fmin_median']:>10.1f} {s['fmin_std']:>9.1f} "
            f"{s['fmax_mean']:>10.1f} {s['fmax_median']:>10.1f} {s['fmax_std']:>9.1f} "
            f"{s['frange_mean']:>11.1f} {s['frange_median']:>10.1f}"
        )

    # Overall frequency row
    all_fmin_arr = np.array(all_fmin)
    all_fmax_arr = np.array(all_fmax)
    all_frange_arr = np.array(all_frange)
    print("-" * len(freq_header))
    print(
        f"{'TOTAL':<12} {len(all_fmin_arr):>7,} "
        f"{all_fmin_arr.mean():>10.1f} {np.median(all_fmin_arr):>10.1f} {all_fmin_arr.std():>9.1f} "
        f"{all_fmax_arr.mean():>10.1f} {np.median(all_fmax_arr):>10.1f} {all_fmax_arr.std():>9.1f} "
        f"{all_frange_arr.mean():>11.1f} {np.median(all_frange_arr):>10.1f}"
    )
    print("=" * len(freq_header))

    # Detailed frequency statistics table
    print("\n")
    freq_detail_header = (
        f"{'Dataset':<12} {'fmin_min':>9} {'fmin_max':>9} "
        f"{'fmax_min':>9} {'fmax_max':>9} "
        f"{'range_min':>10} {'range_max':>10}"
    )
    print("=" * len(freq_detail_header))
    print("Frequency Range Details per Dataset (Hz)")
    print("=" * len(freq_detail_header))
    print(freq_detail_header)
    print("-" * len(freq_detail_header))

    for dataset in sorted(freq_stats.keys()):
        s = freq_stats[dataset]
        print(
            f"{dataset:<12} "
            f"{s['fmin_min']:>9.1f} {s['fmin_max']:>9.1f} "
            f"{s['fmax_min']:>9.1f} {s['fmax_max']:>9.1f} "
            f"{s['frange_min']:>10.1f} {s['frange_max']:>10.1f}"
        )

    print("-" * len(freq_detail_header))
    print(
        f"{'TOTAL':<12} "
        f"{all_fmin_arr.min():>9.1f} {all_fmin_arr.max():>9.1f} "
        f"{all_fmax_arr.min():>9.1f} {all_fmax_arr.max():>9.1f} "
        f"{all_frange_arr.min():>10.1f} {all_frange_arr.max():>10.1f}"
    )
    print("=" * len(freq_detail_header))

    # Per-location frequency breakdown
    for dataset in sorted(frequencies_by_location["f_min"].keys()):
        f_min_locs = frequencies_by_location["f_min"][dataset]
        f_max_locs = frequencies_by_location["f_max"][dataset]
        f_range_locs = frequencies_by_location["f_range"][dataset]
        print(f"\n--- {dataset} frequency by location ---")
        loc_freq_header = (
            f"  {'Location':<25} {'Count':>7} "
            f"{'fmin_mean':>10} {'fmax_mean':>10} {'range_mean':>11}"
        )
        print(loc_freq_header)
        print("  " + "-" * (len(loc_freq_header) - 2))
        for loc in sorted(f_min_locs.keys()):
            fmins = np.array(f_min_locs[loc])
            fmaxs = np.array(f_max_locs[loc])
            franges = np.array(f_range_locs[loc])
            print(
                f"  {loc:<25} {len(fmins):>7,} "
                f"{fmins.mean():>10.1f} {fmaxs.mean():>10.1f} {franges.mean():>11.1f}"
            )


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive annotation analysis for whale datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick summary
  python analyze_annotations.py --mode summary --annotations NOAA_Whales/Beluga_annotations.json
  
  # Detailed statistics
  python analyze_annotations.py --mode stats --annotations NOAA_Whales/Beluga_annotations.json
  
  # Visual analysis with plots (saves to <dataset>_plots/ directory)
  python analyze_annotations.py --mode visual --annotations NOAA_Whales/Beluga_annotations.json
        """
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["summary", "stats", "visual"],
        default="summary",
        help="Analysis mode: summary (basic info), stats (detailed statistics), visual (plots saved to <dataset>_plots/ directory)"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to annotations JSON file"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="invalid_duration_annotations.csv",
        help="Path to export invalid annotations CSV (stats mode only)"
    )
    parser.add_argument(
        "--source-csv",
        type=str,
        default=None,
        help="Path to source _annotations_processed.csv to recover durationSeconds (stats mode only)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save plots to files in visual mode (display only)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.annotations):
        print(f"Error: Annotations file not found: {args.annotations}")
        return

    if args.mode == "summary":
        print("\n🔍 Running basic summary analysis...\n")
        load_dataset_summary(args.annotations)
    
    elif args.mode == "stats":
        print("\n📊 Running detailed statistics analysis...\n")
        (stats, durations, durations_by_location, invalid_by_location, invalid_annotations,
         freq_stats, frequencies, frequencies_by_location) = (
            compute_annotation_stats(args.annotations)
        )
        print_stats(stats, durations, durations_by_location, invalid_by_location,
                   freq_stats, frequencies, frequencies_by_location)
        
        if invalid_annotations:
            export_invalid_annotations(
                invalid_annotations, args.source_csv, args.output_csv,
            )
    
    elif args.mode == "visual":
        save_plots = not args.no_save
        action = "and saving plots" if save_plots else "(display only)"
        print(f"\n📈 Running visual analysis {action}...\n")
        load_dataset_with_stats(args.annotations, save_plots=save_plots)


if __name__ == "__main__":
    main()
