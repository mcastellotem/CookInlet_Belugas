"""
Annotation duration statistics per dataset (Humpback, Orca, Beluga).

Detects invalid annotations whose startSeconds exceeds the audio file
duration (t_min > sound_duration) and exports them to a CSV.

Usage:
    python annotation_stats.py --annotations data/annotations_combined.json
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


CATEGORY_MAP = {
    1: "Humpback",
    2: "Orca",
    3: "Beluga",
}


def compute_annotation_stats(annotations_path: str) -> dict:
    """Compute per-dataset annotation duration statistics.

    Returns a dict keyed by dataset name with sub-dicts containing:
        count, total_sec, mean_sec, median_sec, std_sec, min_sec, max_sec,
        q25_sec, q75_sec.
    """
    with open(annotations_path, "r") as f:
        data = json.load(f)

    annotations = data["annotations"]
    sounds = {s["id"]: s for s in data["sounds"]}

    # Collect durations per category and detect invalid annotations
    durations: dict[str, list[float]] = defaultdict(list)
    invalid_count: dict[str, int] = defaultdict(int)
    invalid_annotations: list[dict] = []
    for ann in annotations:
        cat_id = ann["category_id"]
        dataset = CATEGORY_MAP.get(cat_id, f"Unknown ({cat_id})")
        sound = sounds.get(ann["sound_id"], {})
        sound_duration = sound.get("duration", 0)
        dur = ann["t_max"] - ann["t_min"]
        if ann["t_min"] > sound_duration:
            invalid_count[dataset] += 1
            # Original annotation: startSeconds = t_min,
            # durationSeconds = t_max_original - t_min (before clipping)
            # t_max in JSON is already clipped to sound_duration,
            # so recover original: durationSeconds = (t_min + dur_original) - t_min
            # We don't have the original t_max, but we can recover durationSeconds
            # from the source CSV via t_min match. Instead, store what we have.
            invalid_annotations.append({
                "startSeconds": ann["t_min"],
                "sound_duration": sound_duration,
                "sound_path": sound.get("file_name_path", ""),
                "location": sound.get("location", ""),
            })
        durations[dataset].append(dur)

    # Also collect per-location breakdowns
    durations_by_location: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    invalid_by_location: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for ann in annotations:
        cat_id = ann["category_id"]
        dataset = CATEGORY_MAP.get(cat_id, f"Unknown ({cat_id})")
        dur = ann["t_max"] - ann["t_min"]
        sound = sounds.get(ann["sound_id"], {})
        location = sound.get("location", "Unknown")
        durations_by_location[dataset][location].append(dur)
        if ann["t_min"] > sound.get("duration", 0):
            invalid_by_location[dataset][location] += 1

    stats = {}
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

    return stats, durations, durations_by_location, invalid_by_location, invalid_annotations


def export_invalid_annotations(
    invalid_annotations: list[dict],
    source_csv_path: str,
    output_path: str,
) -> None:
    """Match invalid annotations to source CSV and export with original columns.

    Reads the source Beluga CSV to recover the original durationSeconds for
    each invalid annotation, then writes a CSV with:
        startSeconds, durationSeconds, sound_duration, sound_path
    """
    # Build a lookup from startSeconds -> durationSeconds from source CSV
    start_to_duration: dict[str, str] = {}
    if source_csv_path and os.path.exists(source_csv_path):
        with open(source_csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                start_to_duration[row["startSeconds"]] = row["durationSeconds"]
    else:
        print(f"  Source CSV not found: {source_csv_path}")
        print("  durationSeconds column will be empty in the output.")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["startSeconds", "durationSeconds", "sound_duration", "sound_path"],
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
            })
    print(f"\nExported {len(invalid_annotations)} invalid annotations to {output_path}")


def print_stats(
    stats: dict, durations: dict, durations_by_location: dict,
    invalid_by_location: dict,
) -> None:
    """Pretty-print the statistics."""

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


def main():
    parser = argparse.ArgumentParser(
        description="Compute annotation duration statistics per dataset"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/annotations_combined.json",
        help="Path to annotations JSON file",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/invalid_duration_annotations.csv",
        help="Path to export invalid annotations CSV",
    )
    parser.add_argument(
        "--source-csv",
        type=str,
        default=None,
        help="Path to Beluga_annotations_processed.csv to recover durationSeconds",
    )
    args = parser.parse_args()

    stats, durations, durations_by_location, invalid_by_location, invalid_annotations = (
        compute_annotation_stats(args.annotations)
    )
    print_stats(stats, durations, durations_by_location, invalid_by_location)

    if invalid_annotations:
        export_invalid_annotations(
            invalid_annotations, args.source_csv, args.output_csv,
        )


if __name__ == "__main__":
    main()
