"""
Plot mel-spectrograms with annotation overlays for a given audio file.

Usage:
    # By sound_id:
    python plot_spectrograms.py --sound-id 434

    # By location (plots first sound with annotations):
    python plot_spectrograms.py --location 201D

    # Limit number of windows plotted:
    python plot_spectrograms.py --sound-id 434 --max-windows 12

    # Custom output path:
    python plot_spectrograms.py --location 201D -o my_plot.png
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


SAMPLE_RATE = 24000
NYQUIST = SAMPLE_RATE / 2
CATEGORY_MAP = {0: "Background", 1: "Humpback", 2: "Orca", 3: "Beluga"}
LABEL_COLORS = {0: "white", 1: "cyan", 2: "yellow", 3: "lime"}


def hz_to_mel(f):
    """Convert frequency in Hz to mel scale."""
    return 2595.0 * np.log10(1.0 + f / 700.0)


def load_data(annotations_path: str, windows_path: str):
    with open(annotations_path, "r") as f:
        data = json.load(f)
    with open(windows_path, "r") as f:
        windows = json.load(f)
    sounds = {s["id"]: s for s in data["sounds"]}
    return data, windows, sounds


def find_sound(sounds: dict, data: dict, sound_id=None, location=None):
    """Return a single sound dict matching the criteria."""
    if sound_id is not None:
        if sound_id not in sounds:
            raise ValueError(f"sound_id {sound_id} not found")
        return sounds[sound_id]

    # Find first sound at location that has valid annotations
    from collections import Counter

    anno_counts = Counter()
    for a in data["annotations"]:
        if a["t_max"] >= a["t_min"]:
            anno_counts[a["sound_id"]] += 1

    candidates = [
        s for s in data["sounds"]
        if s.get("location") == location and anno_counts.get(s["id"], 0) > 0
    ]
    if not candidates:
        raise ValueError(f"No sounds with annotations found at location '{location}'")
    # Pick the one with the most annotations
    candidates.sort(key=lambda s: anno_counts.get(s["id"], 0), reverse=True)
    return candidates[0]


def get_overlapping_windows(sound, windows, annotations):
    """Return list of (window, matched_annotations) for windows that overlap annotations."""
    sid = sound["id"]
    valid_annos = [
        a for a in annotations
        if a["sound_id"] == sid and a["t_max"] >= a["t_min"]
    ]
    sound_windows = [w for w in windows if w["sound_id"] == sid]

    overlapping = []
    for w in sound_windows:
        w_start_sec = w["start"] / SAMPLE_RATE
        w_end_sec = w["end"] / SAMPLE_RATE
        matched = []
        for a in valid_annos:
            overlap = max(0, min(a["t_max"], w_end_sec) - max(a["t_min"], w_start_sec))
            if overlap > 0:
                matched.append(a)
        if matched:
            overlapping.append((w, matched))

    # Sort by window_id so the grid follows temporal order
    overlapping.sort(key=lambda x: x[0]["window_id"])
    return overlapping


def plot_spectrograms(
    overlapping,
    sound,
    spec_dir: str,
    output_path: str,
    max_windows: int | None = None,
    ncols: int = 4,
):
    """Plot a grid of spectrograms with annotation bounding boxes."""
    sid = sound["id"]
    location = sound.get("location", "?")
    filename = sound["file_name_path"].split("/")[-1]

    if max_windows:
        overlapping = overlapping[:max_windows]

    n = len(overlapping)
    if n == 0:
        print("No windows overlap with valid annotations for this sound.")
        return

    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))

    # Normalize axes to 2D list
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    elif nrows == 1:
        axes_flat = list(axes)
    else:
        axes_flat = [ax for row in axes for ax in row]

    for idx, (w, annos) in enumerate(overlapping):
        ax = axes_flat[idx]
        wid = w["window_id"]
        start = w["start"]
        end = w["end"]
        label = w["label"]

        spec_name = f"sid{sid}_idx{wid}_start{start}_end{end}_lab{label}.npy"
        spec_path = os.path.join(spec_dir, spec_name)

        if not os.path.exists(spec_path):
            # Try standard naming
            spec_name = f"{filename}_{start}_{end}.npy"
            spec_path = os.path.join(spec_dir, spec_name)

        if os.path.exists(spec_path):
            spec = np.load(spec_path)
            w_start_sec = start / SAMPLE_RATE
            w_end_sec = end / SAMPLE_RATE

            mel_max = hz_to_mel(NYQUIST)
            ax.imshow(
                spec,
                aspect="auto",
                origin="lower",
                extent=[w_start_sec, w_end_sec, 0, mel_max],
                cmap="magma",
            )

            # Overlay annotation boxes (convert Hz to mel scale)
            for a in annos:
                t0 = max(a["t_min"], w_start_sec)
                t1 = min(a["t_max"], w_end_sec)
                f0_hz = a.get("f_min") or 0
                f1_hz = a.get("f_max") or NYQUIST
                f0 = hz_to_mel(f0_hz)
                f1 = hz_to_mel(f1_hz)
                cat_id = a.get("category_id", 0)
                color = LABEL_COLORS.get(cat_id, "lime")

                rect = mpatches.Rectangle(
                    (t0, f0),
                    t1 - t0,
                    f1 - f0,
                    linewidth=1.5,
                    edgecolor=color,
                    facecolor="none",
                    linestyle="--",
                )
                ax.add_patch(rect)

            label_name = CATEGORY_MAP.get(label, str(label))
            ax.set_title(
                f"wid={wid}  label={label_name}\n"
                f"[{w_start_sec:.1f}–{w_end_sec:.1f}s]",
                fontsize=8,
            )
            ax.set_ylim(0, mel_max)

            # Add Hz tick labels on the mel-scaled axis
            hz_ticks = [0, 1000, 2000, 4000, 6000, 8000, 10000, 12000]
            mel_ticks = [hz_to_mel(f) for f in hz_ticks]
            ax.set_yticks(mel_ticks)
            ax.set_yticklabels([f"{f//1000:.0f}k" if f >= 1000 else str(f) for f in hz_ticks])
            ax.set_ylabel("Hz", fontsize=7)
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.tick_params(labelsize=6)
        else:
            ax.set_title(f"wid={wid} NOT FOUND", fontsize=8)
            ax.axis("off")

    # Hide unused axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    # Build legend
    legend_patches = []
    cats_present = {a.get("category_id", 0) for _, annos in overlapping for a in annos}
    for cid in sorted(cats_present):
        legend_patches.append(
            mpatches.Patch(
                edgecolor=LABEL_COLORS.get(cid, "lime"),
                facecolor="none",
                linestyle="--",
                label=CATEGORY_MAP.get(cid, f"cat {cid}"),
            )
        )

    fig.suptitle(
        f"Sound {sid} ({location}) — {n} windows with annotations\n"
        f"File: {filename}",
        fontsize=11,
        fontweight="bold",
    )
    fig.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=9,
        framealpha=0.8,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=150)
    print(f"Saved {n} spectrograms to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot spectrograms with annotation overlays"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sound-id", type=int, help="Sound ID to plot")
    group.add_argument(
        "--location",
        type=str,
        help="Location code (e.g. 201D). Picks the sound with most annotations.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/annotations_combined.json",
        help="Path to annotations JSON",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default="data/windows_multiclass_from_mels.json",
        help="Path to windows JSON",
    )
    parser.add_argument(
        "--spec-dir",
        type=str,
        default="data/mel_spectrograms_multiclass",
        help="Directory containing .npy spectrograms",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output image path (default: data/spectrograms_<id>.png)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Maximum number of windows to plot",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=4,
        help="Number of columns in the grid (default: 4)",
    )
    args = parser.parse_args()

    data, windows, sounds = load_data(args.annotations, args.windows)

    sound = find_sound(
        sounds, data, sound_id=args.sound_id, location=args.location
    )
    sid = sound["id"]
    loc = sound.get("location", "?")

    print(
        f"Sound {sid} | location={loc} | "
        f"file={sound['file_name_path'].split('/')[-1]} | "
        f"duration={sound['duration']:.1f}s"
    )

    overlapping = get_overlapping_windows(sound, windows, data["annotations"])
    print(f"Found {len(overlapping)} windows overlapping annotations")

    output = args.output or f"data/spectrograms_visualizations/spectrograms_sid{sid}_{loc}.png"
    plot_spectrograms(
        overlapping, sound, args.spec_dir, output,
        max_windows=args.max_windows, ncols=args.ncols,
    )


if __name__ == "__main__":
    main()
