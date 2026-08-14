#!/usr/bin/env python3
"""Plot mean voxel R² as a function of network layer.

Expected directory layout:

    ROOT/<backbone>_set<subset>/<layer>/r2.pkl

Each ``r2.pkl`` must be a mapping from ROI name to a mapping containing an
``r2_voxels`` array. Pickle files must come from a trusted source.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ROOT = Path("/user_data/junruz/prf_models/fixed/split_1_zscore/S1")
MODEL_RE = re.compile(r"^(?P<backbone>.+)_set(?P<subset>[^_]+)$")
NUMBER_RE = re.compile(r"(\d+)")
DEFAULT_ROI_ORDER = ["V1", "V2", "V3", "hV4", "FFA", "PPA"]


def natural_key(text: str) -> tuple:
    """Sort strings containing numbers in human order."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in NUMBER_RE.split(text)
    )


def layer_key(layer: str) -> tuple:
    """Order ResNet and ConvNeXt layers from shallowest to deepest."""
    lower = layer.lower()
    if lower in {"relu", "stem", "conv1"}:
        return (0, 0, natural_key(layer))
    match = re.fullmatch(r"(?:layer|stage)([1-4])", lower)
    if match:
        return (1, int(match.group(1)), natural_key(layer))
    return (2, 0, natural_key(layer))


def discover_results(root: Path) -> tuple[dict, list[dict[str, object]]]:
    """Load all available R² files and return nested results plus CSV rows."""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    rows: list[dict[str, object]] = []

    for model_dir in sorted(root.iterdir(), key=lambda path: natural_key(path.name)):
        if not model_dir.is_dir():
            continue
        match = MODEL_RE.match(model_dir.name)
        if not match:
            warnings.warn(f"Skipping unrecognized model directory: {model_dir}")
            continue

        backbone = match.group("backbone")
        subset = match.group("subset")
        found_r2 = False
        for layer_dir in sorted(model_dir.iterdir(), key=lambda path: layer_key(path.name)):
            r2_path = layer_dir / "r2.pkl"
            if not layer_dir.is_dir() or not r2_path.is_file():
                continue
            found_r2 = True
            with r2_path.open("rb") as file:
                roi_results = pickle.load(file)

            if not isinstance(roi_results, dict):
                warnings.warn(f"Skipping {r2_path}: expected a dictionary")
                continue
            for roi, values in roi_results.items():
                if not isinstance(values, dict) or "r2_voxels" not in values:
                    warnings.warn(f"Skipping {r2_path}:{roi}: no r2_voxels entry")
                    continue
                voxel_r2 = np.asarray(values["r2_voxels"], dtype=float).ravel()
                finite = voxel_r2[np.isfinite(voxel_r2)]
                if finite.size == 0:
                    warnings.warn(f"Skipping {r2_path}:{roi}: no finite values")
                    continue
                mean_r2 = float(np.mean(finite))
                results[backbone][subset][layer_dir.name][str(roi)] = mean_r2
                rows.append(
                    {
                        "backbone": backbone,
                        "subset": subset,
                        "layer": layer_dir.name,
                        "roi": str(roi),
                        "mean_r2": mean_r2,
                        "n_voxels": int(finite.size),
                        "source": str(r2_path),
                    }
                )
        if not found_r2:
            warnings.warn(f"No r2.pkl files found under {model_dir}")

    return results, rows


def requested_rois(rows: list[dict[str, object]], rois: list[str] | None) -> list[str]:
    available = sorted({str(row["roi"]) for row in rows}, key=natural_key)
    if rois is None:
        ordered = [roi for roi in DEFAULT_ROI_ORDER if roi in available]
        return ordered + [roi for roi in available if roi not in ordered]
    missing = sorted(set(rois) - set(available), key=natural_key)
    if missing:
        warnings.warn("Requested ROI(s) not found: " + ", ".join(missing))
    return [roi for roi in rois if roi in available]


def plot_backbone_subset(
    backbone: str,
    subset: str,
    layer_results: dict,
    rois: list[str],
    output_dir: Path,
    dpi: int,
    ylim: tuple[float, float],
) -> None:
    """Make one figure for a backbone/subset, with one curve per ROI."""
    layers = sorted(layer_results, key=layer_key)
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(10, 7))
    for roi in rois:
        y = np.asarray(
            [layer_results[layer].get(roi, np.nan) for layer in layers],
            dtype=float,
        )
        if np.isfinite(y).any():
            line = ax.plot(x, y, marker="o", linewidth=2, label=roi)[0]
            best = int(np.nanargmax(y))
            ax.plot(
                x[best],
                y[best],
                marker="*",
                markersize=18,
                color=line.get_color(),
                markeredgecolor="black",
                linestyle="none",
            )
    ax.set(
        xticks=x,
        xticklabels=layers,
        xlabel="Layer",
        ylabel="Mean voxel $R^2$",
        title=f"{backbone} — data subset {subset}",
        ylim=ylim,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="ROI", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / f"{backbone}_set{subset}_mean_r2_by_layer.png", dpi=dpi)
    plt.close(fig)


def plot_subset_comparison(
    backbone: str,
    subset_results: dict,
    rois: list[str],
    output_dir: Path,
    dpi: int,
    ylim: tuple[float, float],
) -> None:
    """Make an ROI panel figure comparing data subsets for one backbone."""
    ncols = min(3, len(rois))
    nrows = math.ceil(len(rois) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False
    )
    for index, (ax, roi) in enumerate(zip(axes.flat, rois)):
        all_layers = sorted(
            {layer for result in subset_results.values() for layer in result},
            key=layer_key,
        )
        x = np.arange(len(all_layers))
        for subset in sorted(subset_results, key=natural_key):
            y = np.asarray(
                [
                    subset_results[subset].get(layer, {}).get(roi, np.nan)
                    for layer in all_layers
                ],
                dtype=float,
            )
            if np.isfinite(y).any():
                line = ax.plot(
                    x, y, marker="o", linewidth=2, label=f"set{subset}"
                )[0]
                best = int(np.nanargmax(y))
                ax.plot(
                    x[best],
                    y[best],
                    marker="*",
                    markersize=18,
                    color=line.get_color(),
                    markeredgecolor="black",
                    linestyle="none",
                )
        ax.set_title(roi)
        ax.set_xticks(x, all_layers, rotation=30, ha="right")
        if index % ncols == 0:
            ax.set_ylabel("Mean voxel $R^2$")
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(rois) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
        )
    fig.suptitle(f"{backbone}: data-subset comparison", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        output_dir / f"{backbone}_subset_comparison_by_roi.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    columns = ["backbone", "subset", "layer", "roi", "mean_r2", "n_voxels", "source"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: ROOT/layerwise_mean_r2",
    )
    parser.add_argument(
        "--rois",
        nargs="+",
        default=None,
        help="ROI names to plot (default: every ROI found)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=(0.0, 0.25),
        help="Fixed y-axis limits for every panel (default: 0.0 0.25)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Input directory does not exist: {root}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / "layerwise_mean_r2"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.ylim[0] >= args.ylim[1]:
        raise SystemExit("--ylim MIN must be smaller than MAX")
    plt.rcParams.update({"font.size": 18})

    results, rows = discover_results(root)
    if not rows:
        raise SystemExit(f"No usable r2.pkl results found below {root}")
    rois = requested_rois(rows, args.rois)
    if not rois:
        raise SystemExit("None of the requested ROIs were found")

    for backbone, subset_results in sorted(results.items(), key=lambda item: natural_key(item[0])):
        for subset, layer_results in sorted(
            subset_results.items(), key=lambda item: natural_key(item[0])
        ):
            plot_backbone_subset(
                backbone,
                subset,
                layer_results,
                rois,
                output_dir,
                args.dpi,
                tuple(args.ylim),
            )
        plot_subset_comparison(
            backbone, subset_results, rois, output_dir, args.dpi, tuple(args.ylim)
        )

    write_csv(rows, output_dir / "layerwise_mean_r2.csv")
    print(f"Wrote {len(rows)} mean R² values and figures to {output_dir}")


if __name__ == "__main__":
    main()
