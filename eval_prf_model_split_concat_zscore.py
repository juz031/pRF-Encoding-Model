"""Evaluate a pRF model fitted to features concatenated across CNN layers."""

import argparse
import json
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fit_prf_model_split_concat_zscore import (
    load_concatenated_prf,
    load_nsd_data,
)


def load_nsd_rois(subject, roi_path):
    """Load ROI masks and noise ceilings for voxels in the NSD big mask."""
    print(f"Loading ROIs from: {roi_path}")
    roi_info = np.load(roi_path, allow_pickle=True).item()
    noise_ceiling = roi_info["noise_ceiling_avgreps"] / 100.0
    big_mask = roi_info["voxel_mask"]

    roi_keys = [
        "roi_labels_retino",
        "roi_labels_kastner",
        "roi_labels_face",
        "roi_labels_place",
        "roi_labels_body",
    ]
    roi_name_keys = [
        "ret_prf_roi_names",
        "kastner_atlas_roi_names",
        "floc_face_roi_names",
        "floc_place_roi_names",
        "floc_body_roi_names",
    ]

    roi_masks = {}
    for label_key, name_key in zip(roi_keys, roi_name_keys):
        labels = roi_info[label_key][big_mask]
        names = roi_info[name_key]
        for name, label in names.items():
            roi_masks[name] = labels == label

    for name in ("V1", "V2", "V3"):
        roi_masks[name] = roi_masks[f"{name}v"] | roi_masks[f"{name}d"]
    roi_masks["FFA"] = roi_masks["FFA-1"] | roi_masks["FFA-2"]

    print(f"Loaded {len(roi_masks)} ROIs: {list(roi_masks)}")
    return roi_masks, noise_ceiling


def load_model(model_path):
    """Load fitted parameters and concatenation metadata."""
    required_paths = {
        "weights": model_path / "best_weights.pkl",
        "prf": model_path / "best_prf_idx.npy",
        "mean": model_path / "best_features_m.npy",
        "std": model_path / "best_features_s.npy",
        "metadata": model_path / "concat_metadata.json",
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing fitted-model files: " + ", ".join(missing))

    with open(required_paths["weights"], "rb") as file:
        best_weights = pickle.load(file)
    best_prf_idx = np.load(required_paths["prf"])
    best_features_m = np.load(required_paths["mean"])
    best_features_s = np.load(required_paths["std"])
    with open(required_paths["metadata"], "r") as file:
        metadata = json.load(file)

    return (
        best_weights,
        best_prf_idx,
        best_features_m,
        best_features_s,
        metadata,
    )


def validate_model(
    best_weights,
    best_prf_idx,
    best_features_m,
    best_features_s,
    metadata,
    num_voxels,
    model_name,
):
    """Check that fitted arrays and metadata agree before prediction."""
    if metadata.get("model_name") != model_name:
        raise ValueError(
            f"Metadata model is {metadata.get('model_name')!r}, "
            f"but --model_name is {model_name!r}"
        )

    num_features = int(metadata["num_features"])
    expected_shape = (num_voxels, num_features)
    if best_prf_idx.shape != (num_voxels,):
        raise ValueError(
            f"best_prf_idx has shape {best_prf_idx.shape}; "
            f"expected {(num_voxels,)}"
        )
    if best_features_m.shape != expected_shape:
        raise ValueError(
            f"best_features_m has shape {best_features_m.shape}; "
            f"expected {expected_shape}"
        )
    if best_features_s.shape != expected_shape:
        raise ValueError(
            f"best_features_s has shape {best_features_s.shape}; "
            f"expected {expected_shape}"
        )
    if len(best_weights) != num_voxels:
        raise ValueError(
            f"best_weights has {len(best_weights)} voxels; expected {num_voxels}"
        )
    missing_weight_keys = [
        str(voxel_idx)
        for voxel_idx in range(num_voxels)
        if str(voxel_idx) not in best_weights
    ]
    if missing_weight_keys:
        raise ValueError(
            "best_weights is missing voxel keys: "
            + ", ".join(missing_weight_keys[:10])
        )
    invalid_prfs = set(np.unique(best_prf_idx)) - set(metadata["prf_ids"])
    if invalid_prfs:
        raise ValueError(f"Selected pRF IDs absent from metadata: {invalid_prfs}")

    first_weight = np.asarray(best_weights["0"])
    if first_weight.shape != (num_features + 1,):
        raise ValueError(
            f"Weight vectors must contain {num_features} features plus one "
            f"intercept; voxel 0 has shape {first_weight.shape}"
        )
    if np.any(best_features_s <= 0):
        raise ValueError("Feature standard deviations must all be positive")


def eval_prf_model_concat(
    voxel_data,
    val_ids,
    model_path,
    features_model_folder,
    model_name,
    device,
    voxel_batch_size=1024,
):
    """Predict held-out responses using each voxel's selected shared pRF."""
    (
        best_weights,
        best_prf_idx,
        best_features_m,
        best_features_s,
        metadata,
    ) = load_model(model_path)

    num_val_images = voxel_data[val_ids, :].shape[0]
    num_voxels = voxel_data.shape[1]
    validate_model(
        best_weights,
        best_prf_idx,
        best_features_m,
        best_features_s,
        metadata,
        num_voxels,
        model_name,
    )

    layer_names = metadata["layers"]
    layer_folders = [features_model_folder / name for name in layer_names]
    missing_layers = [str(path) for path in layer_folders if not path.is_dir()]
    if missing_layers:
        raise FileNotFoundError(
            "Missing layer feature directories: " + ", ".join(missing_layers)
        )

    predicted_neural = np.empty(
        (num_val_images, num_voxels), dtype=np.float32
    )
    selected_prfs = np.unique(best_prf_idx)
    print(
        f"Predicting {num_val_images} validation images for {num_voxels} "
        f"voxels using {len(selected_prfs)} selected pRFs",
        flush=True,
    )

    with torch.no_grad():
        for prf_count, prf_id in enumerate(selected_prfs, start=1):
            features, feature_slices = load_concatenated_prf(
                layer_names, layer_folders, int(prf_id)
            )
            if feature_slices != metadata["feature_slices"]:
                raise ValueError(
                    f"Layer feature slices for pRF {prf_id} do not match "
                    "concat_metadata.json"
                )
            if features.shape[1] != metadata["num_features"]:
                raise ValueError(
                    f"pRF {prf_id} has {features.shape[1]} features; "
                    f"expected {metadata['num_features']}"
                )

            val_features = np.asarray(features[val_ids, :], dtype=np.float32)
            val_features = torch.from_numpy(val_features).to(device)
            voxel_indices = np.flatnonzero(best_prf_idx == prf_id)

            for start in range(0, len(voxel_indices), voxel_batch_size):
                batch_indices = voxel_indices[start : start + voxel_batch_size]
                models = np.column_stack(
                    [best_weights[str(index)] for index in batch_indices]
                ).astype(np.float32, copy=False)
                weights = torch.from_numpy(models[:-1, :]).to(device)
                intercept = torch.from_numpy(models[-1, :]).to(device)
                means = torch.from_numpy(
                    best_features_m[batch_indices, :].astype(
                        np.float32, copy=False
                    )
                ).to(device)
                stds = torch.from_numpy(
                    best_features_s[batch_indices, :].astype(
                        np.float32, copy=False
                    )
                ).to(device)

                # This is algebraically equivalent to normalizing a separate
                # feature matrix for every voxel:
                # ((X - mean) / std) @ weight + intercept.
                scaled_weights = weights / stds.T
                adjusted_intercept = intercept - torch.sum(
                    (means / stds) * weights.T, dim=1
                )
                prediction = val_features @ scaled_weights + adjusted_intercept
                predicted_neural[:, batch_indices] = prediction.cpu().numpy()

            print(
                f"Processed selected pRF {prf_count}/{len(selected_prfs)} "
                f"(ID {prf_id}; {len(voxel_indices)} voxels)",
                flush=True,
            )

    print(f"Predicted neural response shape: {predicted_neural.shape}")
    return predicted_neural


def get_r2(true_neural, predicted_neural):
    """Calculate R-squared independently for every voxel (columns)."""
    residual_sum = np.sum((predicted_neural - true_neural) ** 2, axis=0)
    total_sum = np.sum(
        (true_neural - np.mean(true_neural, axis=0, keepdims=True)) ** 2,
        axis=0,
    )
    return 1.0 - np.divide(
        residual_sum,
        total_sum,
        out=np.full_like(residual_sum, np.nan, dtype=np.float64),
        where=total_sum != 0,
    )


def calculate_r2(
    predicted_neural,
    true_neural,
    roi_masks,
    noise_ceiling,
    roi_names,
    noise_ceiling_threshold,
    save_root,
):
    """Calculate ROI metrics and save the same plots as the layerwise evaluator."""
    os.makedirs(save_root, exist_ok=True)
    nc_mask = noise_ceiling > noise_ceiling_threshold
    results = {}

    for area in roi_names:
        if area not in roi_masks:
            raise KeyError(
                f"Unknown ROI {area!r}. Available ROIs: {list(roi_masks)}"
            )
        mask = roi_masks[area] & nc_mask
        num_voxels = int(np.sum(mask))
        print(
            f"Selected {num_voxels} voxels in {area}, "
            f"NC > {noise_ceiling_threshold}"
        )
        if num_voxels == 0:
            print(f"Skipping {area}: no voxels passed the masks")
            results[area] = {
                "r2_voxels": np.empty((0, 1)),
                "a": np.nan,
                "b": np.nan,
                "noise_ceiling": noise_ceiling[mask],
            }
            continue

        r2_voxels = get_r2(
            true_neural[:, mask], predicted_neural[:, mask]
        )
        mean_r2 = float(np.nanmean(r2_voxels))
        median_r2 = float(np.nanmedian(r2_voxels))
        print(f"{area} mean R2: {mean_r2}")
        print(f"{area} median R2: {median_r2}")

        plt.figure()
        plt.hist(r2_voxels[np.isfinite(r2_voxels)])
        plt.axvline(median_r2, color="k")
        plt.title(
            f"R2 histogram for {area}, concatenated, "
            f"NC > {noise_ceiling_threshold}"
        )
        plt.xlabel("R2")
        plt.ylabel("Voxel count")
        plt.gca().text(
            0.95,
            0.95,
            f"Mean: {mean_r2:.3f}\nMedian: {median_r2:.3f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )
        plt.savefig(
            os.path.join(
                save_root,
                f"hist_{area}_concat_nc_{noise_ceiling_threshold}.png",
            ),
            dpi=300,
        )
        plt.close()

        selected_nc = noise_ceiling[mask]
        finite = np.isfinite(r2_voxels) & np.isfinite(selected_nc)
        if np.sum(finite) >= 2 and np.ptp(selected_nc[finite]) > 0:
            a, b = np.polyfit(selected_nc[finite], r2_voxels[finite], 1)
        else:
            a, b = np.nan, np.nan

        plt.figure()
        plt.rcParams.update({"font.size": 18})
        plt.scatter(selected_nc, r2_voxels, s=5)
        if np.isfinite(a) and np.isfinite(b):
            line_x = np.sort(selected_nc[finite])
            plt.plot(line_x, a * line_x + b, color="r", linestyle="--")
        plt.xlabel("Noise Ceiling")
        plt.ylabel("R2 per voxel")
        plt.ylim(-0.2, 0.8)
        plt.title(f"R2 vs NC for {area}, concatenated")
        plt.gca().text(
            0.05,
            0.95,
            f"Mean: {mean_r2:.3f}\nMedian: {median_r2:.3f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )
        plt.axline((0, 0), slope=1, color="k", linestyle="--")
        plt.savefig(
            os.path.join(
                save_root,
                f"scatter_{area}_concat_nc_{noise_ceiling_threshold}.png",
            ),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        results[area] = {
            "r2_voxels": r2_voxels[:, np.newaxis],
            "a": a,
            "b": b,
            "noise_ceiling": selected_nc,
        }

    return results


def validate_r2_results(
    results,
    roi_masks,
    noise_ceiling,
    roi_names,
    noise_ceiling_threshold,
):
    """Validate ROI result shapes before serializing them."""
    nc_mask = noise_ceiling > noise_ceiling_threshold
    expected_rois = set(roi_names)
    actual_rois = set(results)
    if actual_rois != expected_rois:
        raise ValueError(
            f"R2 ROI keys do not match: expected {sorted(expected_rois)}, "
            f"got {sorted(actual_rois)}"
        )

    for area in roi_names:
        expected_num_voxels = int(np.sum(roi_masks[area] & nc_mask))
        area_results = results[area]
        r2_shape = np.shape(area_results["r2_voxels"])
        expected_r2_shape = (expected_num_voxels, 1)
        if r2_shape != expected_r2_shape:
            raise ValueError(
                f"Incorrect R2 shape for {area}: got {r2_shape}, "
                f"expected {expected_r2_shape}"
            )

        nc_shape = np.shape(area_results["noise_ceiling"])
        expected_nc_shape = (expected_num_voxels,)
        if nc_shape != expected_nc_shape:
            raise ValueError(
                f"Incorrect noise-ceiling shape for {area}: got {nc_shape}, "
                f"expected {expected_nc_shape}"
            )

        for coefficient in ("a", "b"):
            if np.ndim(area_results[coefficient]) != 0:
                raise ValueError(
                    f"Regression coefficient {coefficient!r} for {area} "
                    f"must be scalar, got shape "
                    f"{np.shape(area_results[coefficient])}"
                )

        print(
            f"Validated {area}: r2_voxels shape {r2_shape}, "
            f"noise_ceiling shape {nc_shape}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a concatenated-layer pRF model."
    )
    parser.add_argument("--subject_id", nargs="+", default=[1], type=int)
    parser.add_argument("--model_name", type=str, default="DINO_RN50")
    parser.add_argument("--split_id", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--features_root", type=str, default="/user_data/junruz/prf_features"
    )
    parser.add_argument(
        "--split_root",
        type=str,
        default="/user_data/junruz/prf_models/concat/split_1_zscore",
        help="Root containing both data splits and fitted concatenated models.",
    )
    parser.add_argument(
        "--nsd_path",
        type=str,
        default="/lab_data/hendersonlab/datasets/nsd_preproc",
    )
    parser.add_argument(
        "--roi_path",
        type=str,
        default=None,
        help="ROI file; default: <nsd_path>/rois/S<subject>_voxel_roi_info.npy",
    )
    parser.add_argument(
        "--roi",
        nargs="+",
        default=["V1", "V2", "V3", "hV4", "FFA", "PPA"],
    )
    parser.add_argument("--noise_ceiling_threshold", type=float, default=0.0)
    parser.add_argument(
        "--voxel_batch_size",
        type=int,
        default=1024,
        help="Number of same-pRF voxels predicted together on the device.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.voxel_batch_size <= 0:
        raise ValueError("--voxel_batch_size must be positive")

    subject = args.subject_id[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_folder = os.path.join(args.nsd_path, "data")
    labels_folder = os.path.join(args.nsd_path, "labels")
    if args.roi_path is None:
        roi_path = os.path.join(
            args.nsd_path, "rois", f"S{subject}_voxel_roi_info.npy"
        )
    else:
        roi_path = args.roi_path.format(subject)

    subject_root = Path(args.split_root) / f"S{subject}"
    model_path = subject_root / f"{args.model_name}_set{args.split_id}"
    split_path = subject_root / f"data_splits_S{subject}.pkl"
    features_model_folder = (
        Path(args.features_root) / f"S{subject}" / args.model_name
    )
    print(f"Loading fitted model from: {model_path}")
    print(f"Loading features from: {features_model_folder}")

    with open(split_path, "rb") as file:
        data_splits = pickle.load(file)
    ids = data_splits[args.split_id]
    val_ids = ids["val"]

    roi_masks, noise_ceiling = load_nsd_rois(subject, roi_path)
    voxel_data, _ = load_nsd_data(data_folder, labels_folder, subject)
    if len(noise_ceiling) != voxel_data.shape[1]:
        raise ValueError(
            f"Noise ceiling has {len(noise_ceiling)} voxels, but neural data "
            f"has {voxel_data.shape[1]}"
        )

    num_test_images = voxel_data[val_ids, :].shape[0]
    print(
        f"Testing on the held-out 'val' partition from split "
        f"{args.split_id}: {num_test_images} images"
    )

    predicted_neural = eval_prf_model_concat(
        voxel_data,
        val_ids,
        model_path,
        features_model_folder,
        args.model_name,
        device,
        args.voxel_batch_size,
    )
    true_neural = voxel_data[val_ids, :]
    plots_folder = model_path / "plots"
    r2 = calculate_r2(
        predicted_neural,
        true_neural,
        roi_masks,
        noise_ceiling,
        args.roi,
        args.noise_ceiling_threshold,
        plots_folder,
    )
    validate_r2_results(
        r2,
        roi_masks,
        noise_ceiling,
        args.roi,
        args.noise_ceiling_threshold,
    )
    with open(model_path / "r2.pkl", "wb") as file:
        pickle.dump(r2, file)
    print(f"Saved R2 results to: {model_path / 'r2.pkl'}")
    print(f"Saved plots to: {plots_folder}")


if __name__ == "__main__":
    main()
