"""Fit a pRF encoding model using concatenated features from all CNN layers.

For a candidate pRF ID, this script loads that same ID from every selected
layer and concatenates the arrays along their feature/channel dimension.  The
rest of the fitting procedure (feature normalization, nested-set lambda
selection, and voxel-wise pRF selection) matches
``fit_prf_model_split_layerwise_zscore.py``.
"""

import argparse
import json
import os
import pickle
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

import model_fitting_utils


DEFAULT_LAYER_ORDER = {
    "CLIP_RN50": ["relu", "layer1", "layer2", "layer3", "layer4"],
    "OPEN_CLIP_RN50": ["relu", "layer1", "layer2", "layer3", "layer4"],
    "DINO_RN50": ["relu", "layer1", "layer2", "layer3", "layer4"],
    "SIMCLR_RN50": ["relu", "layer1", "layer2", "layer3", "layer4"],
    "ADV_RN50": ["relu", "layer1", "layer2", "layer3", "layer4"],
    "OPEN_CLIP_CONVNEXT_BASE": [
        "stem",
        "stage1",
        "stage2",
        "stage3",
        "stage4",
    ],
}

PRF_FILENAME_RE = re.compile(r"^features_prf_(\d+)\.npy$")


def load_nsd_data(data_folder, labels_folder, subject):
    """Load NSD responses and discard images with missing responses."""
    info_filename = os.path.join(labels_folder, f"S{subject}_image_info.csv")
    print(info_filename)
    info = pd.read_csv(info_filename)
    n_reps = np.asarray(info["n_reps"])

    data_filename = os.path.join(
        data_folder, f"S{subject}_betas_avg_bigmask.hdf5"
    )
    print(data_filename)
    start = time.time()
    with h5py.File(data_filename, "r") as data_set:
        values = np.asarray(data_set["/betas"])
    print(f"Took {time.time() - start:.5f} seconds to load file")

    good_values = ~np.isnan(values[:, 0])
    assert np.all(good_values[n_reps > 0])
    assert np.all(~good_values[n_reps == 0])
    return values[good_values, :], good_values


def _prf_ids(layer_folder):
    """Return the pRF IDs represented by correctly named feature files."""
    ids = set()
    for path in Path(layer_folder).iterdir():
        match = PRF_FILENAME_RE.match(path.name)
        if path.is_file() and match:
            ids.add(int(match.group(1)))
    return ids


def resolve_layers(model_folder, model_name, requested_layers=None):
    """Resolve and validate the ordered set of layer feature directories."""
    model_folder = Path(model_folder)
    if requested_layers:
        layer_names = list(requested_layers)
    else:
        available = {
            path.name
            for path in model_folder.iterdir()
            if path.is_dir() and _prf_ids(path)
        }
        preferred = DEFAULT_LAYER_ORDER.get(model_name, [])
        layer_names = [name for name in preferred if name in available]
        layer_names.extend(sorted(available.difference(layer_names)))

    if not layer_names:
        raise ValueError(f"No layer feature directories found in {model_folder}")
    if len(layer_names) != len(set(layer_names)):
        raise ValueError(f"Layer names must be unique, got {layer_names}")

    layer_folders = [model_folder / name for name in layer_names]
    missing = [str(path) for path in layer_folders if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Missing layer feature directories: " + ", ".join(missing)
        )

    id_sets = [_prf_ids(folder) for folder in layer_folders]
    if any(not ids for ids in id_sets):
        empty_layers = [
            name for name, ids in zip(layer_names, id_sets) if not ids
        ]
        raise ValueError(
            f"No features_prf_<id>.npy files found for layers: {empty_layers}"
        )

    reference_ids = id_sets[0]
    mismatches = []
    for name, ids in zip(layer_names[1:], id_sets[1:]):
        if ids != reference_ids:
            missing_ids = sorted(reference_ids - ids)
            extra_ids = sorted(ids - reference_ids)
            mismatches.append(
                f"{name}: missing={missing_ids[:10]}, extra={extra_ids[:10]}"
            )
    if mismatches:
        raise ValueError(
            "All layers must contain exactly the same pRF IDs. "
            + "; ".join(mismatches)
        )

    return layer_names, layer_folders, sorted(reference_ids)


def load_concatenated_prf(layer_names, layer_folders, prf_id):
    """Load one pRF from every layer and concatenate its channels."""
    arrays = []
    feature_slices = {}
    start = 0
    num_images = None

    for layer_name, layer_folder in zip(layer_names, layer_folders):
        path = layer_folder / f"features_prf_{prf_id}.npy"
        features = np.load(path)
        if features.ndim != 2:
            raise ValueError(
                f"Expected a 2-D [images, channels] array in {path}, "
                f"got shape {features.shape}"
            )
        if num_images is None:
            num_images = features.shape[0]
        elif features.shape[0] != num_images:
            raise ValueError(
                f"Image count mismatch in {path}: expected {num_images}, "
                f"got {features.shape[0]}"
            )

        stop = start + features.shape[1]
        feature_slices[layer_name] = [start, stop]
        start = stop
        arrays.append(features)

    return np.concatenate(arrays, axis=1).astype(np.float64, copy=False), feature_slices


def model_fitting(
    voxel_data,
    train_ids,
    val_ids,
    nest_ids,
    layer_names,
    layer_folders,
    prf_ids,
    device,
):
    """Fit every shared pRF and retain the best one independently per voxel."""
    train_voxel = torch.from_numpy(voxel_data[train_ids, :]).to(
        device=device, dtype=torch.float64
    )
    nest_voxel = torch.from_numpy(voxel_data[nest_ids, :]).to(
        device=device, dtype=torch.float64
    )
    num_voxels = train_voxel.shape[1]

    """
    The function loads the first pRF from all layers to determine:
        1. Total number of concatenated channels.
        2. The location of each layer inside the weight vector.
    """
    first_features, feature_slices = load_concatenated_prf(
        layer_names, layer_folders, prf_ids[0]
    )
    num_features = first_features.shape[1]
    del first_features

    #### Initialize parameters ####
    best_weights = {}
    best_lambda = np.zeros(num_voxels, dtype=np.float64)
    best_nest_loss = np.full(num_voxels, np.inf, dtype=np.float64)
    best_prf_idx = np.full(num_voxels, -1, dtype=np.int64)
    best_features_s = np.zeros((num_voxels, num_features), dtype=np.float64)
    best_features_m = np.zeros((num_voxels, num_features), dtype=np.float64)

    n_lambdas = 20
    small_value = 0.0001
    lambdas = (
        np.logspace(
            np.log(small_value),
            np.log(10**10 + small_value),
            n_lambdas,
            dtype=np.float64,
            base=np.e,
        )
        - small_value
    )

    print(
        f"Fitting {len(prf_ids)} pRFs with {num_features} concatenated "
        f"features from layers: {', '.join(layer_names)}",
        flush=True,
    )
    start_time = time.time()
    for count, prf_id in enumerate(prf_ids, start=1):
        features, current_slices = load_concatenated_prf(
            layer_names, layer_folders, prf_id
        )
        if features.shape[1] != num_features or current_slices != feature_slices:
            raise ValueError(
                f"Feature dimensions changed at pRF {prf_id}: "
                f"expected {num_features}, got {features.shape[1]}"
            )

        (
            train_features,
            _,
            nest_features,
            features_s,
            features_m,
        ) = model_fitting_utils.split_normalize_feats(
            features, train_ids, val_ids, nest_ids
        )

        #### Add intercept term to features ####
        train_features = np.concatenate(
            [
                train_features,
                np.ones((len(train_features), 1), dtype=train_features.dtype),
            ],
            axis=1,
        )
        nest_features = np.concatenate(
            [
                nest_features,
                np.ones((len(nest_features), 1), dtype=nest_features.dtype),
            ],
            axis=1,
        )

        train_features = torch.from_numpy(train_features).to(device)
        nest_features = torch.from_numpy(nest_features).to(device)
        weights, lambda_indices, nest_loss = model_fitting_utils.solve_ridge(
            train_features,
            train_voxel,
            nest_features,
            nest_voxel,
            lambdas,
            eps=1e-4,
            return_loss=True,
        )

        improved = nest_loss < best_nest_loss
        for voxel_idx in np.flatnonzero(improved):
            best_weights[str(voxel_idx)] = (
                weights[:, voxel_idx].detach().cpu().numpy().astype(np.float32)
            )
        best_lambda[improved] = lambdas[lambda_indices[improved]]
        best_nest_loss[improved] = nest_loss[improved]
        best_prf_idx[improved] = prf_id
        best_features_s[improved] = features_s
        best_features_m[improved] = features_m

        if count == 1 or count % 25 == 0 or count == len(prf_ids):
            print(
                f"Processed {count}/{len(prf_ids)} pRFs "
                f"(pRF ID {prf_id}; updated {improved.sum()} voxels)",
                flush=True,
            )

    print(f"Time taken: {time.time() - start_time:.1f} seconds", flush=True)
    return (
        best_weights,
        best_lambda,
        best_nest_loss,
        best_prf_idx,
        best_features_s,
        best_features_m,
        feature_slices,
    )


def parse_args():
    nsd_path = "/lab_data/hendersonlab/datasets/nsd_preproc"
    parser = argparse.ArgumentParser(
        description="Fit a pRF model to features concatenated across layers."
    )
    parser.add_argument("--subject_id", nargs="+", default=[1], type=int)
    parser.add_argument("--model_name", type=str, default="DINO_RN50")
    parser.add_argument("--split_id", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--layer_names",
        nargs="+",
        default=None,
        help="Ordered layers to concatenate (default: every available layer).",
    )
    parser.add_argument(
        "--features_root", type=str, default="/user_data/junruz/prf_features"
    )
    parser.add_argument(
        "--split_root",
        type=str,
        default="/user_data/junruz/prf_models/concat/split_1_zscore",
    )
    parser.add_argument(
        "--nsd_path",
        type=str,
        default=nsd_path,
        help="Root containing NSD data, labels, and stimuli directories.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    subject = args.subject_id[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    split_dir = os.path.join(args.split_root, f"S{subject}")
    split_path = os.path.join(split_dir, f"data_splits_S{subject}.pkl")
    with open(split_path, "rb") as file:
        data_splits = pickle.load(file)
    ids = data_splits[args.split_id]
    train_ids, val_ids, nest_ids = ids["train"], ids["val"], ids["nest"]

    model_folder = os.path.join(
        args.features_root, f"S{subject}", args.model_name
    )
    layer_names, layer_folders, prf_ids = resolve_layers(
        model_folder, args.model_name, args.layer_names
    )
    print("Layer feature folders:")
    for folder in layer_folders:
        print(f"  {folder}")

    data_folder = os.path.join(args.nsd_path, "data")
    labels_folder = os.path.join(args.nsd_path, "labels")
    voxel_data, _ = load_nsd_data(data_folder, labels_folder, subject)

    (
        best_weights,
        best_lambda,
        best_loss,
        best_prf_idx,
        best_features_s,
        best_features_m,
        feature_slices,
    ) = model_fitting(
        voxel_data,
        train_ids,
        val_ids,
        nest_ids,
        layer_names,
        layer_folders,
        prf_ids,
        device,
    )

    print(
        f"Max loss: {np.max(best_loss)}, Min loss: {np.min(best_loss)}, "
        f"Mean loss: {np.mean(best_loss)}"
    )
    output_folder = os.path.join(
        args.split_root,
        f"S{subject}",
        f"{args.model_name}_set{args.split_id}",
    )
    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(output_folder, "best_weights.pkl"), "wb") as file:
        pickle.dump(best_weights, file)
    np.save(os.path.join(output_folder, "best_lambda.npy"), best_lambda)
    np.save(os.path.join(output_folder, "best_loss.npy"), best_loss)
    np.save(os.path.join(output_folder, "best_prf_idx.npy"), best_prf_idx)
    np.save(os.path.join(output_folder, "best_features_s.npy"), best_features_s)
    np.save(os.path.join(output_folder, "best_features_m.npy"), best_features_m)

    metadata = {
        "model_name": args.model_name,
        "layers": layer_names,
        "feature_slices": feature_slices,
        "num_features": int(best_features_m.shape[1]),
        "intercept_index": int(best_features_m.shape[1]),
        "prf_ids": prf_ids,
    }
    with open(os.path.join(output_folder, "concat_metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)
    print(f"Saved concatenated model fits to: {output_folder}")


if __name__ == "__main__":
    main()
