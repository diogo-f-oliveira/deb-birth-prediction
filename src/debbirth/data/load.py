"""CSV preparation with lazy historical family-loader wrappers."""
from hashlib import sha256

import pandas as pd

from .prepare import SOURCE_FILE, SOURCE_ROW, prepare_features, prepare_split
from .schema import DatasetSpec
from ..utils.paths import portable_path, resolve_repo_path

SPLIT_TYPES = ("train_val_test", "train_test")


def load_splits(dataset_dir, split_type="train_val_test"):
    """Read supplied CSVs, retaining source file/record columns and hash attrs.

    train_test explicitly merges train and validation; unknown modes fail.
    No source CSV is changed. Record positions are zero-based before merging.
    """
    if split_type not in SPLIT_TYPES:
        raise ValueError(f"Unknown split_type {split_type!r}; expected {SPLIT_TYPES}.")
    directory = resolve_repo_path(dataset_dir)
    data, hashes = {}, {}
    for split in ("train", "val", "test"):
        path = directory / f"{split}.csv"
        with path.open("rb") as stream:
            digest = sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        source = portable_path(path)
        hashes[source] = digest.hexdigest()
        frame = pd.read_csv(path, float_precision="round_trip")
        if SOURCE_FILE in frame or SOURCE_ROW in frame:
            raise ValueError(f"{path} contains reserved provenance columns.")
        frame[SOURCE_FILE] = source
        frame[SOURCE_ROW] = range(len(frame))
        data[split] = frame
    if split_type == "train_test":
        data = {"train": pd.concat([data["train"], data["val"]], ignore_index=True), "test": data["test"]}
    for frame in data.values():
        frame.attrs["source_hashes"] = hashes.copy()
    return data


def get_features_targets(data, data_spec: DatasetSpec):
    if data_spec.formulation == "boundary":
        raise ValueError("Use load_prepared_splits for boundary data to retain maturity offsets.")
    features = {split: prepare_features(df, data_spec)[0] for split, df in data.items()}
    targets = {split: df[data_spec.target_col] for split, df in data.items()}
    return features, targets


def load_prepared_splits(dataset_dir, data_spec: DatasetSpec, split_type="train_val_test"):
    frames = load_splits(dataset_dir, split_type)
    prepared = {name: prepare_split(frame, data_spec) for name, frame in frames.items()}
    metadata = {"source_hashes": next(iter(frames.values())).attrs["source_hashes"],
                "split_type": split_type, "rows": {name: len(p.labels) for name, p in prepared.items()},
                "data_spec": data_spec.to_dict(),
                "target_policy": "Recorded reached_birth; solver failures/timeouts retained"}
    return prepared, metadata


def load_data_pytorch(config, scaler=None):
    from ..models.nn.data import load_data_pytorch as load
    return load(config, scaler=scaler)


def load_data_gp(cfg):
    from ..models.gp.data import load_data_gp as load
    return load(cfg)
