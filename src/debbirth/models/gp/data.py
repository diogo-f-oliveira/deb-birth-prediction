"""GP arrays backed by shared, row-aligned preparation."""
from ...data.load import load_prepared_splits


def load_data_gp(cfg, *, return_prepared=False):
    if cfg.data_spec.formulation == "boundary" and not return_prepared:
        raise ValueError("Boundary data require return_prepared=True to retain maturity offsets.")
    prepared, metadata = load_prepared_splits(cfg.data_dir, cfg.data_spec, cfg.data_splits)
    features = {name: split.features for name, split in prepared.items()}
    targets = {name: split.labels for name, split in prepared.items()}
    if return_prepared:
        return features, targets, prepared, metadata
    return features, targets
