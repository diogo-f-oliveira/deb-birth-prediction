"""Small dataclass serialization helpers; construction and loading perform no IO writes."""
from dataclasses import fields, is_dataclass
import json
from pathlib import Path

from .paths import portable_path


def config_dict(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return {field.name: config_dict(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return portable_path(value)
    if isinstance(value, dict):
        return {key: config_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [config_dict(item) for item in value]
    return value


def save_config_json(config, path):
    path = Path(path)
    payload = json.dumps(config_dict(config), indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")


def validate_training_mode(config):
    """Fail before data access/output creation for unsupported training modes."""
    if config.data_splits != "train_val_test":
        raise ValueError("Training requires data_splits='train_val_test'; train_test is loader-only.")
    if config.data_spec.formulation == "boundary":
        raise NotImplementedError("Boundary training requires a fixed-offset loss (T05/T06); "
                                  "shared boundary preparation is available now.")
