"""Configuration of the retained Isolation Forest workflow."""
import json
from pathlib import Path
import numpy as np

def validate_config(data, kind="models"):
    if kind != "models":
        raise ValueError("only the models configuration is supported")
    required = {"preprocessing", "split", "atypicality", "spot", "reproducibility"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"missing models configuration section(s): {sorted(missing)}")
    result = dict(data)
    split = result["split"]
    fractions = [split[k] for k in ("train_fraction", "calibration_fraction", "evaluation_fraction")]
    if any(not np.isfinite(v) or v < 0 for v in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError("split fractions must be nonnegative and sum to one")
    if split.get("strategy", "chronological") != "chronological":
        raise ValueError("only chronological splitting is supported")
    return result

def load_config(path, kind="models"):
    with Path(path).open(encoding="utf-8") as stream:
        return validate_config(json.load(stream), kind)
