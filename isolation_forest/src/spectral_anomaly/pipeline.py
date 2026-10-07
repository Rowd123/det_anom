"""Isolation Forest training and scoring, retaining historical pickle paths."""
from __future__ import annotations
from dataclasses import dataclass
import logging
from time import perf_counter
import numpy as np
import pandas as pd
from .models import AtypicalityModel
from .preprocessing import FeaturePreprocessor, chronological_split, selected_features

LOGGER = logging.getLogger(__name__)

def _elapsed(start):
    return perf_counter() - start


def _window_count(frame):
    """Count logical windows without materialising another copy of the rows."""
    if frame.empty:
        return 0
    return int(frame["window_id"].nunique()) if "window_id" in frame else len(frame)




def _assert_finite_features(frame, features, *, label):
    values = frame.loc[:, features].to_numpy(dtype=float)
    invalid = {
        "NaN": int(np.isnan(values).sum()),
        "+inf": int(np.isposinf(values).sum()),
        "-inf": int(np.isneginf(values).sum()),
    }
    if any(invalid.values()):
        LOGGER.error(
            "[feature-validation] partition=%s non_finite=%s action=abort_no_exclusion",
            label, invalid,
        )
        details = ", ".join(f"{name}={count}" for name, count in invalid.items())
        raise ValueError(f"non-finite atypicality features in {label}: {details}")


@dataclass
class ModelPipeline:
    """Isolation Forest pipeline with persisted preprocessing and feature contract."""

    anomaly_preprocessor: FeaturePreprocessor
    atypicality: AtypicalityModel
    spectral_signature: str | None = None
    exclusion_signature: str | None = None
    window_feature_signature: str | None = None

    def score_atypicality(self, frame):
        """Return raw ``-score_samples`` values (higher means more atypical)."""
        self._validate_compatibility(frame)
        _assert_finite_features(
            frame, self.anomaly_preprocessor.features, label="scoring data"
        )
        transformed = self.anomaly_preprocessor.transform(frame)
        return self.atypicality.score(transformed)

    def _validate_compatibility(self, frame):
        for column, expected in (("spectral_signature", self.spectral_signature),
                                 ("exclusion_signature", self.exclusion_signature),
                                 ("window_feature_signature", getattr(self, "window_feature_signature", None))):
            if expected is None: continue
            if column not in frame or set(frame[column].dropna().astype(str)) != {expected}:
                raise ValueError(f"scoring data has incompatible {column}")

    def analyze(self, frame):
        """Attach only the continuous atypicality score to segment rows."""
        result = frame.copy()
        result["atypicality_score"] = self.score_atypicality(frame)
        return result


def train_atypicality(frame, config):
    """Fit preprocessing and Isolation Forest on train windows only."""
    train, calibration, evaluation = chronological_split(frame, config["split"])
    preprocessing = config["preprocessing"]
    features = selected_features(preprocessing, "atypicality", frame.columns)
    signatures = {}
    for column in ("spectral_signature", "exclusion_signature", "window_feature_signature"):
        if column in train:
            values = set(train[column].dropna().astype(str))
            if len(values) != 1: raise ValueError(f"training data mixes incompatible {column} values")
            signatures[column] = next(iter(values))
    representations = set(train["spectral_representation"].dropna()) if "spectral_representation" in train else {"stft"}
    if len(representations) != 1:
        raise ValueError("training data mixes incompatible spectral representations")
    LOGGER.info("[split] features=%s", list(features))
    for name, split in zip(("train", "calibration", "evaluation"), (train, calibration, evaluation)):
        LOGGER.info("[split] partition=%s windows=%d segments=%d", name, _window_count(split), len(split))
    if "observation_kind" in frame and set(frame["observation_kind"]) == {"window"}:
        if any(not name.startswith("wf_") for name in features):
            raise ValueError("window models require extracted wf_ features")
    else:
        raise ValueError("training requires whole-window features (observation_kind=window)")
    if train.empty:
        raise ValueError("the chronological train split contains no segments")
    for name, split in (
        ("train", train),
        ("calibration", calibration),
        ("evaluation", evaluation),
    ):
        _assert_finite_features(split, features, label=name)
    preprocessor = FeaturePreprocessor(
        features,
        tuple(name for name in preprocessing.get("log1p", []) if name in features),
        preprocessing.get("scaling", "robust"),
    )
    preprocessing_started = perf_counter()
    LOGGER.info("[preprocessing] starting train_matrix=(%d, %d)", len(train), len(features))
    transformed_train = preprocessor.fit_transform(train)
    LOGGER.info("[preprocessing] finished in %.3fs output_shape=%s", _elapsed(preprocessing_started), transformed_train.shape)
    model_config = dict(config["atypicality"])
    backend = model_config.pop("backend")
    model_type = model_config.pop("type")
    model_config.setdefault("random_state", config["reproducibility"]["random_seed"])
    fit_started = perf_counter()
    LOGGER.info("[isolation-forest-fit] starting backend=%s parameters=%s", backend, model_config)
    atypicality = AtypicalityModel(
        backend=backend, model_type=model_type, **model_config
    ).fit(transformed_train)
    LOGGER.info("[isolation-forest-fit] finished in %.3fs resolved_backend=%s", _elapsed(fit_started), atypicality.backend)
    return ModelPipeline(preprocessor, atypicality, signatures.get("spectral_signature"),
                         signatures.get("exclusion_signature"), signatures.get("window_feature_signature")), (train, calibration, evaluation)


def score_atypicality_splits(pipeline, splits):
    """Score all splits with one fitted preprocessor/model and retain row metadata."""
    results = []
    for label, frame in zip(("train", "calibration", "evaluation"), splits):
        started = perf_counter()
        LOGGER.info("[scoring] starting partition=%s matrix=(%d, %d)", label, len(frame), len(pipeline.anomaly_preprocessor.features))
        scored = pipeline.analyze(frame)
        scored["split"] = label
        results.append(scored)
        LOGGER.info("[scoring] finished partition=%s in %.3fs scores=%d", label, _elapsed(started), len(scored))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def train_models(frame, config):
    """Backward-compatible name for Isolation-Forest-only training."""
    return train_atypicality(frame, config)
