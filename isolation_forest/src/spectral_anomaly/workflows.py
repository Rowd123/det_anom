"""Persistent stage workflows; none of these functions reads raw sensor data."""
from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
import uuid

import numpy as np
import pandas as pd

from .artifacts import (SCORE_CONVENTION, build_manifest, fingerprint, read_table,
                        validate_segment_table, write_table)
from .models import load_artifact, save_artifact
from .pipeline import train_atypicality
from .preprocessing import selected_features
from .spot import SPOT

LOGGER = logging.getLogger(__name__)


def _artifact_id(prefix: str, payload) -> str:
    return f"{prefix}-{fingerprint(payload)[:16]}"




def train_isolation_forest(segments_path, model_path, config):
    frame, source = _read_observations(segments_path)
    features = selected_features(config["preprocessing"], "atypicality", source.feature_columns)
    validate_segment_table(frame, features)
    started = perf_counter()
    if source.artifact_type == "window_features":
        config = {**config, "split": {**config["split"], "purge_overlap": True}}
    if source.provenance.get("split") == "train":
        # The persisted train partition is already split: use all of it.
        config = {**config, "split": {**config["split"], "train_fraction": 1.0,
                                    "calibration_fraction": 0.0, "evaluation_fraction": 0.0}}
    pipeline, splits = train_atypicality(frame, config)
    preprocessing_id = _artifact_id("prep", {"features": features, "log1p": pipeline.anomaly_preprocessor.log1p, "method": pipeline.anomaly_preprocessor.method, "center": pipeline.anomaly_preprocessor.center_.tolist(), "scale": pipeline.anomaly_preprocessor.scale_.tolist()})
    model_id = _artifact_id("iforest", {"config": config["atypicality"], "preprocessing_id": preprocessing_id, "nonce": uuid.uuid4().hex})
    artifact = {"kind": "isolation_forest", "schema_version": "1.0", "model_id": model_id,
                "preprocessing_id": preprocessing_id, "score_convention": SCORE_CONVENTION,
                "feature_columns": features, "source_config_hash": source.config_hash,
                "pipeline": pipeline, "config": config}
    save_artifact(artifact, model_path)
    LOGGER.info("trained Isolation Forest model_id=%s rows=%d backend=%s duration=%.3fs", model_id, len(splits[0]), pipeline.atypicality.backend, perf_counter()-started)
    return artifact


def score_isolation_forest(segments_path, model_path, output_path, *, batch_size=65536, resume=False):
    frame, source = _read_observations(segments_path)
    artifact = load_artifact(model_path)
    features = tuple(artifact["feature_columns"])
    validate_segment_table(frame, features)
    if source.config_hash != artifact["source_config_hash"]:
        LOGGER.warning("segment extraction config differs from training; feature contract is still compatible")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    scores = []
    started = perf_counter()
    for start in range(0, len(frame), batch_size):
        scores.append(artifact["pipeline"].score_atypicality(frame.iloc[start:start+batch_size]))
    result = frame[[c for c in frame if c.endswith("_id") or c in {"window_start", "window_end", "time_start", "time_end"}]].copy()
    result["model_id"] = artifact["model_id"]
    result["preprocessing_id"] = artifact["preprocessing_id"]
    result["score"] = np.concatenate(scores) if scores else np.empty(0)
    manifest = build_manifest("scores", config={"batch_size": batch_size}, provenance={"segments": str(segments_path), "model": str(model_path)}, model_id=artifact["model_id"], preprocessing_id=artifact["preprocessing_id"], score_convention=SCORE_CONVENTION)
    write_table(result, output_path, manifest, resume=resume)
    LOGGER.info("scored rows=%d batch_size=%d duration=%.3fs", len(result), batch_size, perf_counter()-started)
    return result






def apply_spot(calibration_path, scores_path, output_path, state_path, config, *, series_columns=("source_id", "channel_id"), resume=False):
    calibration, cm = read_table(calibration_path, expected_type="scores")
    scores, sm = read_table(scores_path, expected_type="scores")
    compatibility = (cm.model_id, cm.preprocessing_id, cm.score_convention)
    if compatibility != (sm.model_id, sm.preprocessing_id, sm.score_convention):
        raise ValueError("calibration and analysis scores use incompatible model/preprocessing/convention")
    missing = set(series_columns) - set(scores)
    if missing: raise ValueError(f"series columns missing from scores: {sorted(missing)}")
    order = [*series_columns, "window_start", "segment_id"]
    scores = scores.sort_values([x for x in order if x in scores], kind="stable")
    rows, states = [], {}
    for key, group in scores.groupby(list(series_columns), sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        selector = np.ones(len(calibration), bool)
        for column, value in zip(series_columns, key): selector &= calibration[column].eq(value).to_numpy()
        initial = calibration.loc[selector, "score"].to_numpy(float)
        spot = SPOT(**config).fit(initial)
        part = group.copy(); part["threshold"] = spot.threshold_; part["is_anomaly"] = spot.predict(part["score"])
        rows.append(part); states[str(key)] = spot
    result = pd.concat(rows, ignore_index=True) if rows else scores.assign(threshold=np.nan, is_anomaly=False)
    write_table(result, output_path, build_manifest("spot_decisions", config={**config, "series_columns": list(series_columns), "update_policy": "fixed"}, provenance={"calibration": str(calibration_path), "scores": str(scores_path)}, model_id=sm.model_id, preprocessing_id=sm.preprocessing_id, score_convention=sm.score_convention), resume=resume)
    save_artifact({"states": states, "last_keys": result[[c for c in order if c in result]].tail(1).to_dict("records"), "compatibility": compatibility}, state_path)
    return result


def _read_observations(path):
    frame, manifest = read_table(path)
    if manifest.artifact_type != "window_features":
        raise ValueError("models require window_features artifacts")
    return frame, manifest
