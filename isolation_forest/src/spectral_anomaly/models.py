"""Interchangeable CPU sklearn and genuine GPU cuML model backends."""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
from .devices import resolve_backend

class AtypicalityModel:
    def __init__(self, *, backend="auto", model_type="isolation_forest", **params):
        self.backend=resolve_backend(backend); self.model_type=model_type; self.params=params; self.model=None
    def fit(self,x):
        if self.model_type != "isolation_forest": raise ValueError("unsupported atypicality model")
        if self.backend=="gpu":
            from cuml.ensemble import IsolationForest
        else:
            from sklearn.ensemble import IsolationForest
        self.model=IsolationForest(**self.params).fit(x); return self
    def score(self,x):
        if self.model is None: raise RuntimeError("model is not fitted")
        values=-self.model.score_samples(x)
        return np.asarray(values.get() if hasattr(values,"get") else values,float)


def save_artifact(value,path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,"wb") as stream: pickle.dump(value,stream)
def load_artifact(path):
    with open(path,"rb") as stream: return pickle.load(stream)
