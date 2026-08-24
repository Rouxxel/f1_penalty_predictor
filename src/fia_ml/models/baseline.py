"""Majority-class and session-stratified baseline classifiers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MajorityClassBaseline:
    """Always predict the most frequent class in the training set."""

    majority_class: int | None = None

    def fit(self, y: pd.Series | np.ndarray) -> MajorityClassBaseline:
        values = pd.Series(y).dropna().astype(int)
        if values.empty:
            raise ValueError("Cannot fit majority baseline on empty labels")
        self.majority_class = int(values.mode().iloc[0])
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.majority_class is None:
            raise RuntimeError("Model is not fitted")
        return np.full(n, self.majority_class, dtype=int)


@dataclass
class SessionStratifiedBaseline:
    """Per-session mode penalty class; falls back to global majority."""

    global_class: int | None = None
    session_modes: dict[str, int] = field(default_factory=dict)

    def fit(self, y: pd.Series | np.ndarray, session: pd.Series | np.ndarray) -> SessionStratifiedBaseline:
        labels = pd.Series(y).astype(int)
        sessions = pd.Series(session).astype(str)
        if labels.empty:
            raise ValueError("Cannot fit session baseline on empty labels")

        self.global_class = int(labels.mode().iloc[0])
        self.session_modes = {}
        for sess, group in labels.groupby(sessions):
            if str(sess).strip() and str(sess) != "nan":
                self.session_modes[str(sess)] = int(group.mode().iloc[0])
        return self

    def predict(self, session: pd.Series | np.ndarray) -> np.ndarray:
        if self.global_class is None:
            raise RuntimeError("Model is not fitted")
        preds = []
        for sess in pd.Series(session).astype(str):
            preds.append(self.session_modes.get(sess, self.global_class))
        return np.array(preds, dtype=int)
