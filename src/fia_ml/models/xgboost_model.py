"""XGBoost multiclass classifier for penalty_severity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from fia_ml.training.data_loaders import compute_sample_weights


@dataclass
class XGBoostTrainer:
    cfg_model: dict[str, Any]
    class_imbalance_strategy: str = "inverse_frequency"
    model: xgb.XGBClassifier | None = None
    best_iteration: int | None = None

    def _build_classifier(self) -> xgb.XGBClassifier:
        params = dict(self.cfg_model)
        early_stopping = int(params.pop("early_stopping_rounds", 30))
        num_class = int(params.pop("num_class", 3))
        params.setdefault("objective", "multi:softprob")
        params.setdefault("eval_metric", "mlogloss")
        params["num_class"] = num_class
        return xgb.XGBClassifier(
            **params,
            early_stopping_rounds=early_stopping,
            verbosity=0,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame,
        y_val: pd.Series | np.ndarray,
    ) -> XGBoostTrainer:
        y_train_arr = np.asarray(y_train, dtype=int)
        y_val_arr = np.asarray(y_val, dtype=int)

        self.model = self._build_classifier()
        sample_weight = compute_sample_weights(y_train_arr, self.class_imbalance_strategy)

        self.model.fit(
            X_train,
            y_train_arr,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val_arr)],
            verbose=False,
        )
        self.best_iteration = int(self.model.best_iteration)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        return self.model.predict(X).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        return self.model.predict_proba(X)

    def feature_importance(self) -> dict[str, dict[str, float]]:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        booster = self.model.get_booster()
        feature_names = list(booster.feature_names)

        def _map_scores(scores: dict[str, float]) -> dict[str, float]:
            mapped: dict[str, float] = {}
            for key, value in scores.items():
                if key.startswith("f") and key[1:].isdigit():
                    idx = int(key[1:])
                    name = feature_names[idx] if idx < len(feature_names) else key
                else:
                    name = key
                mapped[name] = float(value)
            return mapped

        return {
            "gain": _map_scores(booster.get_score(importance_type="gain")),
            "weight": _map_scores(booster.get_score(importance_type="weight")),
        }
