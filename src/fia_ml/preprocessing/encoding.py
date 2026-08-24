"""Encode features and impute missing values (fit on train only)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from fia_ml.preprocessing.leakage_filter import (
    V1_BOOLEAN_FEATURES,
    V1_CATEGORICAL_FEATURES,
    V1_NUMERIC_FEATURES,
    assert_no_leakage,
)
from fia_ml.utils import secure_file_io as sio

TARGET_COLUMN = "penalty_severity"
METADATA_COLUMNS = ("penalty", "row_id", "incident_id", "session")


@dataclass
class EncodingArtifacts:
    feature_columns: list[str]
    preprocessor: ColumnTransformer
    encoder_path: Path | None = None


def _bool_to_int(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    return series.map({True: 1, False: 0, "True": 1, "False": 0}).astype("float")


def _prepare_raw_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    out = df[feature_columns].copy()
    for col in feature_columns:
        if col in V1_BOOLEAN_FEATURES:
            out[col] = _bool_to_int(out[col])
        elif col in V1_NUMERIC_FEATURES:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = out[col].astype(str).replace({"": np.nan, "nan": np.nan})
    return out


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    cat_cols = [c for c in feature_columns if c in V1_CATEGORICAL_FEATURES]
    num_cols = [c for c in feature_columns if c in V1_NUMERIC_FEATURES or c in V1_BOOLEAN_FEATURES]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            )
        )
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def fit_encode_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, EncodingArtifacts]:
    assert_no_leakage(feature_columns)

    preprocessor = build_preprocessor(feature_columns)
    train_raw = _prepare_raw_features(train_df, feature_columns)
    preprocessor.fit(train_raw)

    def transform_split(split_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        if split_df.empty:
            return split_df
        encoded = preprocessor.transform(_prepare_raw_features(split_df, feature_columns))
        col_names = preprocessor.get_feature_names_out()
        feature_frame = pd.DataFrame(encoded, columns=col_names, index=split_df.index)
        for meta in (*METADATA_COLUMNS, TARGET_COLUMN):
            if meta in split_df.columns:
                feature_frame[meta] = split_df[meta].values
        feature_frame["split"] = split_name
        return feature_frame

    train_out = transform_split(train_df, "train")
    val_out = transform_split(val_df, "validation") if not val_df.empty else val_df
    test_out = transform_split(test_df, "test") if test_df is not None and not test_df.empty else test_df

    artifacts = EncodingArtifacts(feature_columns=feature_columns, preprocessor=preprocessor)
    return train_out, val_out, test_out, artifacts


def save_encoding_artifacts(artifacts: EncodingArtifacts, path: Path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.preprocessor, path)
    meta = {
        "feature_columns": artifacts.feature_columns,
        "output_columns": list(artifacts.preprocessor.get_feature_names_out()),
    }
    sio.write_json(path.with_suffix(".meta.json"), meta)
