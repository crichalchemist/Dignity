"""Baselines the difficulty gate scores; M3 reuses them."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data.pipeline import WindowSet
from data.source.synthetic import RAW_COLUMNS


def majority(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Constant scores equal to the train prevalence."""
    return np.full(n_test, float(np.mean(y_train)))


def risk_score_forecast(risk_score: np.ndarray, horizon: int) -> np.ndarray:
    """Early-warning baseline: today's risk score is the score at every horizon."""
    return np.repeat(risk_score[:, None], horizon, axis=1).astype(np.float32)


def _window_features(X: np.ndarray) -> np.ndarray:
    """[max, min, std] over time for every column, shape [N, 3F]."""
    return np.concatenate([X.max(axis=1), X.min(axis=1), X.std(axis=1)], axis=1)


def window_logistic(train: WindowSet, test: WindowSet) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(_window_features(train.X), train.y_risk)
    return model.predict_proba(_window_features(test.X))[:, 1]


def _block_features(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("block_id", sort=True)[list(RAW_COLUMNS)]
        .agg(["mean", "std", "max"])
        .fillna(0.0)
    )


def block_tree(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on noisy block_label; return (true block-has-anomaly, score)."""
    train_labels = df_train.groupby("block_id", sort=True)["block_label"].max()
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(_block_features(df_train).to_numpy(), train_labels.to_numpy())
    y = (
        df_test.groupby("block_id", sort=True)["is_anomalous"]
        .any()
        .to_numpy()
        .astype(np.int8)
    )
    return y, tree.predict_proba(_block_features(df_test).to_numpy())[:, 1]
