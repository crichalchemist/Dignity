"""Baselines the difficulty gate scores; M3 reuses them."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data.pipeline import WindowSet
from data.source.synthetic import RAW_COLUMNS
from models.head.policy import review_capacity


def majority(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Constant scores equal to the train prevalence."""
    return np.full(n_test, float(np.mean(y_train)))


def risk_score_forecast(risk_score: np.ndarray, horizon: int) -> np.ndarray:
    """Early-warning baseline: today's risk score is the score at every horizon."""
    return np.repeat(risk_score[:, None], horizon, axis=1).astype(np.float32)


def threshold_policy(
    val_scores: np.ndarray,
    val_y: np.ndarray,
    test_scores: np.ndarray,
    costs: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Two thresholds on the risk score, tuned on val under the review budget.

    A score >= t_block is blocked (block never counts toward the budget), a
    score in [t_review, t_block) is reviewed, anything else is allowed. The pair
    with the lowest val mean cost whose val review rate fits the budget wins
    (first on ties). On test, reviews beyond review_capacity keep only the
    highest scores; the rest go to allow. Returns int64 [N].
    """
    costs = costs.astype(np.float64)  # float64: ties must not depend on NEP 50
    val_scores = np.asarray(val_scores, dtype=np.float64)
    val_y = np.asarray(val_y).astype(np.int64)
    grid = np.unique(np.append(np.quantile(val_scores, np.linspace(0, 1, 101)), np.inf))

    best_cost, best = np.inf, (np.inf, np.inf)
    for t_review in grid:
        for t_block in grid[grid >= t_review]:
            blocked = val_scores >= t_block
            reviewed = (val_scores >= t_review) & ~blocked
            if reviewed.mean() > budget + 1e-12:
                continue
            actions = np.where(blocked, 2, np.where(reviewed, 1, 0))
            cost = costs[actions, val_y].mean()
            if cost < best_cost:
                best_cost, best = cost, (t_review, t_block)

    t_review, t_block = best
    test_scores = np.asarray(test_scores, dtype=np.float64)
    actions = np.where(
        test_scores >= t_block, 2, np.where(test_scores >= t_review, 1, 0)
    ).astype(np.int64)
    review = np.flatnonzero(actions == 1)
    k = review_capacity(len(test_scores), budget)
    if len(review) > k:
        order = review[np.argsort(-test_scores[review], kind="stable")]
        actions[order[k:]] = 0
    return actions


def _window_features(X: np.ndarray) -> np.ndarray:
    """[max, min, std] over time for every column, shape [N, 3F]."""
    return np.concatenate([X.max(axis=1), X.min(axis=1), X.std(axis=1)], axis=1)


def window_logistic(train: WindowSet, test: WindowSet) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(_window_features(train.X), train.y_risk)
    return model.predict_proba(_window_features(test.X))[:, 1]


def _forecast_features(X: np.ndarray) -> np.ndarray:
    """`_window_features`, last-10, last-30 and whole-window means, two slopes.

    The slopes are the mean of the last 10 rows minus the mean of rows -30 to -10,
    and the mean of the last 5 rows minus the mean of rows -15 to -5.
    """
    m10 = X[:, -10:].mean(axis=1)
    m30 = X[:, -30:].mean(axis=1)
    mall = X.mean(axis=1)
    slow = m10 - X[:, -30:-10].mean(axis=1)
    fast = X[:, -5:].mean(axis=1) - X[:, -15:-5].mean(axis=1)
    return np.concatenate([_window_features(X), m10, m30, mall, slow, fast], axis=1)


def _probe_inputs(ws: WindowSet, rows, risk_aware: bool) -> np.ndarray:
    """`_forecast_features` of the rows, plus `y_risk` as one more column if asked."""
    features = _forecast_features(ws.X[rows])
    if not risk_aware:
        return features
    return np.concatenate([features, ws.y_risk[rows, None].astype(float)], axis=1)


def _fit_forecast_probe(
    train: WindowSet, risk_aware: bool = False
) -> Callable[[WindowSet], np.ndarray]:
    """Fit one logistic probe per horizon on the windows `forecast_mask` keeps.

    Returns a function that scores any WindowSet: probabilities [N, H], every
    row scored. `risk_aware` gives the probe `y_risk` as one more input column.
    """
    keep = train.forecast_mask
    x_train = _probe_inputs(train, keep, risk_aware)
    models = []
    for h in range(train.y_forecast.shape[1]):
        y = train.y_forecast[keep, h]
        if len(np.unique(y)) < 2:
            raise ValueError(f"horizon {h}: one class in the masked train windows")
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=0.3, max_iter=1000)
        )
        models.append(model.fit(x_train, y))

    def score(ws: WindowSet) -> np.ndarray:
        x = _probe_inputs(ws, slice(None), risk_aware)
        return np.stack([m.predict_proba(x)[:, 1] for m in models], axis=1)

    return score


def window_logistic_forecast(
    train: WindowSet, test: WindowSet, risk_aware: bool = False
) -> np.ndarray:
    """One logistic probe per horizon on the windows `forecast_mask` keeps.

    Returns probabilities [N_test, H]; every test row is scored. With
    `risk_aware` the probe is also told, as `y_risk`, whether an anomaly lies
    earlier in the window.
    """
    return _fit_forecast_probe(train, risk_aware)(test)


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
