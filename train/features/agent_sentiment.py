from collections.abc import Callable, Sequence

import pandas as pd


def score_news(
        news_df: pd.DataFrame,
        scorer: Callable[[str], float],
        text_col: str = "headline",
        score_col: str = "sentiment_score",
) -> pd.DataFrame:
    """
    Apply a user-provided scoring function to each headline.
    scorer should map text -> float in [-1, 1] (negative to positive).
    """
    out = news_df.copy()
    out[score_col] = out[text_col].apply(scorer)
    return out


def aggregate_sentiment(
        news_df: pd.DataFrame,
        price_df: pd.DataFrame,
        time_col: str = "datetime",
        score_col: str = "sentiment_score",
        freq: str = "1min",
        rolling_windows: Sequence[int] = (5, 15, 60),
) -> pd.DataFrame:
    """
    Aggregate timestamped sentiment scores to the bar frequency and align to price_df.
    Only past news is used; no forward-fill of prior values into empty bars.
    """
    if score_col not in news_df.columns:
        raise ValueError(f"news_df must contain a '{score_col}' column")

    news = news_df.copy()
    news[time_col] = pd.to_datetime(news[time_col])
    price_times = pd.to_datetime(price_df[time_col])

    # Keep only news within the price horizon to avoid leakage from the future.
    news = news.loc[news[time_col] <= price_times.max()]

    per_freq = (
        news.set_index(time_col)
        .sort_index()
        .resample(freq)
        .agg({score_col: ["mean", "count", "std"]})
    )
    per_freq.columns = ["sent_mean_1m", "sent_count_1m", "sent_std_1m"]
    per_freq = per_freq.fillna({"sent_mean_1m": 0.0, "sent_count_1m": 0.0, "sent_std_1m": 0.0})

    features = per_freq.copy()
    for win in rolling_windows:
        # Require a full window to avoid partial/implicit carry-over.
        features[f"sent_mean_{win}m"] = per_freq["sent_mean_1m"].rolling(win, min_periods=win).mean()
        features[f"sent_std_{win}m"] = per_freq["sent_mean_1m"].rolling(win, min_periods=win).std()
        features[f"sent_count_{win}m"] = per_freq["sent_count_1m"].rolling(win, min_periods=win).sum()
        features[f"sent_ewm_{win}m"] = per_freq["sent_mean_1m"].ewm(span=win, min_periods=1).mean()

    # Align to the price timeline without carrying prior sentiment into empty bars.
    aligned = features.reindex(price_times).fillna(0.0)
    aligned.index = price_df.index
    return aligned.reset_index(drop=True)


def attach_sentiment_features(
        feature_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        drop_na: bool = True,
) -> pd.DataFrame:
    """
    Concatenate sentiment features (already time-aligned) to the existing feature frame.
    """
    merged = pd.concat([feature_df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)
    return merged.dropna() if drop_na else merged


def build_finbert_tone_scorer(model_dir: str = "models/finBERT-tone", device: int = -1) -> Callable[[str], float]:
    """
    Returns a callable that scores text with a local FinBERT-tone model.
    device: -1 for CPU, 0 for CUDA:0, etc.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TextClassificationPipeline,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=device,
        truncation=True,
    )
    label_sign = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    def _score(text: str) -> float:
        scores = pipe(text)[0]
        return sum(label_sign[s["label"].lower()] * s["score"] for s in scores)

    return _score
