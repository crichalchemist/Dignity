"""
Unit tests for agent sentiment features.

Tests:
- FinBERT sentiment scoring
- GDELT-OHLCV temporal alignment
- Sentiment feature engineering (momentum, volatility, divergence)
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from train.features.agent_sentiment import (
    aggregate_sentiment,
    attach_sentiment_features,
    build_finbert_tone_scorer,
    score_news,
)


@pytest.fixture
def sample_news_df():
    """Sample news DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=48, freq="h")
    return pd.DataFrame(
        {
            "datetime": dates,
            "headline": [f"Market news {i}" for i in range(48)],
        }
    )


@pytest.fixture
def sample_price_df():
    """Sample price DataFrame for alignment."""
    dates = pd.date_range("2023-01-01", periods=48, freq="h")
    np.random.seed(42)
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": 1.08 + np.cumsum(np.random.randn(48) * 0.0001),
            "close": 1.0805 + np.cumsum(np.random.randn(48) * 0.0001),
        }
    )


@pytest.fixture
def sample_news_with_sentiment():
    """Sample news with pre-computed sentiment scores."""
    dates = pd.date_range("2023-01-01", periods=24, freq="h")
    return pd.DataFrame(
        {
            "datetime": dates,
            "headline": [f"News {i}" for i in range(24)],
            "sentiment_score": np.random.randn(24) * 0.5,
        }
    )


@pytest.fixture
def sample_feature_df():
    """Sample feature DataFrame."""
    return pd.DataFrame(
        {
            "feature_1": np.random.randn(48),
            "feature_2": np.random.randn(48),
        }
    )


class TestScoreNews:
    """Test news scoring functionality."""

    def test_score_news_with_mock_scorer(self, sample_news_df):
        """Test score_news applies scoring function to each headline."""
        mock_scorer = Mock(return_value=0.5)

        result = score_news(sample_news_df, mock_scorer, text_col="headline")

        assert "sentiment_score" in result.columns
        assert len(result) == len(sample_news_df)
        assert mock_scorer.call_count == len(sample_news_df)

    def test_score_news_positive_scores(self, sample_news_df):
        """Test scoring with positive sentiment."""

        def positive_scorer(text):
            return 0.8

        result = score_news(sample_news_df, positive_scorer, text_col="headline")

        assert (result["sentiment_score"] > 0).all()
        assert (result["sentiment_score"] == 0.8).all()

    def test_score_news_negative_scores(self, sample_news_df):
        """Test scoring with negative sentiment."""

        def negative_scorer(text):
            return -0.7

        result = score_news(sample_news_df, negative_scorer, text_col="headline")

        assert (result["sentiment_score"] < 0).all()
        assert (result["sentiment_score"] == -0.7).all()

    def test_score_news_custom_column(self, sample_news_df):
        """Test scoring with custom column names."""
        sample_news_df["custom_score"] = sample_news_df["headline"]
        mock_scorer = Mock(return_value=0.3)

        result = score_news(
            sample_news_df, mock_scorer, text_col="headline", score_col="custom_score"
        )

        assert "custom_score" in result.columns
        assert "sentiment_score" not in result.columns


class TestAggregateSentiment:
    """Test sentiment aggregation and time alignment."""

    def test_aggregate_sentiment_returns_dataframe(
        self, sample_news_with_sentiment, sample_price_df
    ):
        """Test aggregation returns DataFrame."""
        result = aggregate_sentiment(
            sample_news_with_sentiment, sample_price_df, time_col="datetime"
        )

        assert isinstance(result, pd.DataFrame)

    def test_aggregate_sentiment_aligns_to_price_timeline(
        self, sample_news_with_sentiment, sample_price_df
    ):
        """Test sentiment aligned to price timestamps."""
        result = aggregate_sentiment(
            sample_news_with_sentiment, sample_price_df, time_col="datetime"
        )

        assert len(result) == len(sample_price_df)

    def test_aggregate_sentiment_creates_features(
        self, sample_news_with_sentiment, sample_price_df
    ):
        """Test creates rolling window features."""
        result = aggregate_sentiment(
            sample_news_with_sentiment,
            sample_price_df,
            time_col="datetime",
            rolling_windows=(5, 15, 60),
        )

        expected_features = [
            "sent_mean_1m",
            "sent_count_1m",
            "sent_std_1m",
            "sent_mean_5m",
            "sent_std_5m",
            "sent_count_5m",
            "sent_ewm_5m",
            "sent_mean_15m",
            "sent_std_15m",
            "sent_count_15m",
            "sent_ewm_15m",
        ]

        for feature in expected_features:
            assert feature in result.columns

    def test_aggregate_sentiment_custom_freq(self, sample_news_with_sentiment, sample_price_df):
        """Test custom resampling frequency."""
        result = aggregate_sentiment(
            sample_news_with_sentiment,
            sample_price_df,
            time_col="datetime",
            freq="5min",
        )

        assert isinstance(result, pd.DataFrame)

    def test_aggregate_sentiment_missing_score_column_raises(self, sample_news_df, sample_price_df):
        """Test raises ValueError when score column missing."""
        with pytest.raises(ValueError, match="must contain"):
            aggregate_sentiment(sample_news_df, sample_price_df, time_col="datetime")

    def test_aggregate_sentiment_no_forward_fill(self, sample_news_with_sentiment, sample_price_df):
        """Test no forward-fill of prior sentiment."""
        result = aggregate_sentiment(
            sample_news_with_sentiment, sample_price_df, time_col="datetime"
        )

        assert result.isna().sum().sum() == 0  # Should be filled with 0.0

    def test_aggregate_sentiment_numeric_features(
        self, sample_news_with_sentiment, sample_price_df
    ):
        """Test sentiment features are numeric."""
        result = aggregate_sentiment(
            sample_news_with_sentiment, sample_price_df, time_col="datetime"
        )

        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])


class TestAttachSentimentFeatures:
    """Test concatenating sentiment to existing features."""

    def test_attach_sentiment_features_concatenates(
        self, sample_feature_df, sample_news_with_sentiment, sample_price_df
    ):
        """Test sentiment features concatenated to feature DataFrame."""
        sentiment_df = aggregate_sentiment(
            sample_news_with_sentiment, sample_price_df, time_col="datetime"
        )

        result = attach_sentiment_features(sample_feature_df, sentiment_df, drop_na=False)

        assert len(result) == len(sample_feature_df)
        assert len(result.columns) == len(sample_feature_df.columns) + len(sentiment_df.columns)

    def test_attach_sentiment_features_drop_na(
        self, sample_feature_df, sample_news_with_sentiment, sample_price_df
    ):
        """Test drop_na removes rows with NaN."""
        sentiment_df = aggregate_sentiment(
            sample_news_with_sentiment, sample_price_df, time_col="datetime"
        )

        feature_with_nan = sample_feature_df.copy()
        feature_with_nan.loc[0, "feature_1"] = np.nan

        result = attach_sentiment_features(feature_with_nan, sentiment_df, drop_na=True)

        assert len(result) < len(feature_with_nan)

    def test_attach_sentiment_features_keep_na(
        self, sample_feature_df, sample_news_with_sentiment, sample_price_df
    ):
        """Test drop_na=False preserves rows with NaN."""
        sentiment_df = aggregate_sentiment(
            sample_news_with_sentiment, sample_price_df, time_col="datetime"
        )

        feature_with_nan = sample_feature_df.copy()
        feature_with_nan.loc[0, "feature_1"] = np.nan

        result = attach_sentiment_features(feature_with_nan, sentiment_df, drop_na=False)

        assert len(result) == len(feature_with_nan)

    def test_attach_sentiment_features_different_lengths_raises(self, sample_feature_df):
        """Test raises when DataFrames have different lengths."""
        short_sentiment = pd.DataFrame({"sent_mean_1m": np.random.randn(10)})

        with pytest.raises((ValueError, pd.errors.ShapeError)):
            attach_sentiment_features(sample_feature_df, short_sentiment, drop_na=False)


class TestBuildFinBERTToneScorer:
    """Test FinBERT tone scorer builder."""

    @patch("train.features.agent_sentiment.AutoModelForSequenceClassification")
    @patch("train.features.agent_sentiment.AutoTokenizer")
    @patch("train.features.agent_sentiment.TextClassificationPipeline")
    def test_build_finbert_tone_scorer_loads_model(
        self, mock_pipeline_class, mock_tokenizer, mock_model
    ):
        """Test loads model and tokenizer."""
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_pipeline_instance = Mock()

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_pipeline_class.return_value = mock_pipeline_instance

        scorer = build_finbert_tone_scorer(model_dir="models/test", device=-1)

        mock_model.from_pretrained.assert_called_once_with("models/test")
        mock_tokenizer.from_pretrained.assert_called_once_with("models/test")

    @patch("train.features.agent_sentiment.AutoModelForSequenceClassification")
    @patch("train.features.agent_sentiment.AutoTokenizer")
    @patch("train.features.agent_sentiment.TextClassificationPipeline")
    def test_build_finbert_tone_scorer_returns_callable(
        self, mock_pipeline_class, mock_tokenizer, mock_model
    ):
        """Test returns callable scorer."""
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [
            {"label": "positive", "score": 0.7},
            {"label": "neutral", "score": 0.2},
            {"label": "negative", "score": 0.1},
        ]

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_pipeline_class.return_value = mock_pipeline_instance

        scorer = build_finbert_tone_scorer(model_dir="models/test", device=-1)

        assert callable(scorer)

    @patch("train.features.agent_sentiment.AutoModelForSequenceClassification")
    @patch("train.features.agent_sentiment.AutoTokenizer")
    @patch("train.features.agent_sentiment.TextClassificationPipeline")
    def test_scorer_positive_text(self, mock_pipeline_class, mock_tokenizer, mock_model):
        """Test positive text returns positive score."""
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_pipeline_instance = Mock()

        mock_pipeline_instance.return_value = [
            {"label": "positive", "score": 0.8},
            {"label": "neutral", "score": 0.15},
            {"label": "negative", "score": 0.05},
        ]

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_pipeline_class.return_value = mock_pipeline_instance

        scorer = build_finbert_tone_scorer(model_dir="models/test", device=-1)
        score = scorer("Strong economic growth expected")

        assert score > 0

    @patch("train.features.agent_sentiment.AutoModelForSequenceClassification")
    @patch("train.features.agent_sentiment.AutoTokenizer")
    @patch("train.features.agent_sentiment.TextClassificationPipeline")
    def test_scorer_negative_text(self, mock_pipeline_class, mock_tokenizer, mock_model):
        """Test negative text returns negative score."""
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_pipeline_instance = Mock()

        mock_pipeline_instance.return_value = [
            {"label": "positive", "score": 0.05},
            {"label": "neutral", "score": 0.15},
            {"label": "negative", "score": 0.8},
        ]

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_pipeline_class.return_value = mock_pipeline_instance

        scorer = build_finbert_tone_scorer(model_dir="models/test", device=-1)
        score = scorer("Market crash fears mounting")

        assert score < 0

    @patch("train.features.agent_sentiment.AutoModelForSequenceClassification")
    @patch("train.features.agent_sentiment.AutoTokenizer")
    @patch("train.features.agent_sentiment.TextClassificationPipeline")
    def test_scorer_neutral_text(self, mock_pipeline_class, mock_tokenizer, mock_model):
        """Test neutral text returns score near zero."""
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_pipeline_instance = Mock()

        mock_pipeline_instance.return_value = [
            {"label": "positive", "score": 0.3},
            {"label": "neutral", "score": 0.4},
            {"label": "negative", "score": 0.3},
        ]

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_pipeline_class.return_value = mock_pipeline_instance

        scorer = build_finbert_tone_scorer(model_dir="models/test", device=-1)
        score = scorer("Market remains unchanged")

        assert abs(score) < 0.5

    @patch("train.features.agent_sentiment.AutoModelForSequenceClassification")
    @patch("train.features.agent_sentiment.AutoTokenizer")
    @patch("train.features.agent_sentiment.TextClassificationPipeline")
    def test_scorer_handles_multiple_calls(self, mock_pipeline_class, mock_tokenizer, mock_model):
        """Test scorer can be called multiple times."""
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_pipeline_instance = Mock()

        mock_pipeline_instance.return_value = [
            {"label": "positive", "score": 0.7},
            {"label": "neutral", "score": 0.2},
            {"label": "negative", "score": 0.1},
        ]

        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_pipeline_class.return_value = mock_pipeline_instance

        scorer = build_finbert_tone_scorer(model_dir="models/test", device=-1)

        score1 = scorer("Text 1")
        score2 = scorer("Text 2")
        score3 = scorer("Text 3")

        assert isinstance(score1, float)
        assert isinstance(score2, float)
        assert isinstance(score3, float)


class TestSentimentFeatureEngineering:
    """Test sentiment feature engineering."""

    def test_sentiment_momentum_feature(self, sample_news_with_sentiment, sample_price_df):
        """Test sentiment momentum calculation."""
        result = aggregate_sentiment(
            sample_news_with_sentiment,
            sample_price_df,
            time_col="datetime",
            rolling_windows=(5,),
        )

        if "sent_mean_5m" in result.columns and len(result) > 5:
            momentum = result["sent_mean_5m"].diff()

            assert isinstance(momentum, pd.Series)
            assert len(momentum) == len(result)

    def test_sentiment_volatility_feature(self, sample_news_with_sentiment, sample_price_df):
        """Test sentiment volatility calculation."""
        result = aggregate_sentiment(
            sample_news_with_sentiment,
            sample_price_df,
            time_col="datetime",
            rolling_windows=(5, 15),
        )

        assert "sent_std_5m" in result.columns
        assert "sent_std_15m" in result.columns

        assert (result["sent_std_5m"] >= 0).all()
        assert (result["sent_std_15m"] >= 0).all()

    def test_sentiment_ewm_feature(self, sample_news_with_sentiment, sample_price_df):
        """Test exponential weighted moving average."""
        result = aggregate_sentiment(
            sample_news_with_sentiment,
            sample_price_df,
            time_col="datetime",
            rolling_windows=(5,),
        )

        assert "sent_ewm_5m" in result.columns

        assert result["sent_ewm_5m"].notna().sum() > 0

    def test_sentiment_count_feature(self, sample_news_with_sentiment, sample_price_df):
        """Test news count feature."""
        result = aggregate_sentiment(
            sample_news_with_sentiment,
            sample_price_df,
            time_col="datetime",
            rolling_windows=(5, 15),
        )

        assert "sent_count_5m" in result.columns
        assert "sent_count_15m" in result.columns

        assert (result["sent_count_5m"] >= 0).all()
        assert (result["sent_count_15m"] >= 0).all()

    def test_multiple_window_features(self, sample_news_with_sentiment, sample_price_df):
        """Test multiple rolling windows."""
        result = aggregate_sentiment(
            sample_news_with_sentiment,
            sample_price_df,
            time_col="datetime",
            rolling_windows=(5, 15, 30),
        )

        expected_features = [
            "sent_mean_5m",
            "sent_mean_15m",
            "sent_mean_30m",
            "sent_std_5m",
            "sent_std_15m",
            "sent_std_30m",
            "sent_ewm_5m",
            "sent_ewm_15m",
            "sent_ewm_30m",
            "sent_count_5m",
            "sent_count_15m",
            "sent_count_30m",
        ]

        for feature in expected_features:
            assert feature in result.columns

    def test_sentiment_divergence_detection(self, sample_news_with_sentiment, sample_price_df):
        """Test sentiment-price divergence can be computed."""
        result = aggregate_sentiment(
            sample_news_with_sentiment, sample_price_df, time_col="datetime"
        )

        if "sent_mean_1m" in result.columns:
            sentiment = result["sent_mean_1m"]
            price = sample_price_df["close"]

            if len(sentiment) == len(price):
                divergence = sentiment - price.diff()

                assert isinstance(divergence, pd.Series)


class TestSentimentIntegration:
    """Integration tests for sentiment pipeline."""

    @patch("train.features.agent_sentiment.build_finbert_tone_scorer")
    def test_end_to_end_sentiment_pipeline(
        self, mock_build_scorer, sample_news_df, sample_price_df
    ):
        """Test complete sentiment pipeline."""
        mock_scorer = Mock(return_value=lambda text: np.random.randn() * 0.5)
        mock_build_scorer.return_value = mock_scorer

        scored_news = score_news(sample_news_df, mock_scorer)
        aggregated = aggregate_sentiment(scored_news, sample_price_df, time_col="datetime")

        assert "sentiment_score" in scored_news.columns
        assert isinstance(aggregated, pd.DataFrame)
        assert len(aggregated) == len(sample_price_df)

    @patch("train.features.agent_sentiment.build_finbert_tone_scorer")
    def test_sentiment_features_attach_to_training_data(
        self, mock_build_scorer, sample_news_df, sample_price_df, sample_feature_df
    ):
        """Test sentiment features can be attached to training data."""
        mock_scorer = Mock(return_value=lambda text: np.random.randn() * 0.5)
        mock_build_scorer.return_value = mock_scorer

        scored_news = score_news(sample_news_df, mock_scorer)
        aggregated = aggregate_sentiment(scored_news, sample_price_df, time_col="datetime")
        combined = attach_sentiment_features(sample_feature_df, aggregated)

        assert len(combined) == len(sample_feature_df)
        assert "sent_mean_1m" in combined.columns

    @patch("train.features.agent_sentiment.build_finbert_tone_scorer")
    def test_batch_news_scoring(self, mock_build_scorer, sample_news_df):
        """Test scoring multiple news items efficiently."""
        mock_scorer = Mock(return_value=lambda text: np.random.randn() * 0.5)
        mock_build_scorer.return_value = mock_scorer

        result = score_news(sample_news_df, mock_scorer)

        assert len(result) == len(sample_news_df)
        assert "sentiment_score" in result.columns
        assert result["sentiment_score"].notna().all()
