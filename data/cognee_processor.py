"""
Cognee Data Processor

High-level orchestration for ingesting FX/crypto training data into Cognee Cloud.
Handles batch processing, progress tracking, and knowledge graph caching.

Features:
- ✅ Batch ingest GDELT news articles (handles millions of rows)
- ✅ Ingest economic indicators (Fed minutes, ECB announcements)
- ✅ Ingest price pattern narratives (OHLCV → text descriptions)
- ✅ Wait for cognify jobs with timeout
- ✅ Export entity/relationship cache for offline use

Usage:
    from data.cognee_client import CogneeClient
    from data.cognee_processor import CogneeDataProcessor

    client = CogneeClient(api_key=os.getenv("COGNEE_API_KEY"))
    processor = CogneeDataProcessor(client)

    # Ingest GDELT news
    processor.ingest_gdelt_news(gdelt_df, dataset_name="fx_eurusd_2023")

    # Trigger knowledge graph building
    job_id = client.cognify("fx_eurusd_2023")
    processor.wait_for_cognify(job_id, timeout=600)

    # Export for caching
    processor.export_entity_cache("fx_eurusd_2023", Path("cache/cognee"))
"""

import sys
import time
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from data.cognee_client import CogneeAPIError, CogneeClient
from utils.logger import get_logger

logger = get_logger(__name__)


class CogneeDataProcessor:
    """High-level orchestration for Cognee Cloud data ingestion."""

    def __init__(self, client: CogneeClient, batch_size: int = 100):
        """
        Initialize data processor.

        Args:
            client: Configured CogneeClient instance
            batch_size: Number of documents to ingest per batch
        """
        self.client = client
        self.batch_size = batch_size
        logger.info(f"[cognee] Initialized processor with batch_size={batch_size}")

    def ingest_gdelt_news(
            self,
            gdelt_df: pd.DataFrame,
            dataset_name: str,
            text_col: str = "source_url",
            include_themes: bool = True
    ) -> int:
        """
        Ingest GDELT news articles into Cognee.

        Args:
            gdelt_df: DataFrame with GDELT data (from gdelt_bigquery.py)
                Required columns: ['datetime', 'sentiment_score', 'source_url']
                Optional columns: ['themes']
            dataset_name: Name for Cognee dataset (e.g., "fx_eurusd_2023")
            text_col: Column to use as text content (default: source_url as identifier)
            include_themes: Whether to include GDELT themes in metadata

        Returns:
            Number of documents ingested

        Note:
            GDELT provides URLs and themes but not full article text.
            We ingest the metadata and let Cognee fetch/parse the articles if needed.
        """
        logger.info(f"[cognee] Ingesting {len(gdelt_df):,} GDELT records into '{dataset_name}'")

        if 'datetime' not in gdelt_df.columns or 'sentiment_score' not in gdelt_df.columns:
            raise ValueError("gdelt_df must contain 'datetime' and 'sentiment_score' columns")

        ingested_count = 0
        failed_count = 0

        # Process in batches to avoid memory issues and show progress
        for batch_start in range(0, len(gdelt_df), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(gdelt_df))
            batch = gdelt_df.iloc[batch_start:batch_end]

            texts = []
            metadatas = []

            for _, row in batch.iterrows():
                # Create text representation
                # Option 1: Use URL as identifier (Cognee can fetch content)
                # Option 2: Generate summary text from metadata
                text = row.get(text_col, "")

                if not text or pd.isna(text):
                    # Fallback: create text from datetime and sentiment
                    text = f"News event at {row['datetime']}: sentiment {row['sentiment_score']:.2f}"

                # Build metadata
                metadata = {
                    "datetime": str(row['datetime']),
                    "sentiment_score": float(row['sentiment_score']),
                    "source": "GDELT"
                }

                if include_themes and 'themes' in row and not pd.isna(row['themes']):
                    metadata['themes'] = str(row['themes'])

                texts.append(text)
                metadatas.append(metadata)

            # Ingest batch
            try:
                doc_ids = self.client.add_texts_batch(
                    dataset_name=dataset_name,
                    texts=texts,
                    metadatas=metadatas
                )
                ingested_count += len(doc_ids)

                if (batch_end % 1000) == 0 or batch_end == len(gdelt_df):
                    logger.info(
                        f"  Progress: {batch_end:,}/{len(gdelt_df):,} records ({batch_end / len(gdelt_df):.1%})")

            except CogneeAPIError as e:
                logger.error(f"  Failed to ingest batch {batch_start}-{batch_end}: {e}")
                failed_count += len(texts)
                continue

        logger.info(f"[cognee] GDELT ingestion complete: {ingested_count:,} successful, {failed_count:,} failed")
        return ingested_count

    def ingest_economic_indicators(
            self,
            indicators_df: pd.DataFrame,
            dataset_name: str,
            text_col: str = "full_text",
            title_col: str = "title"
    ) -> int:
        """
        Ingest economic indicator announcements into Cognee.

        Args:
            indicators_df: DataFrame with economic data
                Required columns: ['date', 'title', 'full_text']
                Optional columns: ['bank_name', 'event_type']
            dataset_name: Cognee dataset name
            text_col: Column containing full text content
            title_col: Column containing document title

        Returns:
            Number of documents ingested
        """
        logger.info(f"[cognee] Ingesting {len(indicators_df):,} economic indicators into '{dataset_name}'")

        if text_col not in indicators_df.columns:
            raise ValueError(f"indicators_df must contain '{text_col}' column")

        ingested_count = 0

        for batch_start in range(0, len(indicators_df), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(indicators_df))
            batch = indicators_df.iloc[batch_start:batch_end]

            texts = []
            metadatas = []

            for _, row in batch.iterrows():
                # Use full text as main content
                text = row[text_col]

                if not text or pd.isna(text):
                    logger.warning(f"Skipping row with empty text: {row.get(title_col, 'unknown')}")
                    continue

                # Build metadata
                metadata = {
                    "date": str(row.get("date", "")),
                    "title": str(row.get(title_col, "")),
                    "source": "economic_indicators"
                }

                # Add optional fields
                if "bank_name" in row:
                    metadata["bank_name"] = str(row["bank_name"])
                if "event_type" in row:
                    metadata["event_type"] = str(row["event_type"])

                texts.append(text)
                metadatas.append(metadata)

            # Ingest batch
            try:
                doc_ids = self.client.add_texts_batch(
                    dataset_name=dataset_name,
                    texts=texts,
                    metadatas=metadatas
                )
                ingested_count += len(doc_ids)

                logger.info(f"  Progress: {batch_end:,}/{len(indicators_df):,} records")

            except CogneeAPIError as e:
                logger.error(f"  Failed to ingest batch {batch_start}-{batch_end}: {e}")
                continue

        logger.info(f"[cognee] Economic indicators ingestion complete: {ingested_count:,} documents")
        return ingested_count

    def ingest_price_patterns(
            self,
            pattern_df: pd.DataFrame,
            dataset_name: str,
            pair: str,
            description_col: str = "pattern_description"
    ) -> int:
        """
        Ingest price pattern narratives into Cognee.

        Args:
            pattern_df: DataFrame with price pattern descriptions
                Required columns: ['datetime', 'pattern_description']
            dataset_name: Cognee dataset name
            pair: Currency pair (e.g., "EUR/USD")
            description_col: Column containing pattern text

        Returns:
            Number of documents ingested
        """
        logger.info(f"[cognee] Ingesting {len(pattern_df):,} price patterns for {pair} into '{dataset_name}'")

        if description_col not in pattern_df.columns or 'datetime' not in pattern_df.columns:
            raise ValueError(f"pattern_df must contain '{description_col}' and 'datetime' columns")

        ingested_count = 0

        for batch_start in range(0, len(pattern_df), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(pattern_df))
            batch = pattern_df.iloc[batch_start:batch_end]

            texts = []
            metadatas = []

            for _, row in batch.iterrows():
                text = row[description_col]

                if not text or pd.isna(text):
                    continue

                metadata = {
                    "datetime": str(row['datetime']),
                    "pair": pair,
                    "source": "price_patterns"
                }

                # Add OHLCV values if available
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in row:
                        metadata[col] = float(row[col])

                texts.append(text)
                metadatas.append(metadata)

            # Ingest batch
            try:
                doc_ids = self.client.add_texts_batch(
                    dataset_name=dataset_name,
                    texts=texts,
                    metadatas=metadatas
                )
                ingested_count += len(doc_ids)

                if (batch_end % 1000) == 0 or batch_end == len(pattern_df):
                    logger.info(f"  Progress: {batch_end:,}/{len(pattern_df):,} patterns")

            except CogneeAPIError as e:
                logger.error(f"  Failed to ingest batch {batch_start}-{batch_end}: {e}")
                continue

        logger.info(f"[cognee] Price patterns ingestion complete: {ingested_count:,} documents")
        return ingested_count

    def wait_for_cognify(
            self,
            job_id: str,
            timeout: int = 600,
            poll_interval: int = 5
    ) -> bool:
        """
        Wait for cognify job to complete with progress updates.

        Args:
            job_id: Job ID from client.cognify()
            timeout: Maximum wait time in seconds (default: 10 minutes)
            poll_interval: Seconds between status checks

        Returns:
            True if completed successfully, False if failed or timed out

        Example:
            >>> job_id = client.cognify("fx_eurusd")
            >>> success = processor.wait_for_cognify(job_id, timeout=300)
            >>> if success:
            ...     print("Knowledge graph ready!")
        """
        logger.info(f"[cognee] Waiting for cognify job {job_id} (timeout={timeout}s)")

        elapsed = 0
        last_progress = -1.0

        while elapsed < timeout:
            try:
                status = self.client.check_status(job_id)
                state = status["state"]
                progress = status.get("progress", 0.0)

                # Log progress updates
                if progress != last_progress:
                    logger.info(f"  Job {state}: {progress:.1%} complete")
                    last_progress = progress

                # Check terminal states
                if state == "completed":
                    logger.info(f"[cognee] ✅ Cognify job {job_id} completed successfully")
                    return True

                if state == "failed":
                    error_msg = status.get("message", "Unknown error")
                    logger.error(f"[cognee] ❌ Cognify job {job_id} failed: {error_msg}")
                    return False

                # Continue waiting
                time.sleep(poll_interval)
                elapsed += poll_interval

            except CogneeAPIError as e:
                logger.error(f"[cognee] Error checking job status: {e}")
                time.sleep(poll_interval)
                elapsed += poll_interval
                continue

        # Timeout
        logger.error(f"[cognee] ❌ Cognify job {job_id} timed out after {timeout}s")
        return False

    def export_entity_cache(
            self,
            dataset_name: str,
            output_dir: Path,
            include_relationships: bool = True
    ) -> dict[str, Path]:
        """
        Export entities and relationships from knowledge graph to local cache.

        This creates offline cache files that can be used for feature extraction
        without making API calls.

        Args:
            dataset_name: Cognee dataset name
            output_dir: Directory to save cache files
            include_relationships: Whether to export relationships (can be large)

        Returns:
            Dict mapping cache type to file path:
                {"entities": Path("..."), "relationships": Path("...")}

        Cache Format:
            entities_cache.feather: DataFrame with [name, type, mentions, metadata]
            relationships_cache.feather: DataFrame with [from, to, type, weight]
        """
        logger.info(f"[cognee] Exporting knowledge graph cache for '{dataset_name}'")

        output_dir.mkdir(parents=True, exist_ok=True)
        cache_files = {}

        # Export entities
        try:
            entities = self.client.get_entities(dataset_name, limit=10000)

            if entities:
                entities_df = pd.DataFrame(entities)
                entities_path = output_dir / f"{dataset_name}_entities.feather"
                entities_df.to_feather(entities_path)
                cache_files["entities"] = entities_path
                logger.info(f"  ✅ Exported {len(entities):,} entities to {entities_path}")
            else:
                logger.warning("  No entities found in knowledge graph")

        except CogneeAPIError as e:
            logger.error(f"  Failed to export entities: {e}")

        # Export relationships
        if include_relationships:
            try:
                relationships = self.client.get_relationships(dataset_name, limit=10000)

                if relationships:
                    relationships_df = pd.DataFrame(relationships)
                    relationships_path = output_dir / f"{dataset_name}_relationships.feather"
                    relationships_df.to_feather(relationships_path)
                    cache_files["relationships"] = relationships_path
                    logger.info(f"  ✅ Exported {len(relationships):,} relationships to {relationships_path}")
                else:
                    logger.warning("  No relationships found in knowledge graph")

            except CogneeAPIError as e:
                logger.error(f"  Failed to export relationships: {e}")

        logger.info(f"[cognee] Cache export complete: {len(cache_files)} files created")
        return cache_files


if __name__ == "__main__":
    # Example usage
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Test Cognee data processor")
    parser.add_argument("--api-key", help="Cognee API key")
    parser.add_argument("--dataset", default="test_processor", help="Test dataset name")
    parser.add_argument("--gdelt-sample", type=int, default=100, help="Number of GDELT rows to test")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("COGNEE DATA PROCESSOR TEST")
    logger.info("=" * 80)

    # Initialize client and processor
    client = CogneeClient(api_key=args.api_key or os.getenv("COGNEE_API_KEY"))
    processor = CogneeDataProcessor(client, batch_size=50)

    # Create sample GDELT data
    logger.info(f"\nCreating {args.gdelt_sample} sample GDELT records...")
    gdelt_sample = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=args.gdelt_sample, freq='1h'),
        'sentiment_score': pd.Series([0.5, -0.3, 0.8] * (args.gdelt_sample // 3 + 1))[:args.gdelt_sample],
        'source_url': [f"https://example.com/article_{i}" for i in range(args.gdelt_sample)],
        'themes': ['ECON_CURRENCY,TAX_FNCACT'] * args.gdelt_sample
    })

    # Test GDELT ingestion
    count = processor.ingest_gdelt_news(gdelt_sample, args.dataset)
    logger.info(f"✅ Ingested {count} GDELT records")

    # Trigger cognify
    job_id = client.cognify(args.dataset)
    logger.info(f"✅ Started cognify job: {job_id}")

    # Wait for completion
    success = processor.wait_for_cognify(job_id, timeout=120)

    if success:
        # Export cache
        cache_dir = ROOT / "data" / "cognee_cache"
        cache_files = processor.export_entity_cache(args.dataset, cache_dir)
        logger.info(f"✅ Exported cache: {list(cache_files.keys())}")

        # Cleanup
        client.delete_dataset(args.dataset)
        logger.info(f"✅ Cleaned up dataset '{args.dataset}'")

        logger.info("\n" + "=" * 80)
        logger.info("✅ PROCESSOR TEST PASSED")
        logger.info("=" * 80)
    else:
        logger.error("\n❌ PROCESSOR TEST FAILED")
