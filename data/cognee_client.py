"""
Cognee Cloud API Client

Lightweight wrapper around Cognee Cloud REST API for knowledge graph operations.
Handles authentication, request formatting, and error handling.

Features:
- ✅ Text ingestion into datasets
- ✅ Knowledge graph building (cognify)
- ✅ Job status polling
- ✅ Semantic search queries
- ✅ Dataset management

API Documentation: https://docs.cognee.ai/api-reference

Example:
    import os
    from data.cognee_client import CogneeClient

    client = CogneeClient(api_key=os.getenv("COGNEE_API_KEY"))

    # Ingest text
    client.add_text(
        dataset_name="fx_trading_eurusd",
        text="Federal Reserve announces 25bps rate hike",
        metadata={"date": "2023-05-03", "source": "GDELT"}
    )

    # Build knowledge graph
    job_id = client.cognify("fx_trading_eurusd")

    # Wait for completion
    while True:
        status = client.check_status(job_id)
        if status["state"] in ["completed", "failed"]:
            break
        time.sleep(5)

    # Search
    results = client.search("Fed rate hike", "fx_trading_eurusd")
"""

import os
import sys
import time
from pathlib import Path

import requests

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from utils.logger import get_logger

logger = get_logger(__name__)


class CogneeAPIError(Exception):
    """Exception raised for Cognee API errors."""

    def __init__(self, status_code: int, message: str, response: dict | None = None):
        self.status_code = status_code
        self.message = message
        self.response = response
        super().__init__(f"Cognee API Error {status_code}: {message}")


class CogneeClient:
    """Client for interacting with Cognee Cloud API."""

    def __init__(
            self,
            api_key: str | None = None,
            base_url: str = "https://api.cognee.ai",
            timeout: int = 30,
            max_retries: int = 3
    ):
        """
        Initialize Cognee API client.

        Args:
            api_key: Cognee API key (defaults to COGNEE_API_KEY env var)
            base_url: Base URL for Cognee API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests

        Raises:
            ValueError: If no API key is provided
        """
        self.api_key = api_key or os.getenv("COGNEE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Cognee API key not provided. Set COGNEE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries

        # Configure session with auth headers
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

        logger.info(f"[cognee] Initialized client for {self.base_url}")

    def _request(
            self,
            method: str,
            endpoint: str,
            json_data: dict | None = None,
            params: dict | None = None,
            retry_count: int = 0
    ) -> dict:
        """
        Make HTTP request to Cognee API with retry logic.

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path (e.g., "/datasets")
            json_data: JSON body for POST requests
            params: Query parameters
            retry_count: Current retry attempt

        Returns:
            Response JSON as dict

        Raises:
            CogneeAPIError: If request fails after retries
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=json_data,
                params=params,
                timeout=self.timeout
            )

            # Handle rate limiting
            if response.status_code == 429:
                if retry_count < self.max_retries:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning(f"[cognee] Rate limited, retrying after {retry_after}s")
                    time.sleep(retry_after)
                    return self._request(method, endpoint, json_data, params, retry_count + 1)
                else:
                    raise CogneeAPIError(429, "Rate limit exceeded", response.json())

            # Handle server errors with retry
            if response.status_code >= 500 and retry_count < self.max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff
                logger.warning(f"[cognee] Server error, retrying in {wait_time}s")
                time.sleep(wait_time)
                return self._request(method, endpoint, json_data, params, retry_count + 1)

            # Raise for other errors
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"[cognee] Request failed: {e}")
            raise CogneeAPIError(
                status_code=getattr(e.response, 'status_code', 0),
                message=str(e),
                response=getattr(e.response, 'json', lambda: None)()
            )

    def add_text(
            self,
            dataset_name: str,
            text: str,
            metadata: dict | None = None
    ) -> str:
        """
        Add text document to a dataset in Cognee.

        Args:
            dataset_name: Name of dataset to add text to
            text: Text content to ingest
            metadata: Optional metadata dict (e.g., {"date": "2023-05-03", "source": "GDELT"})

        Returns:
            Document ID (string)

        Example:
            >>> client.add_text(
            ...     dataset_name="fx_news",
            ...     text="ECB maintains interest rates at current levels",
            ...     metadata={"date": "2023-05-03", "theme": "ECON_CURRENCY"}
            ... )
            'doc_abc123'
        """
        payload = {
            "dataset": dataset_name,
            "content": text,
            "metadata": metadata or {}
        }

        result = self._request("POST", "/v1/documents", json_data=payload)
        doc_id = result.get("document_id", result.get("id", "unknown"))

        logger.debug(f"[cognee] Added text to dataset '{dataset_name}': {doc_id}")
        return doc_id

    def add_texts_batch(
            self,
            dataset_name: str,
            texts: list[str],
            metadatas: list[dict] | None = None
    ) -> list[str]:
        """
        Add multiple text documents in batch.

        Args:
            dataset_name: Name of dataset
            texts: List of text strings
            metadatas: Optional list of metadata dicts (same length as texts)

        Returns:
            List of document IDs
        """
        if metadatas is None:
            metadatas = [{}] * len(texts)

        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have same length")

        doc_ids = []
        for text, metadata in zip(texts, metadatas, strict=False):
            doc_id = self.add_text(dataset_name, text, metadata)
            doc_ids.append(doc_id)

        logger.info(f"[cognee] Added {len(doc_ids)} documents to '{dataset_name}'")
        return doc_ids

    def cognify(self, dataset_name: str) -> str:
        """
        Trigger knowledge graph building for a dataset.

        This starts an asynchronous job that extracts entities, relationships,
        and builds the knowledge graph from the ingested documents.

        Args:
            dataset_name: Name of dataset to process

        Returns:
            Job ID (string) for polling status

        Example:
            >>> job_id = client.cognify("fx_trading_eurusd")
            >>> print(job_id)
            'job_xyz789'
        """
        payload = {"dataset": dataset_name}
        result = self._request("POST", "/v1/cognify", json_data=payload)

        job_id = result.get("job_id", result.get("id", "unknown"))
        logger.info(f"[cognee] Started cognify job for '{dataset_name}': {job_id}")

        return job_id

    def check_status(self, job_id: str) -> dict:
        """
        Check status of a cognify job.

        Args:
            job_id: Job ID returned from cognify()

        Returns:
            Status dict with keys:
                - state: "pending" | "processing" | "completed" | "failed"
                - progress: float (0.0 to 1.0)
                - message: str (optional error message)
                - metadata: dict (additional job info)

        Example:
            >>> status = client.check_status("job_xyz789")
            >>> print(status["state"])
            'completed'
        """
        result = self._request("GET", f"/v1/jobs/{job_id}")

        # Normalize response format
        status = {
            "state": result.get("state", result.get("status", "unknown")),
            "progress": result.get("progress", 0.0),
            "message": result.get("message", ""),
            "metadata": result.get("metadata", {})
        }

        logger.debug(f"[cognee] Job {job_id} status: {status['state']} ({status['progress']:.1%})")
        return status

    def search(
            self,
            query: str,
            dataset_name: str,
            limit: int = 10,
            search_type: str = "semantic"
    ) -> list[dict]:
        """
        Perform semantic search on knowledge graph.

        Args:
            query: Search query (natural language)
            dataset_name: Dataset to search within
            limit: Maximum number of results
            search_type: "semantic" or "keyword"

        Returns:
            List of result dicts with keys:
                - text: str (matched text)
                - score: float (relevance score)
                - metadata: dict (document metadata)
                - entities: list (extracted entities)
                - relationships: list (related entities/events)

        Example:
            >>> results = client.search(
            ...     query="What caused USD to strengthen in May 2023?",
            ...     dataset_name="fx_trading_eurusd"
            ... )
            >>> for r in results[:3]:
            ...     print(f"{r['score']:.2f}: {r['text'][:100]}")
        """
        params = {
            "query": query,
            "dataset": dataset_name,
            "limit": limit,
            "type": search_type
        }

        result = self._request("GET", "/v1/search", params=params)
        results = result.get("results", result.get("data", []))

        logger.info(f"[cognee] Search returned {len(results)} results for query: '{query[:50]}...'")
        return results

    def get_entities(
            self,
            dataset_name: str,
            entity_type: str | None = None,
            limit: int = 100
    ) -> list[dict]:
        """
        Retrieve entities from knowledge graph.

        Args:
            dataset_name: Dataset name
            entity_type: Optional filter by entity type (e.g., "ORGANIZATION", "CURRENCY")
            limit: Maximum number of entities to return

        Returns:
            List of entity dicts with keys:
                - name: str
                - type: str
                - mentions: int
                - metadata: dict
        """
        params = {
            "dataset": dataset_name,
            "limit": limit
        }
        if entity_type:
            params["type"] = entity_type

        result = self._request("GET", "/v1/entities", params=params)
        entities = result.get("entities", result.get("data", []))

        logger.info(f"[cognee] Retrieved {len(entities)} entities from '{dataset_name}'")
        return entities

    def get_relationships(
            self,
            dataset_name: str,
            entity_name: str | None = None,
            limit: int = 100
    ) -> list[dict]:
        """
        Retrieve relationships from knowledge graph.

        Args:
            dataset_name: Dataset name
            entity_name: Optional filter by entity name
            limit: Maximum number of relationships

        Returns:
            List of relationship dicts with keys:
                - from: str (source entity)
                - to: str (target entity)
                - type: str (relationship type)
                - weight: float (relationship strength)
        """
        params = {
            "dataset": dataset_name,
            "limit": limit
        }
        if entity_name:
            params["entity"] = entity_name

        result = self._request("GET", "/v1/relationships", params=params)
        relationships = result.get("relationships", result.get("data", []))

        logger.info(f"[cognee] Retrieved {len(relationships)} relationships")
        return relationships

    def delete_dataset(self, dataset_name: str) -> bool:
        """
        Delete a dataset and all associated documents/graphs.

        Args:
            dataset_name: Name of dataset to delete

        Returns:
            True if deletion successful

        Warning:
            This operation is irreversible!
        """
        result = self._request("DELETE", f"/v1/datasets/{dataset_name}")
        success = result.get("success", result.get("deleted", False))

        if success:
            logger.info(f"[cognee] Deleted dataset '{dataset_name}'")
        else:
            logger.warning(f"[cognee] Failed to delete dataset '{dataset_name}'")

        return success

    def list_datasets(self) -> list[str]:
        """
        List all datasets in the account.

        Returns:
            List of dataset names
        """
        result = self._request("GET", "/v1/datasets")
        datasets = result.get("datasets", result.get("data", []))

        # Handle both list of strings and list of dicts
        if datasets and isinstance(datasets[0], dict):
            dataset_names = [d.get("name", d.get("id")) for d in datasets]
        else:
            dataset_names = datasets

        logger.info(f"[cognee] Found {len(dataset_names)} datasets")
        return dataset_names


if __name__ == "__main__":
    # Example usage and testing
    import argparse

    parser = argparse.ArgumentParser(description="Test Cognee API client")
    parser.add_argument("--api-key", help="Cognee API key (or set COGNEE_API_KEY)")
    parser.add_argument("--test-dataset", default="test_fx_trading", help="Dataset name for testing")
    parser.add_argument("--cleanup", action="store_true", help="Delete test dataset after completion")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("COGNEE API CLIENT TEST")
    logger.info("=" * 80)

    try:
        # Initialize client
        client = CogneeClient(api_key=args.api_key)
        logger.info("✅ Client initialized")

        # List datasets
        datasets = client.list_datasets()
        logger.info(f"✅ Listed {len(datasets)} existing datasets")

        # Add test documents
        test_texts = [
            "Federal Reserve announces 25 basis point rate hike to combat inflation",
            "ECB maintains current interest rate policy, citing economic uncertainty",
            "USD strengthens against EUR following strong non-farm payroll data"
        ]

        doc_ids = client.add_texts_batch(
            dataset_name=args.test_dataset,
            texts=test_texts,
            metadatas=[
                {"date": "2023-05-03", "source": "test"},
                {"date": "2023-05-04", "source": "test"},
                {"date": "2023-05-05", "source": "test"}
            ]
        )
        logger.info(f"✅ Added {len(doc_ids)} test documents")

        # Trigger cognify
        job_id = client.cognify(args.test_dataset)
        logger.info(f"✅ Started cognify job: {job_id}")

        # Poll status
        logger.info("Waiting for cognify to complete...")
        max_wait = 120  # 2 minutes
        elapsed = 0
        while elapsed < max_wait:
            status = client.check_status(job_id)
            logger.info(f"  Status: {status['state']} ({status['progress']:.1%})")

            if status["state"] in ["completed", "failed"]:
                break

            time.sleep(5)
            elapsed += 5

        if status["state"] == "completed":
            logger.info("✅ Cognify completed")

            # Test search
            results = client.search(
                query="interest rate policy",
                dataset_name=args.test_dataset,
                limit=5
            )
            logger.info(f"✅ Search returned {len(results)} results")

            # Get entities
            entities = client.get_entities(args.test_dataset, limit=20)
            logger.info(f"✅ Retrieved {len(entities)} entities")

        else:
            logger.error(f"❌ Cognify failed or timed out: {status['message']}")

        # Cleanup
        if args.cleanup:
            client.delete_dataset(args.test_dataset)
            logger.info(f"✅ Cleaned up test dataset '{args.test_dataset}'")

        logger.info("=" * 80)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
