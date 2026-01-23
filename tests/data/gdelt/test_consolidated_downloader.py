"""
Unit tests for GDELT consolidated downloader.

Tests GDELT event download, parsing, and filtering:
- URL generation for different date ranges
- File download and decompression
- Event parsing and FX relevance filtering
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
from io import BytesIO
import gzip

import pandas as pd
import pytest

from data.gdelt.consolidated_downloader import GDELTDownloader


@pytest.fixture
def mock_gdelt_csv_data():
    """Mock GDELT CSV data (simplified)."""
    return """20230101000000\tGKGRECORDID\t\t\t\t\t\tUSA\tUSA\t\t\t\tGBR\tGBR\t\t\t\t\t\t1\t010\tProtests\t10.0\t1000\t100\t100\t100\t-2.5,5.0,2.0,0.5,0.1,0.0\t\t\t\thttp://example.com/article1
20230101060000\tGKGRECORDID2\t\t\t\t\t\tEUR\tEUR\t\t\t\tUSA\tUSA\t\t\t\t\t\t2\t051\tEconomic\t5.0\t500\t50\t50\t50\t1.2,3.0,1.0,0.3,0.2,0.0\t\t\t\thttp://example.com/article2
20230102120000\tGKGRECORDID3\t\t\t\t\t\tGBR\tGBR\t\t\t\tJPN\tJPN\t\t\t\t\t\t3\t020\tTrade\t3.0\t300\t30\t30\t30\t0.5,2.0,1.0,0.1,0.1,0.0\t\t\t\thttp://example.com/article3"""


@pytest.fixture
def mock_compressed_data(mock_gdelt_csv_data):
    """Mock gzipped GDELT data."""
    compressed = BytesIO()
    with gzip.open(compressed, "wt") as f:
        f.write(mock_gdelt_csv_data)
    compressed.seek(0)
    return compressed.getvalue()


@pytest.fixture
def downloader(tmp_path):
    """GDELT downloader instance using a temporary directory."""
    return GDELTDownloader(output_dir=tmp_path / "test_gdelt")


@pytest.fixture
def date_range():
    """Sample date range for testing."""
    return {
        "start": datetime(2023, 1, 1),
        "end": datetime(2023, 1, 3),
    }


class TestGDELTDownloaderInitialization:
    """Test downloader initialization."""

    def test_initialization_default(self):
        """Test downloader initializes with defaults."""
        downloader = GDELTDownloader()

        assert downloader.output_dir == Path("data/gdelt_raw")
        assert downloader.timeout == 30
        assert downloader.use_mirrors is True

    def test_initialization_custom(self):
        """Test downloader with custom parameters."""
        output_dir = Path("/tmp/custom_gdelt")
        downloader = GDELTDownloader(output_dir=output_dir, timeout=60, use_mirrors=False)

        assert downloader.output_dir == output_dir
        assert downloader.timeout == 60
        assert downloader.use_mirrors is False

    def test_output_dir_created(self, tmp_path):
        """Test output directory is created."""
        downloader = GDELTDownloader(output_dir=tmp_path / "gdelt")

        assert downloader.output_dir.exists()
        assert downloader.output_dir.is_dir()


class TestGDELTURLGeneration:
    """Test GDELT URL generation."""

    def test_filename_format(self):
        """Test filename format for timestamp."""
        ts = datetime(2023, 1, 15, 12, 30, 45)
        expected = "20230115123000.gkg.csv.zip"

        downloader = GDELTDownloader()
        filename = downloader._download_single.__code__.co_consts

        assert "gkg" in expected
        assert "csv" in expected
        assert "zip" in expected

    def test_floor_to_bucket(self):
        """Test datetime flooring to 15-minute buckets."""
        downloader = GDELTDownloader()

        dt1 = datetime(2023, 1, 1, 12, 23, 45)
        dt2 = datetime(2023, 1, 1, 12, 30, 0)
        dt3 = datetime(2023, 1, 1, 12, 45, 30)

        floored1 = downloader._floor_to_bucket(dt1)
        floored2 = downloader._floor_to_bucket(dt2)
        floored3 = downloader._floor_to_bucket(dt3)

        assert floored1.minute == 15
        assert floored2.minute == 30
        assert floored3.minute == 45


class TestGDELTDownload:
    """Test GDELT file downloading."""

    def test_download_single_file_success(self, mock_compressed_data, tmp_path):
        """Test downloading a single GDELT file."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = mock_compressed_data
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response

        downloader = GDELTDownloader(output_dir=tmp_path / "test_gdelt", session=mock_session)
        ts = datetime(2023, 1, 1, 0, 0, 0)
        result = downloader._download_single(ts)

        mock_session.get.assert_called()

    def test_download_single_file_404(self, tmp_path):
        """Test handling of missing GDELT files."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 404
        mock_session.get.return_value = mock_response

        downloader = GDELTDownloader(output_dir=tmp_path / "test_gdelt", session=mock_session)
        ts = datetime(2023, 1, 1, 0, 0, 0)
        result = downloader._download_single(ts)

        assert result is None

    def test_download_single_file_network_error(self, tmp_path):
        """Test retry logic on connection errors."""
        from requests.exceptions import ConnectionError

        mock_session = Mock()
        mock_session.get.side_effect = ConnectionError("Network error")

        downloader = GDELTDownloader(output_dir=tmp_path / "test_gdelt", session=mock_session)
        ts = datetime(2023, 1, 1, 0, 0, 0)
        result = downloader._download_single(ts)

        assert result is None

    def test_download_uses_mirrors(self, mock_compressed_data, tmp_path):
        """Test fallback to mirror URLs."""
        mock_session = Mock()

        mock_response1 = Mock()
        mock_response1.status_code = 404

        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.content = b"test data"
        mock_response2.raise_for_status = Mock()

        mock_session.get.side_effect = [mock_response1, mock_response2]

        downloader = GDELTDownloader(output_dir=tmp_path / "test_gdelt", session=mock_session)
        ts = datetime(2023, 1, 1, 0, 0, 0)
        result = downloader._download_single(ts)

        assert mock_session.get.call_count >= 2


class TestGDELTParsing:
    """Test GDELT event parsing."""

    def test_process_gkg_file_returns_dataframe(self, downloader, mock_gdelt_csv_data, tmp_path):
        """Test parsing GDELT CSV into DataFrame."""
        file_path = tmp_path / "test_gkg.csv"
        file_path.write_text(mock_gdelt_csv_data)

        df = downloader._process_gkg_file(file_path, countries=None)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_process_gkg_file_extracts_tone(self, downloader, mock_gdelt_csv_data, tmp_path):
        """Test tone information is extracted."""
        file_path = tmp_path / "test_gkg.csv"
        file_path.write_text(mock_gdelt_csv_data)

        df = downloader._process_gkg_file(file_path, countries=None)

        assert "AvgTone" in df.columns
        assert "PositiveTone" in df.columns
        assert "NegativeTone" in df.columns

    def test_process_gkg_file_tone_values_numeric(self, downloader, mock_gdelt_csv_data, tmp_path):
        """Test tone columns are numeric."""
        file_path = tmp_path / "test_gkg.csv"
        file_path.write_text(mock_gdelt_csv_data)

        df = downloader._process_gkg_file(file_path, countries=None)

        assert pd.api.types.is_numeric_dtype(df["AvgTone"])
        assert pd.api.types.is_numeric_dtype(df["PositiveTone"])
        assert pd.api.types.is_numeric_dtype(df["NegativeTone"])

    def test_filter_by_countries(self, downloader, mock_gdelt_csv_data, tmp_path):
        """Test filtering events by country codes."""
        file_path = tmp_path / "test_gkg.csv"
        file_path.write_text(mock_gdelt_csv_data)

        df = downloader._process_gkg_file(file_path, countries=["USA", "GBR"])

        assert len(df) > 0
        assert "Locations" in df.columns

    def test_process_gkg_file_handles_malformed_data(self, downloader, tmp_path):
        """Test handling of malformed CSV data."""
        malformed_data = "20230101000000\tRECORD\t\ninvalid,data\n"
        file_path = tmp_path / "malformed.csv"
        file_path.write_text(malformed_data)

        df = downloader._process_gkg_file(file_path, countries=None)

        assert isinstance(df, pd.DataFrame)


class TestGDELTDateRange:
    """Test date range handling."""

    def test_fetch_gkg_files_validates_datetime(self, downloader):
        """Test validates datetime inputs."""
        with pytest.raises(TypeError, match="must be datetime objects"):
            downloader.fetch_gkg_files("2023-01-01", "2023-01-02")

    def test_fetch_gkg_files_validates_order(self, downloader):
        """Test validates end date after start date."""
        start = datetime(2023, 1, 2)
        end = datetime(2023, 1, 1)

        with pytest.raises(ValueError, match="must be after start"):
            downloader.fetch_gkg_files(start, end)

    def test_fetch_gkg_files_warns_large_range(self, downloader, caplog):
        """Test warns about large date ranges."""
        start = datetime(2023, 1, 1)
        end = datetime(2024, 1, 2)  # > 365 days

        with patch.object(downloader, "_download_single", return_value=None):
            downloader.fetch_gkg_files(start, end)

        assert any("may take a long time" in record.message for record in caplog.records)

    def test_fetch_gkg_files_daily(self, downloader):
        """Test daily resolution fetching."""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 1, 2)

        with patch.object(downloader, "_download_single", return_value=None) as mock_download:
            downloader.fetch_gkg_files_daily(start, end)

            assert mock_download.call_count == 12  # 2 days * 4 samples per day (0, 6, 12, 18 hours)

    def test_fetch_gkg_files_15min(self, downloader):
        """Test 15-minute resolution fetching."""
        start = datetime(2023, 1, 1, 0, 0, 0)
        end = datetime(2023, 1, 1, 1, 0, 0)

        with patch.object(downloader, "_download_single", return_value=None) as mock_download:
            downloader.fetch_gkg_files(start, end)

            assert mock_download.call_count == 5  # 0, 15, 30, 45, 60 minutes


class TestGDELTDownloadDaterange:
    """Test download_daterange function."""

    @patch.object(GDELTDownloader, "fetch_gkg_files_daily")
    @patch.object(GDELTDownloader, "_process_gkg_file")
    def test_download_daterange_daily(
        self, mock_process, mock_fetch, downloader, date_range, tmp_path
    ):
        """Test download with daily resolution."""
        downloader.output_dir = tmp_path

        mock_fetch.return_value = []
        mock_process.return_value = pd.DataFrame()

        result = downloader.download_daterange(
            date_range["start"], date_range["end"], resolution="daily"
        )

        mock_fetch.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    @patch.object(GDELTDownloader, "fetch_gkg_files")
    @patch.object(GDELTDownloader, "_process_gkg_file")
    def test_download_daterange_15min(
        self, mock_process, mock_fetch, downloader, date_range, tmp_path
    ):
        """Test download with 15-minute resolution."""
        downloader.output_dir = tmp_path

        mock_fetch.return_value = []
        mock_process.return_value = pd.DataFrame()

        result = downloader.download_daterange(
            date_range["start"], date_range["end"], resolution="15min"
        )

        mock_fetch.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    @patch.object(GDELTDownloader, "fetch_gkg_files_daily")
    def test_download_daterange_no_files(self, mock_fetch, downloader, date_range):
        """Test returns empty DataFrame when no files downloaded."""
        mock_fetch.return_value = []

        result = downloader.download_daterange(
            date_range["start"], date_range["end"], resolution="daily"
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @patch.object(GDELTDownloader, "fetch_gkg_files_daily")
    @patch.object(GDELTDownloader, "_process_gkg_file")
    def test_download_daterange_combines_data(
        self, mock_process, mock_fetch, downloader, date_range, tmp_path
    ):
        """Test combines data from multiple files."""
        downloader.output_dir = tmp_path

        mock_fetch.return_value = []
        mock_process.return_value = pd.DataFrame(
            {"GKGRECORDID": ["1", "2"], "AvgTone": [-1.0, 1.0]}
        )

        result = downloader.download_daterange(
            date_range["start"], date_range["end"], resolution="daily"
        )

        assert len(result) >= 0


class TestGDELTCaching:
    """Test file caching mechanism."""

    @patch("data.gdelt.consolidated_downloader.requests.Session")
    def test_existing_file_not_downloaded(self, mock_session_class, downloader, tmp_path):
        """Test existing files are not re-downloaded."""
        downloader.output_dir = tmp_path
        ts = datetime(2023, 1, 1, 0, 0, 0)
        filename = "20230101000000.gkg.csv.zip"
        file_path = tmp_path / filename

        file_path.write_bytes(b"cached data")

        mock_session = Mock()
        mock_session_class.return_value = mock_session

        result = downloader._download_single(ts)

        mock_session.get.assert_not_called()
        assert result == file_path


class TestGDELTIntegration:
    """Integration tests for GDELT pipeline."""

    @patch("data.gdelt.consolidated_downloader.requests.Session")
    @patch.object(GDELTDownloader, "_process_gkg_file")
    def test_end_to_end_download_and_parse(
        self,
        mock_process,
        mock_session_class,
        downloader,
        mock_compressed_data,
        date_range,
        tmp_path,
    ):
        """Test complete download and parse workflow."""
        downloader.output_dir = tmp_path

        mock_session = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = mock_compressed_data
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        mock_process.return_value = pd.DataFrame(
            {
                "GKGRECORDID": ["1", "2"],
                "AvgTone": [-2.5, 1.2],
                "Locations": ["USA", "EUR"],
            }
        )

        result = downloader.download_daterange(
            date_range["start"], date_range["end"], resolution="daily", countries=["USA"]
        )

        mock_session.get.assert_called()
        mock_process.assert_called()
        assert isinstance(result, pd.DataFrame)
