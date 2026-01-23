"""Consolidated GDELT downloader with enhanced functionality."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from .config import GDELT_TIME_DELTA_MINUTES

logger = logging.getLogger(__name__)

BASE_URL = "https://data.gdeltproject.org/gdeltv2"
GDELT_GKG_EXTENSION = ".gkg.csv.zip"

HUGGINGFACE_MIRRORS = {
    "gdelt": BASE_URL,
    "hf-maxlong-2022": "https://huggingface.co/datasets/MaxLong/gdelt-news-headlines-2022/resolve/main",
    "hf-olm": "https://huggingface.co/datasets/olm/gdelt-news-headlines/resolve/main",
}


class GDELTDownloader:
    """Enhanced GDELT downloader with mirror support and data processing."""

    def __init__(
            self,
            output_dir: Path | str = Path("data/gdelt_raw"),
            session: requests.Session | None = None,
            timeout: int = 30,
            use_mirrors: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.timeout = timeout
        self.use_mirrors = use_mirrors
        self.mirrors = list(HUGGINGFACE_MIRRORS.values())
        self.current_mirror_idx = 0

    def download_daterange(
            self,
            start_date: datetime,
            end_date: datetime,
            countries: list[str] | None = None,
            resolution: str = "daily",
    ) -> pd.DataFrame:
        """Download and process GDELT data for a date range."""
        if resolution == "daily":
            files = self.fetch_gkg_files_daily(start_date, end_date)
        else:
            files = self.fetch_gkg_files(start_date, end_date)

        if not files:
            return pd.DataFrame()

        # Process and combine all files
        all_data = []
        for file_path in files:
            try:
                data = self._process_gkg_file(file_path, countries)
                if not data.empty:
                    all_data.append(data)
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            return combined
        return pd.DataFrame()

    def fetch_gkg_files(self, start_dt: datetime, end_dt: datetime) -> list[Path]:
        """Download GDELT GKG files between start_dt and end_dt (15-minute resolution)."""
        if not isinstance(start_dt, datetime) or not isinstance(end_dt, datetime):
            raise TypeError("start_dt and end_dt must be datetime objects")
        if end_dt < start_dt:
            raise ValueError(f"end_dt ({end_dt}) must be after start_dt ({start_dt})")

        # Warn about very large date ranges
        days_diff = (end_dt - start_dt).days
        if days_diff > 365:
            logger.warning(f"Date range spans {days_diff} days (>365), this may take a long time")

        start = self._floor_to_bucket(start_dt)
        end = self._floor_to_bucket(end_dt)
        current = start
        downloaded: list[Path] = []

        while current <= end:
            path = self._download_single(current)
            if path:
                downloaded.append(path)
            current += timedelta(minutes=GDELT_TIME_DELTA_MINUTES)

        return downloaded

    def fetch_gkg_files_daily(self, start_dt: datetime, end_dt: datetime) -> list[Path]:
        """Download GDELT GKG files with daily resolution."""
        downloaded: list[Path] = []
        current = start_dt.replace(hour=0, minute=0, second=0)

        while current.date() <= end_dt.date():
            # Sample a few times per day for better coverage
            for hour in [0, 6, 12, 18]:
                timestamp = current.replace(hour=hour)
                path = self._download_single(timestamp)
                if path:
                    downloaded.append(path)
            current += timedelta(days=1)

        return downloaded

    def _download_single(self, ts: datetime) -> Path | None:
        """Download a single GDELT file with mirror fallback."""
        ts = ts.astimezone(timezone.utc)
        timestamp = ts.strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}{GDELT_GKG_EXTENSION}"
        target_path = self.output_dir / filename

        if target_path.exists():
            logger.debug("GDELT file already present: %s", target_path)
            return target_path

        # Try primary URL and mirrors
        for attempt, base_url in enumerate(self.mirrors):
            url = f"{base_url}/{filename}"
            logger.info(f"Downloading GDELT GKG (attempt {attempt + 1}): {url}")

            try:
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 404:
                    if attempt == 0:
                        logger.warning("GDELT file missing from primary source: %s", timestamp)
                    continue

                resp.raise_for_status()
                target_path.write_bytes(resp.content)
                return target_path

            except requests.RequestException as e:
                logger.warning(f"Network error downloading from {base_url}: {e}")
                continue
            except (OSError, PermissionError) as e:
                logger.error(f"File system error writing to {target_path}: {e}")
                return None

        logger.error(f"Failed to download file from all mirrors: {filename}")
        return None

    def _process_gkg_file(
            self, file_path: Path, countries: list[str] | None = None
    ) -> pd.DataFrame:
        """Process a single GKG file and extract relevant data."""
        try:
            # GDELT GKG files are tab-separated
            df = pd.read_csv(file_path, sep="\t", low_memory=False, on_bad_lines="skip")

            # Basic column mapping (adjust based on actual GDELT schema)
            if len(df.columns) >= 15:
                df.columns = [
                    "GKGRECORDID",
                    "DATE",
                    "SourceCollectionIdentifier",
                    "SourceCommonName",
                    "DocumentIdentifier",
                    "Counts",
                    "V2Counts",
                    "Themes",
                    "V2Themes",
                    "Locations",
                    "V2Locations",
                    "Persons",
                    "V2Persons",
                    "Organizations",
                    "V2Organizations",
                    "V2Tone",
                    "Dates",
                    "GCAM",
                    "SharingImage",
                    "RelatedImages",
                    "SocialImageEmbeds",
                    "SocialVideoEmbeds",
                    "Quotations",
                    "AllNames",
                    "Amounts",
                    "TranslationInfo",
                    "Extras",
                ][: len(df.columns)]

            # Extract tone information
            if "V2Tone" in df.columns:
                tone_parts = df["V2Tone"].str.split(",")
                df["AvgTone"] = pd.to_numeric(tone_parts.str[0], errors="coerce")
                df["PositiveTone"] = pd.to_numeric(tone_parts.str[1], errors="coerce")
                df["NegativeTone"] = pd.to_numeric(tone_parts.str[2], errors="coerce")
                df["Polarity"] = pd.to_numeric(tone_parts.str[3], errors="coerce")
                df["ActivityReferenceDensity"] = pd.to_numeric(tone_parts.str[4], errors="coerce")
                df["SelfGroupReferenceDensity"] = pd.to_numeric(tone_parts.str[5], errors="coerce")

            # Filter by countries if specified
            if countries and "Locations" in df.columns:
                country_pattern = "|".join(countries)
                df = df[df["Locations"].str.contains(country_pattern, na=False, case=False)]

            # Add derived metrics
            df["NumMentions"] = 1  # Each row is one mention
            df["NumArticles"] = df.groupby("DocumentIdentifier").ngroup() + 1

            return df

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return pd.DataFrame()

    @staticmethod
    def _floor_to_bucket(dt: datetime) -> datetime:
        """Floor datetime to nearest 15-minute bucket."""
        dt = dt.astimezone(timezone.utc)
        minute_bucket = (dt.minute // GDELT_TIME_DELTA_MINUTES) * GDELT_TIME_DELTA_MINUTES
        return dt.replace(minute=minute_bucket, second=0, microsecond=0)
