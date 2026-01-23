"""
Downloader for GDELT 2.1 GKG files into a local folder.
Supports Hugging Face mirrors when gdeltproject.org is unavailable.

Example:
  python data/download_gdelt.py --start-date 2016-01-01 --end-date 2016-01-03 --resolution daily --out-dir data/gdelt
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BASE_URL = "https://data.gdeltproject.org/gdeltv2"

HUGGINGFACE_MIRRORS = {
    # News headline mirrors for when the primary GDELT endpoint is unavailable.
    "gdelt": BASE_URL,
    "hf-maxlong-2022": "https://huggingface.co/datasets/MaxLong/gdelt-news-headlines-2022/resolve/main",
    "hf-olm": "https://huggingface.co/datasets/olm/gdelt-news-headlines/resolve/main",
    "hf-andreas-helgesson": "https://huggingface.co/datasets/andreas-helgesson/gdelt-big/resolve/main",
}


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def iter_timestamps(start: datetime, end: datetime, step_minutes: int = 15) -> Iterator[datetime]:
    current = start
    delta = timedelta(minutes=step_minutes)
    while current <= end:
        yield current
        current += delta


def iter_days(start: datetime, end: datetime) -> Iterator[datetime]:
    current = start
    delta = timedelta(days=1)
    while current <= end:
        yield current
        current += delta


def floor_to_step(dt: datetime, step_minutes: int) -> datetime:
    return dt - timedelta(
        minutes=dt.minute % step_minutes,
        seconds=dt.second,
        microseconds=dt.microsecond,
    )


def _utcnow_naive() -> datetime:
    """Return the current UTC time as a naive datetime without deprecation warnings."""

    # datetime.UTC is only available in Python 3.11+; timezone.utc keeps compatibility
    # while ensuring we generate a naive UTC timestamp for downstream formatting.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def latest_available_end(resolution: str, step_minutes: int) -> datetime:
    """
    GDELT only publishes 15-minute batches a few minutes after the window ends
    and daily files the following day. Cap requests to avoid 404s on future windows.
    """
    now = _utcnow_naive()
    if resolution == "daily":
        return datetime(now.year, now.month, now.day) - timedelta(seconds=1)
    latest_slot = now - timedelta(minutes=step_minutes)
    return floor_to_step(latest_slot, step_minutes)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _load_http_dependencies() -> tuple[Any, Any, Any]:
    """
    Import requests and its retry helpers only when available, with an explicit
    error guiding users to install the project's dependencies.
    """

    if importlib.util.find_spec("requests") is None:
        raise ModuleNotFoundError(
            "Missing dependency: requests. Install it with `pip install -r requirements.txt` "
            "before running GDELT downloads."
        )

    requests = importlib.import_module("requests")
    adapters_spec = importlib.util.find_spec("requests.adapters")
    urllib3_retry_spec = importlib.util.find_spec("urllib3.util.retry")
    if adapters_spec is None or urllib3_retry_spec is None:
        missing = []
        if adapters_spec is None:
            missing.append("requests.adapters")
        if urllib3_retry_spec is None:
            missing.append("urllib3.util.retry")
        missing_list = ", ".join(missing)
        raise ModuleNotFoundError(
            f"Missing dependencies: {missing_list}. Install via `pip install -r requirements.txt`."
        )

    adapters = importlib.import_module("requests.adapters")
    urllib3_retry = importlib.import_module("urllib3.util.retry")
    return requests, adapters.HTTPAdapter, urllib3_retry.Retry


def download_file(
        session: Any,
        url: str,
        dest: Path,
        overwrite: bool,
        timeout: int,
        max_retries: int,
        backoff: float,
        expected_checksum: str | None,
) -> bool:
    if dest.exists() and not overwrite:
        if expected_checksum:
            local_checksum = sha256_file(dest)
            if local_checksum == expected_checksum:
                print(f"[skip] exists {dest.name} (checksum verified)")
                return True
            print(
                f"[warn] exists {dest.name} (checksum mismatch: local {local_checksum} vs expected {expected_checksum}); re-downloading"
            )
            dest.unlink()
        else:
            print(f"[skip] exists {dest.name} (no checksum provided)")
            return True
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 200:
                content = resp.content
                if expected_checksum:
                    checksum = sha256_bytes(content)
                    if checksum != expected_checksum:
                        print(
                            f"[fail] {dest.name} checksum mismatch (got {checksum}, expected {expected_checksum})"
                        )
                        # Treat as retryable corruption rather than accepting bad data.
                        raise ValueError("checksum mismatch")
                    print(f"[ok]   {dest.name} (checksum verified)")
                else:
                    print(f"[ok]   {dest.name} (no checksum provided)")
                dest.write_bytes(content)
                return True
            print(f"[warn] {dest.name} -> HTTP {resp.status_code}")
        except Exception as exc:
            print(f"[err]  {dest.name} attempt {attempt}/{max_retries} -> {exc}")
        if attempt < max_retries:
            sleep_time = backoff * attempt
            print(f"[info] retrying {dest.name} in {sleep_time:.1f}s...")
            time.sleep(sleep_time)

    print(f"[fail] {dest.name} after {max_retries} attempts")
    return False


def load_checksums(checksum_file: str | None) -> dict[str, str]:
    if not checksum_file:
        return {}
    path = Path(checksum_file)
    if not path.exists():
        raise FileNotFoundError(f"Checksum file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Checksum file must contain a JSON object mapping filenames to SHA256 values")
    return {str(key): str(value) for key, value in data.items()}


def resolve_base_url(mirror: str, base_override: str | None) -> str:
    if base_override:
        if not base_override.startswith("https://"):
            raise ValueError("--base-url must start with https:// to enforce TLS verification")
        return base_override.rstrip("/")

    if mirror not in HUGGINGFACE_MIRRORS:
        raise ValueError(
            f"Unknown mirror '{mirror}'. Choose from: {', '.join(sorted(HUGGINGFACE_MIRRORS))}"
        )

    return HUGGINGFACE_MIRRORS[mirror].rstrip("/")


def main():
    requests, HTTPAdapter, Retry = _load_http_dependencies()

    parser = argparse.ArgumentParser(description="Download GDELT 2.1 GKG zip files.")
    parser.add_argument("--start-date", default="2016-01-01", type=parse_date, help="YYYY-MM-DD (UTC) inclusive")
    parser.add_argument(
        "--end-date",
        default=_utcnow_naive().strftime("%Y-%m-%d"),
        type=parse_date,
        help="YYYY-MM-DD (UTC) inclusive",
    )
    parser.add_argument(
        "--resolution",
        choices=["daily", "15min"],
        default="daily",
        help="daily = one file per day; 15min = full 15-minute stream",
    )
    parser.add_argument(
        "--mirror",
        choices=sorted(HUGGINGFACE_MIRRORS.keys()),
        default="hf-maxlong-2022",
        help=(
            "Primary data source (defaults to the hf-maxlong-2022 mirror to avoid gdeltproject.org TLS issues). "
            "Options: gdelt, hf-maxlong-2022, hf-olm, hf-andreas-helgesson."
        ),
    )
    parser.add_argument(
        "--mirror-fallbacks",
        default="hf-olm,hf-andreas-helgesson,gdelt",
        help=(
            "Comma-separated list of mirrors to try if the primary source fails (ignored when --base-url is set). "
            "Defaults to other Hugging Face mirrors then the primary gdelt host."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the download base URL (https:// only). Takes precedence over --mirror.",
    )
    parser.add_argument("--step-minutes", type=int, default=15, help="Cadence between files when --resolution=15min")
    parser.add_argument("--out-dir", default="data/gdelt", help="Destination directory for zip files")
    parser.add_argument("--overwrite", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--timeout", type=int, default=10, help="HTTP timeout per request (seconds)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max attempts per file before giving up")
    parser.add_argument("--retry-backoff", type=float, default=2.0,
                        help="Seconds multiplied by attempt number between retries")
    parser.add_argument(
        "--ca-bundle",
        default=None,
        help="Optional CA bundle path for TLS verification when using private roots",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help=(
            "Disable TLS certificate verification. Only use when mirrors have hostname mismatches "
            "and you accept the risk."
        ),
    )
    parser.add_argument(
        "--checksum-file",
        type=str,
        default=None,
        help="Optional JSON mapping filenames to SHA256 digests for verification",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checksum_map = load_checksums(args.checksum_file)

    base_urls: list[tuple[str, str]] = []
    if args.base_url and args.mirror_fallbacks:
        print("[warn] Ignoring --mirror-fallbacks because --base-url is set.")

    if args.base_url:
        base_urls.append(("custom", resolve_base_url(args.mirror, args.base_url)))
    else:
        base_urls.append((args.mirror, resolve_base_url(args.mirror, args.base_url)))
        fallback_mirrors = [m.strip() for m in args.mirror_fallbacks.split(",") if m.strip()]
        for mirror in fallback_mirrors:
            if mirror == args.mirror:
                continue
            base_urls.append((mirror, resolve_base_url(mirror, None)))

    # Deduplicate any repeated base URLs while preserving order.
    seen_urls = set()
    unique_base_urls: list[tuple[str, str]] = []
    for mirror_name, base_url in base_urls:
        if base_url in seen_urls:
            continue
        seen_urls.add(base_url)
        unique_base_urls.append((mirror_name, base_url))
    base_urls = unique_base_urls
    base_url = resolve_base_url(args.mirror, args.base_url)

    start_dt = datetime(args.start_date.year, args.start_date.month, args.start_date.day)
    end_dt = datetime(args.end_date.year, args.end_date.month, args.end_date.day, 23, 59, 59)
    max_end_dt = latest_available_end(args.resolution, args.step_minutes)

    if args.resolution == "15min":
        start_dt = floor_to_step(start_dt, args.step_minutes)
        end_dt = floor_to_step(end_dt, args.step_minutes)

    if end_dt > max_end_dt:
        print(f"[info] Clamping end datetime to last available window {max_end_dt} UTC to avoid 404s")
        end_dt = max_end_dt

    if start_dt > end_dt:
        print(f"[warn] Start datetime {start_dt} is after available data ({end_dt}); nothing to download.")
        return

    if args.resolution == "daily":
        generator = iter_days(start_dt, end_dt)
        def formatter(dt):
            return dt.strftime("%Y%m%d")
    else:
        generator = iter_timestamps(start_dt, end_dt, step_minutes=args.step_minutes)
        def formatter(dt):
            return dt.strftime("%Y%m%d%H%M%S")

    session = requests.Session()
    if args.insecure:
        print("[warn] TLS verification disabled; downloads are susceptible to MITM attacks.")
        session.verify = False
    elif args.ca_bundle:
        session.verify = args.ca_bundle
    else:
        session.verify = True  # enforce TLS verification
    retry_cfg = Retry(total=0)
    adapter = HTTPAdapter(max_retries=retry_cfg)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    for ts in generator:
        stamp = formatter(ts)
        filename = f"{stamp}.gkg.csv.zip"
        url = f"{base_url}/{filename}"
        dest = out_dir / filename
        expected_checksum = checksum_map.get(filename)

        success = False
        for idx, (mirror_name, base_url) in enumerate(base_urls):
            if idx > 0:
                print(f"[info] switching to mirror '{mirror_name}' for {filename}")
            url = f"{base_url}/{filename}"
            success = download_file(
                session,
                url,
                dest,
                overwrite=args.overwrite,
                timeout=args.timeout,
                max_retries=args.max_retries,
                backoff=args.retry_backoff,
                expected_checksum=expected_checksum,
            )
            if success:
                break

        if not success:
            print(f"[fail] {filename} could not be downloaded from any configured source")


if __name__ == "__main__":
    main()
