import importlib.util
import sys
import types
from datetime import timedelta
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


ROOT = Path(__file__).resolve().parents[1]

gdelt_pkg = types.ModuleType("gdelt")
gdelt_pkg.__path__ = [str(ROOT / "data" / "gdelt")]
sys.modules.setdefault("gdelt", gdelt_pkg)

config = _load_module("gdelt.config", ROOT / "data" / "gdelt" / "config.py")
sys.modules["gdelt"].config = config
alignment = _load_module("gdelt.alignment", ROOT / "data" / "gdelt" / "alignment.py")

align_candle_to_regime = alignment.align_candle_to_regime
iter_gdelt_buckets = alignment.iter_gdelt_buckets
GDELT_TIME_DELTA_MINUTES = config.GDELT_TIME_DELTA_MINUTES
get_gdelt_bucket_minutes = config.get_gdelt_bucket_minutes


def test_align_default_uses_config_bucket():
    bucket = get_gdelt_bucket_minutes()
    candle_time = datetime(2024, 1, 1, 10, 16, 45, tzinfo=timezone.utc)

    aligned = align_candle_to_regime(candle_time)

    expected_minute = (candle_time.minute // bucket) * bucket
    expected = candle_time.replace(minute=expected_minute, second=0, microsecond=0)
    assert aligned == expected
    assert bucket == GDELT_TIME_DELTA_MINUTES


def test_iter_gdelt_buckets_matches_alignment():
    bucket = get_gdelt_bucket_minutes()
    start = datetime(2024, 1, 1, 10, 2, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 10, 50, tzinfo=timezone.utc)

    buckets = iter_gdelt_buckets(start, end)

    assert buckets[0] == align_candle_to_regime(start, bucket_minutes=bucket)
    assert buckets[-1] == datetime(2024, 1, 1, 10, 45, tzinfo=timezone.utc)
    assert all(
        b2 - b1 == timedelta(minutes=bucket) for b1, b2 in zip(buckets, buckets[1:], strict=False)
    )
    assert bucket == GDELT_TIME_DELTA_MINUTES


"""Tests for GDELT alignment utilities."""
import sys
import types
from datetime import datetime, timezone
from importlib import util
from pathlib import Path


def _load_config_module():
    config_path = Path(__file__).resolve().parents[1] / "gdelt" / "config.py"
    spec = util.spec_from_file_location("gdelt.config", config_path)
    assert spec and spec.loader
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[spec.name] = module
    return module


def _load_alignment_module():
    if "gdelt" not in sys.modules:
        gdelt_module = types.ModuleType("gdelt")
        gdelt_module.__path__ = [str(Path(__file__).resolve().parents[1] / "gdelt")]
        sys.modules["gdelt"] = gdelt_module
    _load_config_module()
    alignment_path = Path(__file__).resolve().parents[1] / "gdelt" / "alignment.py"
    spec = util.spec_from_file_location("gdelt.alignment", alignment_path)
    assert spec and spec.loader
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_align_defaults_to_config_bucket_minutes() -> None:
    alignment = _load_alignment_module()
    config = _load_config_module()
    ts = datetime(2024, 1, 1, 0, 23, tzinfo=timezone.utc)

    aligned = alignment.align_candle_to_regime(ts)

    assert aligned == datetime(2024, 1, 1, 0, 15, tzinfo=timezone.utc)
    assert alignment.get_gdelt_bucket_minutes() == config.GDELT_TIME_DELTA_MINUTES


def test_iter_gdelt_buckets_respects_configured_bucket_size() -> None:
    alignment = _load_alignment_module()
    start = datetime(2024, 1, 1, 0, 7, tzinfo=timezone.utc)
    end = datetime(2024, 1, 1, 0, 52, tzinfo=timezone.utc)

    buckets = alignment.iter_gdelt_buckets(start, end)

    assert buckets == [
        datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 0, 15, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 0, 30, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 0, 45, tzinfo=timezone.utc),
    ]
