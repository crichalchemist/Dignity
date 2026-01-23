import csv
import importlib.util
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARSER_PATH = PROJECT_ROOT / "data" / "gdelt" / "parser.py"

spec = importlib.util.spec_from_file_location("gdelt_parser", PARSER_PATH)
assert spec and spec.loader
gdelt_parser = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gdelt_parser
spec.loader.exec_module(gdelt_parser)
GDELTParser = gdelt_parser.GDELTParser


def _build_sample_zip(tmp_path: Path) -> Path:
    data_row = [
        "0",
        "20240101000000",
        "",
        "",
        "",
        "CountType#1.0",
        "THEME1;THEME2",
        "ENHTHEME#detail",
        "LOC#Name#0#USA#10#20",
        "Person1;Person2",
        "Org1;Org2",
        "0.1,0.2,0.3,0.4,0.5",
        "",
        "",
        "",
        "c1:1.5,c2:2.5",
    ]

    csv_path = tmp_path / "sample.gkgv2.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(data_row)

    zip_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(csv_path, arcname=csv_path.name)

    return zip_path


def test_parse_zip_file(tmp_path: Path):
    zip_path = _build_sample_zip(tmp_path)
    parser = GDELTParser()

    records = list(parser.parse_file(zip_path))

    assert len(records) == 1
    record = records[0]
    assert record.datetime.year == 2024
    assert record.themes == ["ENHANCED1", "ENHANCED2", "THEME1", "THEME2"]
    assert {count.type for count in record.counts} == {"STATENAME", "NUMARTS"}
    assert record.tone.polarity == 4.0
    assert record.counts[0].value == 1.0
    assert record.locations[0].country_code == "USA"
    assert record.locations[0].name == "New York"
    assert record.persons == ["Person1", "Person2"]
    assert record.orgs == ["Org1", "Org2"]
    assert record.gcam == {"c1": 0.5, "c2": 1.5}


def _load_parser_module():
    module_path = Path(__file__).resolve().parents[1] / "gdelt" / "parser.py"
    spec = importlib.util.spec_from_file_location("gdelt_parser", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
