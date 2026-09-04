"""Public docs must not reintroduce claims or APIs this project removed."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Removed claims and removed API names. Matching is case-insensitive. Historical
# records under docs/plans/ and docs/superpowers/ are allowed to mention them;
# public docs are not.
REMOVED = (
    "Secure Aggregation",
    "hash_identifiers(",
    "anonymize_addresses",
    "sanitize_dataset",
    "suppress_rare_events",
    "differentially private model",
    ".hash_identifier(",
    ".add_noise(",
    ".quantize_amounts(",
    "anonymize_amounts(",
    "add_differential_privacy_noise(",
    "anonymization",
    "hash_ids",
    "quantize_amounts",
    "add_noise",
    "gaussian mechanism",
)
EXCLUDED_PREFIXES = ("docs/plans/", "docs/superpowers/")


def _public_docs():
    yield ROOT / "README.md"
    for path in sorted((ROOT / "docs").rglob("*.md")):
        rel = path.relative_to(ROOT).as_posix()
        if not rel.startswith(EXCLUDED_PREFIXES):
            yield path


class TestDocs:
    def test_docs_do_not_reintroduce_removed_claims(self):
        hits = []
        for path in _public_docs():
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                for term in REMOVED:
                    if term.lower() in line.lower():
                        hits.append(f"{path.relative_to(ROOT)}:{lineno}: {term!r}")
        assert hits == []
