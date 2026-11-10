"""Public docs must not reintroduce claims or APIs this project removed."""

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Removed claims and removed API names. Matching is case-insensitive. The design
# records that legitimately name the removed API live in the gitignored .internal/,
# so everything under docs/ is public and in scope.
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
    "generate_dataset(",
    "volume forecast",
    "price forecast",
    "A3C",
    "value estimate",
    "policy learning (RL)",
    "RL policy",
)


PRIVACY_DOCS = (ROOT / "docs" / "PRIVACY.md", ROOT / "docs" / "THREAT-MODEL.md")
CITED = re.compile(r"tests/(test_\w+\.py)::(\w+)(?:::(\w+))?")
SPAN = re.compile(r"`([^`]*)`")


def _public_docs():
    yield ROOT / "README.md"
    yield from sorted((ROOT / "docs").rglob("*.md"))


class TestDocs:
    def test_docs_do_not_reintroduce_removed_claims(self):
        hits = []
        for path in _public_docs():
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                for term in REMOVED:
                    if term.lower() in line.lower():
                        hits.append(f"{path.relative_to(ROOT)}:{lineno}: {term!r}")
        assert hits == []

    def test_a_configured_timeout_is_enforced_by_a_plugin_ci_installs(self):
        # M0.2 removed a timeout claim nothing enforced; a timeout line is only
        # honest if pytest-timeout is installed wherever the suite runs.
        ini = (ROOT / "pytest.ini").read_text()
        assert re.search(r"^timeout\s*=\s*\d+\s*$", ini, re.MULTILINE)
        assert "pytest-timeout" in (ROOT / "requirements-dev.txt").read_text()
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        assert "requirements-dev.txt" in ci

    def test_docs_do_not_link_to_files_that_do_not_exist(self):
        import re

        dead = []
        for path in _public_docs():
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", line):
                    if re.match(r"\w+:", target):
                        continue  # http(s):, mailto:
                    if not (path.parent / target).exists():
                        dead.append(f"{path.relative_to(ROOT)}:{lineno}: {target}")
        assert dead == []

    def test_stale_docs_stay_deleted(self):
        for name in (
            "QUICK_START",
            "CONFIGURATION",
            "SIGNALS",
            "DOCUMENTATION_SUMMARY",
        ):
            assert not (ROOT / "docs" / f"{name}.md").exists(), name

    def test_every_cited_privacy_test_exists(self):
        # A citation that names a test which does not exist is a claim with no
        # test behind it (spec section 3).
        missing = []
        for doc in PRIVACY_DOCS:
            for fname, cls, method in CITED.findall(doc.read_text()):
                tree = ast.parse((ROOT / "tests" / fname).read_text())
                classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
                node = f"tests/{fname}::{cls}" + (f"::{method}" if method else "")
                methods = (
                    {
                        n.name
                        for n in classes[cls].body
                        if isinstance(n, ast.FunctionDef)
                    }
                    if cls in classes
                    else set()
                )
                if cls not in classes or (method and method not in methods):
                    missing.append(f"{doc.name}: {node}")
        assert missing == []
        # No citation may omit its file (DGNTY-37): a backticked span naming a
        # Test* class must start with "tests/".
        bare = [
            span
            for doc in PRIVACY_DOCS
            for span in SPAN.findall(doc.read_text())
            if re.search(r"\bTest[A-Z]\w*", span) and not span.startswith("tests/")
        ]
        assert bare == []

    def test_privacy_docs_state_the_per_split_claim(self):
        for path in PRIVACY_DOCS:
            text = path.read_text()
            assert "parallel composition" in text, path.name
            assert "k-anonymity holds for the train release only" in text, path.name

    def test_privacy_curve_statement_matches_the_committed_results(self):
        # PRIVACY.md says the private model is not distinguishable from chance at
        # any epsilon on the grid. If regenerated results stop supporting that
        # sentence, the sentence has to change with them.
        text = " ".join((ROOT / "docs/PRIVACY.md").read_text().split())
        assert "PR-AUC not distinguishable from chance on the hard preset" in text
        assert "Raising ε a hundredfold" in text
        metrics = json.loads((ROOT / "results/v1.0/privacy.json").read_text())[
            "metrics"
        ]
        reference = metrics["reference"]["risk"]
        chance = reference["prevalence"]["value"]
        epsilons = [point["epsilon"] for point in metrics["points"]]
        assert len(epsilons) >= 5 and max(epsilons) >= 100 * min(epsilons)
        for point in metrics["points"]:
            pr_auc = point["risk"]["pr_auc"]
            assert pr_auc["ci_low"] <= chance <= pr_auc["ci_high"], point["epsilon"]
            assert pr_auc["ci_high"] < reference["pr_auc"]["ci_low"], point["epsilon"]
