"""Public docs must not reintroduce claims or APIs this project removed."""

import ast
import json
import re
from pathlib import Path

from core.config import EvalConfig

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
        # PRIVACY.md says which grid points are at chance and which are above it.
        # If regenerated results stop supporting those sentences, they have to
        # change with them.
        text = " ".join((ROOT / "docs/PRIVACY.md").read_text().split())
        for sentence in (
            "On the tuning seed, the shipped allocation leaves the risk model's "
            "PR-AUC not distinguishable from chance on the hard preset at every ε "
            "on the grid up to 10, and above chance at the two largest grid values, "
            "30 and 100",
            "contains the prevalence",
            "wholly below the non-private reference's interval",
            "the interval lies wholly above the prevalence",
            "PR-AUC is higher at 100 than at 30",
            "both point estimates are below the reference's",
            "every PR-AUC at ε = 100 is above every PR-AUC at a smaller ε",
            "At ε = 30 the replicates spread more widely than the gap",
            "the same allocation at ε = 1.0 leaves the model's PR-AUC not "
            "distinguishable from chance on the tuning seed",
            "up to ε = 10 its intervals do not all contain 0.5",
            "Only the noise on `volume` changes along the curve",
            "The rise at the two largest ε therefore comes from less noise on `volume`",
        ):
            assert sentence in text, sentence
        payload = json.loads((ROOT / "results/v1.0/privacy.json").read_text())
        assert payload["seed"] == EvalConfig().tuning_seed
        metrics = payload["metrics"]
        reference = metrics["reference"]["risk"]
        chance = reference["prevalence"]["value"]
        points = {point["epsilon"]: point for point in metrics["points"]}
        at_chance = [epsilon for epsilon in points if epsilon <= 10]
        grid = "the sentences name the grid: every value up to 10, then 30 and 100"
        assert max(at_chance) == 10 and 1.0 in at_chance, grid
        assert sorted(points)[len(at_chance) :] == [30, 100], grid

        def replicates(epsilon):
            return points[epsilon]["replicate_spread"]["pr_auc"]["values"]

        for epsilon in at_chance:
            pr_auc = points[epsilon]["risk"]["pr_auc"]
            assert pr_auc["ci_low"] <= chance <= pr_auc["ci_high"], epsilon
            assert pr_auc["ci_high"] < reference["pr_auc"]["ci_low"], epsilon
        for epsilon in (30, 100):
            pr_auc = points[epsilon]["risk"]["pr_auc"]
            assert pr_auc["ci_low"] > chance, epsilon
            assert pr_auc["value"] < reference["pr_auc"]["value"], epsilon
        values = {e: points[e]["risk"]["pr_auc"]["value"] for e in points}
        assert values[100] > values[30] > max(values[e] for e in at_chance)
        below_30 = max(max(replicates(e)) for e in at_chance)
        assert min(replicates(100)) > max(below_30, max(replicates(30)))
        assert (
            max(replicates(30)) - min(replicates(30)) > min(replicates(30)) - below_30
        )
        aurocs = [points[e]["risk"]["auroc"] for e in at_chance]
        assert not all(a["ci_low"] <= 0.5 <= a["ci_high"] for a in aurocs)
