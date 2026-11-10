"""Public docs must not reintroduce claims or APIs this project removed."""

import ast
import dataclasses
import importlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from core.config import DataConfig, EvalConfig, SyntheticConfig, TrainConfig

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
        for doc in _public_docs():
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
            for doc in _public_docs()
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


FIX = ROOT / "tests" / "fixtures" / "docs_numbers"
NUM_RE = re.compile(r"([^\s>]+)\s*<!--num (\S+?)\.json (\S+) (\S+?)-->")
ZONE_RE = re.compile(r"<!--results-start-->(.*?)<!--results-end-->", re.DOTALL)
NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?%?")
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
CODE_RE = re.compile(r"`[^`\n]*`")


def _resolve(data, dotted: str):
    """Index dicts by key and lists by integer; KeyError/IndexError if absent."""
    for part in dotted.split("."):
        if isinstance(data, list):
            try:
                data = data[int(part)]
            except ValueError:
                raise KeyError(part) from None
        else:
            data = data[part]
    return data


def _blank(text: str, span: tuple[int, int]) -> str:
    """Replace a span with spaces, keeping newlines so offsets stay valid."""
    start, end = span
    return text[:start] + re.sub(r"[^\n]", " ", text[start:end]) + text[end:]


def check_numbers(md_path: Path, results_dir: Path) -> list[str]:
    """Violations for every quoted number: stale, dangling, or unmarked in a zone."""
    text = md_path.read_text()
    problems = []
    loaded: dict[str, object] = {}
    covered = []

    def line_of(pos: int) -> int:
        return text.count("\n", 0, pos) + 1

    for match in NUM_RE.finditer(text):
        covered.append(match.span(1))
        token, name, dotted, fmt = match.groups()
        where = f"{md_path.name}:{line_of(match.start())}"
        if name not in loaded:
            path = results_dir / f"{name}.json"
            loaded[name] = json.loads(path.read_text()) if path.exists() else None
        if loaded[name] is None:
            problems.append(f"{where}: no such results file {name}.json")
            continue
        try:
            value = _resolve(loaded[name], dotted)
        except (KeyError, IndexError):
            problems.append(f"{where}: no such path {dotted} in {name}.json")
            continue
        try:
            shown = "n/a" if value is None else format(value, fmt)
        except (TypeError, ValueError) as error:
            problems.append(
                f"{where}: {dotted} cannot be formatted with {fmt}: {error}"
            )
            continue
        quoted = token.strip(",])").lstrip("[(")
        if quoted != shown:
            problems.append(f"{where}: quoted {quoted} but {name}.json gives {shown}")

    if text.count("<!--results-start-->") != text.count("<!--results-end-->"):
        problems.append(f"{md_path.name}: results-start and results-end do not pair")
    scan = text
    for span in covered:
        scan = _blank(scan, span)
    for pattern in (COMMENT_RE, CODE_RE):
        for match in pattern.finditer(scan):
            scan = _blank(scan, match.span())
    for zone in ZONE_RE.finditer(text):
        for number in NUMERIC_RE.finditer(scan, *zone.span(1)):
            problems.append(
                f"{md_path.name}:{line_of(number.start())}: "
                f"unmarked number {number.group()} in a results zone"
            )
    return problems


class TestQuotedNumbers:
    def test_check_numbers_accepts_values_that_match_the_results_file(self):
        assert check_numbers(FIX / "good.md", FIX) == []

    def test_check_numbers_rejects_stale_unmarked_and_dangling_numbers(self):
        msgs = check_numbers(FIX / "bad.md", FIX)
        assert len(msgs) == 3
        assert any("0.85" in m and "0.84" in m for m in msgs)  # stale
        assert any("unmarked" in m for m in msgs)  # results zone
        assert any("no such path" in m for m in msgs)  # dangling marker

    @pytest.mark.parametrize(
        "line, reason",
        [
            ("0.85 <!--num results.json metrics.pr_auc.value .2f-->", "quoted 0.85"),
            ("0.84 <!--num results.json metrics.nope .2f-->", "no such path"),
            ("0.84 <!--num results.json metrics.pr_auc_by_horizon.9 .2f-->", "no such"),
            (
                "0.84 <!--num nowhere.json metrics.pr_auc.value .2f-->",
                "no such results",
            ),
            ("0.50 <!--num results.json metrics.lead_time .2f-->", "gives n/a"),
            ("0.84 <!--num results.json metrics.pr_auc .2f-->", "cannot be formatted"),
        ],
    )
    def test_each_violation_kind_is_reported_for_its_own_reason(
        self, tmp_path, line, reason
    ):
        doc = tmp_path / "doc.md"
        doc.write_text(f"<!--results-start-->{line}<!--results-end-->\n")
        msgs = check_numbers(doc, FIX)
        assert len(msgs) == 1 and reason in msgs[0], msgs

    def test_an_unmarked_number_in_a_results_zone_is_the_only_complaint(self, tmp_path):
        doc = tmp_path / "doc.md"
        doc.write_text("<!--results-start-->| row | 3 |<!--results-end-->\n")
        msgs = check_numbers(doc, FIX)
        assert len(msgs) == 1 and "unmarked number 3" in msgs[0], msgs

    def test_a_zone_that_never_closes_is_not_silently_ignored(self, tmp_path):
        doc = tmp_path / "doc.md"
        doc.write_text("<!--results-start-->no end marker\n")
        msgs = check_numbers(doc, FIX)
        assert len(msgs) == 1 and "do not pair" in msgs[0], msgs

    def test_a_marker_is_matched_without_swallowing_neighbouring_comments(self):
        # A greedy pattern read "<!--results-start-->0.10" as the token and
        # ".2f--><!--results-end" as the format.
        text = (
            "<!--results-start-->0.10 <!--num r.json m.v .2f--> "
            "0.50 <!--num r.json m.w .2f--><!--results-end-->"
        )
        assert [m.groups() for m in NUM_RE.finditer(text)] == [
            ("0.10", "r", "m.v", ".2f"),
            ("0.50", "r", "m.w", ".2f"),
        ]

    def test_readme_and_results_numbers_equal_the_committed_results(self):
        for name in ("README.md", "docs/RESULTS.md"):
            path = ROOT / name
            if path.exists():
                assert check_numbers(path, ROOT / "results" / "v1.0") == [], name

    def test_provisional_banner_is_gone_once_results_use_the_evaluation_seed(self):
        risk = json.loads((ROOT / "results/v1.0/risk.json").read_text())
        final = risk["seed"] == EvalConfig().eval_seed
        for name in ("README.md", "docs/RESULTS.md"):
            path = ROOT / name
            if path.exists():
                assert ("<!--provisional-->" in path.read_text()) != final, name


SYMBOL_RE = re.compile(r"^(core|data|models|train|export|eval)(\.\w+)+$")
FENCE_RE = re.compile(r"^[ \t]*```.*?^[ \t]*```", re.DOTALL | re.MULTILINE)
PATH_EXT = (".py", ".yaml", ".md", ".json", ".png", ".ipynb")
# Config sections a doc may name as `section.key`; a key is a dataclass field.
CONFIG_SECTIONS = {
    "data": DataConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}
NESTED_CONFIG = {(DataConfig, "synthetic"): SyntheticConfig}


def _spans(text: str) -> list[str]:
    """Single-backtick spans of prose; fenced blocks are skipped."""
    return [s.strip() for s in SPAN.findall(FENCE_RE.sub("", text))]


def doc_symbols(text: str) -> list[str]:
    return [span for span in _spans(text) if SYMBOL_RE.match(span)]


def _path_like(span: str) -> bool:
    if re.search(r"[\s*{}<>]", span) or re.match(r"\w+://", span):
        return False
    return "/" in span or span.split("::")[0].split("#")[0].endswith(PATH_EXT)


def doc_paths(text: str) -> list[str]:
    spans = [s for s in _spans(text) if _path_like(s)]
    return [s.split("::")[0].split("#")[0] for s in spans]


def doc_nodes(text: str) -> list[tuple[str, str]]:
    """(file, name) of every non-test `path::name` citation."""
    nodes = []
    for span in _spans(text):
        if _path_like(span) and "::" in span and not span.startswith("tests/"):
            path, name = span.split("::")[:2]
            nodes.append((path, name))
    return nodes


def _top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }


def _resolve_config_key(dotted: str):
    section, *keys = dotted.split(".")
    cls = CONFIG_SECTIONS[section]
    for key in keys:
        if cls is None or key not in {f.name for f in dataclasses.fields(cls)}:
            raise AttributeError(f"{cls.__name__} has no field {key}")
        cls = NESTED_CONFIG.get((cls, key))
    return cls


def resolve_symbol(dotted: str):
    """Longest importable module prefix, then getattr; else a config field."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        try:
            for part in parts[cut:]:
                obj = getattr(obj, part)
        except AttributeError:
            if parts[0] in CONFIG_SECTIONS:
                return _resolve_config_key(dotted)
            raise
        return obj
    raise ImportError(dotted)


class TestDocReferences:
    def test_every_symbol_the_docs_name_still_exists(self):
        missing = []
        for path in _public_docs():
            for sym in doc_symbols(path.read_text()):
                try:
                    resolve_symbol(sym)
                except (ImportError, AttributeError):
                    missing.append(f"{path.relative_to(ROOT)}: {sym}")
        assert missing == []

    def test_every_path_the_docs_name_exists(self):
        allowed = ("runs/", "checkpoints/")
        missing = [
            f"{p.relative_to(ROOT)}: {ref}"
            for p in _public_docs()
            for ref in doc_paths(p.read_text())
            if not ref.startswith(allowed) and not (ROOT / ref).exists()
        ]
        assert missing == []

    def test_every_non_test_node_the_docs_name_is_defined_there(self):
        missing = [
            f"{p.relative_to(ROOT)}: {ref}::{name}"
            for p in _public_docs()
            for ref, name in doc_nodes(p.read_text())
            if name not in _top_level_names(ROOT / ref)
        ]
        assert missing == []

    def test_symbol_extraction_ignores_prose_and_code_fences(self):
        text = "see `core.privacy.PrivacyManager` and `x.y`\n```\n`data.nope.Q`\n```"
        # a package-rooted token in prose is returned; x.y has no package root;
        # fenced text is skipped
        assert doc_symbols(text) == ["core.privacy.PrivacyManager"]

    def test_path_extraction_strips_suffixes_and_skips_placeholders(self):
        text = (
            "`tests/test_core.py::TestX::test_y` `README.md#usage` "
            "`config/synthetic_{easy,hard}.yaml` `runs/<task>/best.pt` "
            "`a b/c.py` `docs/*.md` `https://example.org/x.md` `core.privacy`\n"
            "```\n`fenced/path.py`\n```"
        )
        assert doc_paths(text) == ["tests/test_core.py", "README.md"]

    def test_node_extraction_returns_only_non_test_citations(self):
        text = "`train/loop.py::fit` `tests/test_core.py::TestX` `eval/run.py`"
        assert doc_nodes(text) == [("train/loop.py", "fit")]

    def test_a_config_key_resolves_through_the_config_dataclasses(self):
        for key in (
            "data.label_overlap",
            "data.synthetic",
            "data.synthetic.block_len",
            "eval.epsilons",
            "eval.n_train_seeds",
            "train.epochs",
            "core.privacy.PrivacyManager",
        ):
            resolve_symbol(key)

    @pytest.mark.parametrize(
        "key", ["data.nope", "data.synthetic.nope", "eval.nope", "core.nope.Q"]
    )
    def test_a_renamed_symbol_or_config_key_is_reported(self, key):
        with pytest.raises((ImportError, AttributeError)):
            resolve_symbol(key)

    def test_a_node_that_is_not_defined_in_its_file_is_reported(self):
        assert "fit" in _top_level_names(ROOT / "train/loop.py")
        assert "missing_name" not in _top_level_names(ROOT / "train/loop.py")


FIXQ = ROOT / "tests" / "fixtures" / "docs_quickstart"
QUICKSTART_FENCE_RE = re.compile(
    r"^[ \t]*```bash quickstart[ \t]*\n(.*?)^[ \t]*```", re.DOTALL | re.MULTILINE
)


def quickstart_commands(md_text: str) -> list[str]:
    """One command per logical line of every `bash quickstart` fence."""
    commands = []
    for block in QUICKSTART_FENCE_RE.findall(md_text):
        joined = re.sub(r"\\[ \t]*\n", " ", block)
        for line in joined.splitlines():
            line = " ".join(line.split())
            if line and not line.startswith("#"):
                commands.append(line)
    return commands


def sandboxed_config(tmp: Path) -> Path:
    """config/smoke.yaml with every output directory moved under tmp."""
    cfg = yaml.safe_load((ROOT / "config" / "smoke.yaml").read_text())
    cfg["train"]["checkpoint_dir"] = f"{tmp}/runs/smoke/train"
    cfg["eval"]["out_dir"] = f"{tmp}/results"
    cfg["eval"]["img_dir"] = f"{tmp}/img"
    out = tmp / "smoke.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


def _git_status_paths() -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.splitlines()


def _sandboxed_argv(cmd: str, cfg: Path, tmp: Path) -> list[str]:
    """Token-wise rewrite so a command only ever writes under tmp."""
    argv = []
    for token in shlex.split(cmd):
        if token == "config/smoke.yaml":
            token = str(cfg)
        elif token.startswith("runs/"):
            token = f"{tmp}/{token}"
        elif token == "dignity_risk.onnx":
            token = f"{tmp}/m.onnx"
        argv.append(token)
    if argv[0] == "python":
        argv[0] = sys.executable
    return argv


class TestQuickstart:
    def test_quickstart_extraction_reads_only_marked_fences(self):
        assert quickstart_commands((FIXQ / "sample.md").read_text()) == [
            "python -m train.cli --config config/smoke.yaml",
            "python -m export.to_onnx --checkpoint runs/smoke/train/best.pt "
            "--output dignity_risk.onnx",
        ]

    def test_sandboxed_config_moves_every_output_under_tmp(self, tmp_path):
        cfg = yaml.safe_load(sandboxed_config(tmp_path).read_text())
        for out in (
            cfg["train"]["checkpoint_dir"],
            cfg["eval"]["out_dir"],
            cfg["eval"]["img_dir"],
        ):
            assert out.startswith(str(tmp_path))

    @pytest.mark.slow
    @pytest.mark.timeout(900)  # three subprocesses, each capped at 300 s
    def test_readme_quickstart_runs_end_to_end_without_touching_committed_files(
        self, tmp_path
    ):
        before = _git_status_paths()
        cfg = sandboxed_config(tmp_path)
        cmds = quickstart_commands((ROOT / "README.md").read_text())
        assert cmds, "README has no `bash quickstart` fence"
        for cmd in cmds:
            proc = subprocess.run(
                _sandboxed_argv(cmd, cfg, tmp_path),
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert proc.returncode == 0, f"{cmd}\n{proc.stderr[-2000:]}"
        assert _git_status_paths() == before  # results/ and docs/img untouched
