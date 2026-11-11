"""Public docs must not reintroduce claims or APIs this project removed."""

import ast
import collections
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from core.config import (
    DataConfig,
    DignityConfig,
    EvalConfig,
    ModelConfig,
    PolicyConfig,
    PrivacyConfig,
    SyntheticConfig,
    TrainConfig,
)

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


def heading_slugs(path: Path) -> set[str]:
    """The anchors GitHub gives the headings of a markdown file."""
    heads = re.findall(r"^#{1,6} +(.+?) *$", FENCE_RE.sub("", path.read_text()), re.M)
    return {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in heads}


def dead_links(docs=None) -> list[str]:
    """Every relative link of a doc (default: every public doc) whose file is
    missing or, for a markdown file, whose #fragment names no heading."""
    dead = []
    for path in _public_docs() if docs is None else docs:
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            for target, frag in re.findall(r"\]\(([^)#\s]+)(?:#([^)]*))?\)", line):
                if re.match(r"\w+:", target):
                    continue  # http(s):, mailto:
                where = f"{_label(path)}:{lineno}: {target}" + (
                    f"#{frag}" if frag else ""
                )
                linked = path.parent / target
                missing_anchor = (
                    linked.exists()
                    and frag
                    and target.endswith(".md")
                    and frag not in heading_slugs(linked)
                )
                if not linked.exists() or missing_anchor:
                    dead.append(where)
    return dead


def missing_citations(docs=None) -> list[str]:
    """Every `tests/file::Class[::method]` citation of a doc (default: every public
    doc) that names a class or method the test file does not define."""
    missing = []
    for doc in _public_docs() if docs is None else docs:
        for fname, cls, method in CITED.findall(doc.read_text()):
            tree = ast.parse((ROOT / "tests" / fname).read_text())
            classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
            node = f"tests/{fname}::{cls}" + (f"::{method}" if method else "")
            methods = (
                {n.name for n in classes[cls].body if isinstance(n, ast.FunctionDef)}
                if cls in classes
                else set()
            )
            if cls not in classes or (method and method not in methods):
                missing.append(f"{doc.name}: {node}")
    return missing


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
        assert dead_links() == []

    def test_a_link_to_a_heading_that_does_not_exist_is_reported(self, tmp_path):
        (tmp_path / "target.md").write_text("# Title\n\n## Data contract\n")
        doc = tmp_path / "doc.md"
        doc.write_text("[ok](target.md#data-contract) [bad](target.md#no-such)\n")
        assert dead_links([doc]) == [f"{_label(doc)}:1: target.md#no-such"]

    def test_stale_docs_stay_deleted(self):
        for name in (
            "QUICK_START",
            "CONFIGURATION",
            "SIGNALS",
            "DOCUMENTATION_SUMMARY",
        ):
            assert not (ROOT / "docs" / f"{name}.md").exists(), name

    def test_every_cited_test_exists(self):
        # A citation that names a test which does not exist is a claim with no
        # test behind it (spec section 3). Every public doc is checked.
        assert missing_citations() == []
        # No citation may omit its file (DGNTY-37): a backticked span naming a
        # Test* class must start with "tests/".
        bare = [
            span
            for doc in _public_docs()
            for span in SPAN.findall(doc.read_text())
            if re.search(r"\bTest[A-Z]\w*", span) and not span.startswith("tests/")
        ]
        assert bare == []

    def test_a_citation_of_a_test_that_does_not_exist_is_reported(self, tmp_path):
        arch = ROOT / "docs" / "ARCHITECTURE.md"
        real = "test_forecast_probabilities_never_decrease_with_horizon"
        assert real in arch.read_text()
        doc = tmp_path / "ARCHITECTURE.md"
        doc.write_text(arch.read_text())
        assert missing_citations([doc]) == []
        doc.write_text(arch.read_text().replace(real, "test_no_such_method"))
        assert any("test_no_such_method" in m for m in missing_citations([doc]))
        doc.write_text(arch.read_text().replace("::TestTaskHeads", "::TestNoClass"))
        assert any("TestNoClass" in m for m in missing_citations([doc]))

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
        assert payload["seed"] == EvalConfig().tuning_seed, (
            "this statement is about the tuning seed: rewrite it in PRIVACY.md and "
            "in docs/RESULTS.md for the evaluation results, then drop this check"
        )
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


def _load_script(name: str):
    """Import scripts/<name>.py by path, as tests/test_colab_eval.py does."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The marker grammar has one definition, in the refresh script; the checker
# below and the refresh both read it from there.
_refresh = _load_script("refresh_doc_numbers")
FENCE_RE = _refresh.FENCE_RE
CODE_RE = _refresh.CODE_RE
NUM_RE = _refresh.NUM_RE
TOKEN_RE = _refresh.TOKEN_RE
split_token = _refresh.split_token
_resolve = _refresh.resolve
_blank = _refresh._blank
strip_code = _refresh.strip_code
FIX = ROOT / "tests" / "fixtures" / "docs_numbers"
FIXR = ROOT / "tests" / "fixtures" / "docs_refs"
RESULTS_DIR = ROOT / "results" / "v1.0"
NUM_OPEN_RE = re.compile(r"<!--\s*num\b")
ZONE_TAG_RE = re.compile(r"<!--results-(start|end)-->")
# Anything that looks like a zone tag or the provisional tag; only the three exact
# spellings are tags, the rest would be skipped silently.
TAG_LIKE_RE = re.compile(
    r"<!--\s*(?:results-(?:start|end)|provisional)\b", re.IGNORECASE
)
EXACT_TAGS = ("<!--results-start-->", "<!--results-end-->", "<!--provisional-->")
NUMERIC_RE = re.compile(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?%?")
# A backticked span that is only a number: signed, with separators, a decimal
# point, an exponent or a percent sign. Labels and paths that merely contain
# digits are not matched.
NUMERIC_SPAN_RE = re.compile(r"[+-]?[\d.,_]*\d[\d.,_]*(?:[eE][+-]?\d+)?%?")
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
BANNER = (
    "<!--provisional-->**Provisional: numbers are from the tuning seed; "
    "the final evaluation has not run.**"
)


def _zone_scan_text(text: str) -> str:
    """Text for the unmarked-number scan: fences and code spans blanked, except a
    backticked span that is only a number, which stays visible."""
    for match in list(FENCE_RE.finditer(text)):
        text = _blank(text, match.span())
    for match in list(CODE_RE.finditer(text)):
        start, end = match.span()
        if NUMERIC_SPAN_RE.fullmatch(text[start + 1 : end - 1].strip()):
            text = _blank(text, (start, start + 1))
            text = _blank(text, (end - 1, end))
        else:
            text = _blank(text, (start, end))
    return text


def check_numbers(md_path: Path, results_dir: Path) -> list[str]:
    """Violations for every quoted number: stale, dangling, or unmarked in a zone."""
    text = md_path.read_text()
    live = strip_code(text)
    problems = []
    loaded: dict[str, object] = {}
    covered = []
    marker_starts = set()

    def line_of(pos: int) -> int:
        return text.count("\n", 0, pos) + 1

    for match in NUM_RE.finditer(live):
        covered.append(match.span(1))
        marker_starts.add(
            match.start() + match.group(0).index("<!--num", len(match.group(1)))
        )
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
            shown = _refresh.render(value, fmt)
        except (TypeError, ValueError) as error:
            problems.append(
                f"{where}: {dotted} cannot be formatted with {fmt}: {error}"
            )
            continue
        try:
            _, quoted, _ = split_token(token, fmt)
        except ValueError as error:
            problems.append(f"{where}: {error}")
            continue
        if quoted != shown:
            problems.append(f"{where}: quoted {quoted} but {name}.json gives {shown}")

    for opener in NUM_OPEN_RE.finditer(live):
        if opener.start() not in marker_starts:
            problems.append(
                f"{md_path.name}:{line_of(opener.start())}: malformed marker, "
                "expected a number, a space and <!--num <file>.json <path> <fmt>--> "
                "outside code"
            )

    for like in TAG_LIKE_RE.finditer(live):
        if not live.startswith(EXACT_TAGS, like.start()):
            problems.append(
                f"{md_path.name}:{line_of(like.start())}: malformed zone tag, "
                "write exactly <!--results-start-->, <!--results-end--> or "
                "<!--provisional-->"
            )

    zones = []
    opened = None
    for tag in ZONE_TAG_RE.finditer(live):
        where = f"{md_path.name}:{line_of(tag.start())}"
        if tag[1] == "start" and opened is None:
            opened = tag
        elif tag[1] == "end" and opened is not None:
            zones.append((opened.end(), tag.start()))
            opened = None
        else:
            why = (
                "results-start inside an open zone"
                if opened
                else ("results-end before any results-start")
            )
            problems.append(
                f"{where}: results-start and results-end do not pair: {why}"
            )
            break
    else:
        if opened is not None:
            problems.append(
                f"{md_path.name}:{line_of(opened.start())}: results-start and "
                "results-end do not pair: results-start is never closed"
            )

    scan = _zone_scan_text(text)
    for span in covered:
        scan = _blank(scan, span)
    for match in list(COMMENT_RE.finditer(scan)):
        scan = _blank(scan, match.span())
    for start, end in zones:
        for number in NUMERIC_RE.finditer(scan, start, end):
            problems.append(
                f"{md_path.name}:{line_of(number.start())}: "
                f"unmarked number {number.group()} in a results zone"
            )
    return problems


def numbers_problems(docs=None, results_dir: Path = RESULTS_DIR) -> list[str]:
    """check_numbers over every doc given (default: every public doc)."""
    return [
        problem
        for doc in (_public_docs() if docs is None else docs)
        for problem in check_numbers(doc, results_dir)
    ]


def banner_problems(name: str, text: str, final: bool) -> list[str]:
    """The provisional banner is the full convention-6 line until the results are
    on the evaluation seed, and the word Provisional appears nowhere after."""
    if final:
        if "Provisional" in text or "<!--provisional-->" in text:
            return [
                f"{name}: the results carry the evaluation seed, so delete the "
                "banner line (the word Provisional and the <!--provisional--> "
                "comment) from this doc"
            ]
        return []
    if BANNER not in text:
        return [
            f"{name}: the results are still on the tuning seed, so the doc must "
            f"carry the banner line {BANNER}"
        ]
    return []


def run_identity_problems(results_dir: Path = RESULTS_DIR) -> list[str]:
    """The four results files must come from one run: same seed, git_sha and
    config_hash, with the seed one of the two frozen ones."""
    payloads = {
        name: json.loads((results_dir / f"{name}.json").read_text())
        for name in ("risk", "forecast", "policy", "privacy")
    }
    problems = []
    for key in ("seed", "git_sha", "config_hash"):
        values = {name: payload[key] for name, payload in payloads.items()}
        if len(set(values.values())) != 1:
            problems.append(f"{key} differs between results files: {values}")
    seed = payloads["risk"]["seed"]
    if seed not in (EvalConfig().tuning_seed, EvalConfig().eval_seed):
        problems.append(f"seed {seed} is neither the tuning nor the evaluation seed")
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
            (
                "0.84 <!--num results.json metrics.pr_auc_by_horizon.9 .2f-->",
                "no such path",
            ),
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


MARK_PR = "<!--num results.json metrics.pr_auc.value .2f-->"
MARK_LOW = "<!--num results.json metrics.pr_auc.ci_low .2f-->"
MARK_HIGH = "<!--num results.json metrics.pr_auc.ci_high .2f-->"
MARK_MISSED = "<!--num results.json counts.n_missed d-->"
MARK_ONSETS = "<!--num results.json counts.n_onsets d-->"


def _doc(tmp_path, text, name="doc.md"):
    doc = tmp_path / name
    doc.write_text(text)
    return doc


def _zone(body):
    return f"<!--results-start-->{body}<!--results-end-->\n"


class TestMarkerToken:
    @pytest.mark.parametrize(
        "token, expected",
        [
            ("0.81", ("", "0.81", "")),
            ("[0.72", ("[", "0.72", "")),
            ("0.88]", ("", "0.88", "]")),
            ("(0.72,", ("(", "0.72", ",")),
            ("0.88)", ("", "0.88", ")")),
            ("**0.81**", ("**", "0.81", "**")),
            ("0.81.", ("", "0.81", ".")),
            ("0.81,", ("", "0.81", ",")),
            ("0.81;", ("", "0.81", ";")),
            ("0.81:", ("", "0.81", ":")),
            ("|0.81|", ("|", "0.81", "|")),
            ("/34", ("/", "34", "")),
            ("n/a", ("", "n/a", "")),
            ("(n/a,", ("(", "n/a", ",")),
            ("-0.5", ("", "-0.5", "")),
            ("84.1%", ("", "84.1%", "")),
            ("1,000,", ("", "1,000", ",")),
            ("1e-3", ("", "1e-3", "")),
            ("100", ("", "100", "")),
        ],
    )
    def test_a_markdown_or_punctuation_wrapper_is_not_part_of_the_number(
        self, token, expected
    ):
        assert split_token(token) == expected

    def test_an_s_format_token_is_the_whole_token(self):
        assert split_token("**ε=100**", "s") == ("", "**ε=100**", "")

    def test_a_range_is_rejected_with_a_message_naming_the_bracket_form(self):
        with pytest.raises(ValueError, match="bracket form"):
            split_token("0.72–0.88")

    def test_a_token_without_a_number_is_rejected(self):
        with pytest.raises(ValueError, match="cannot read a number"):
            split_token("ε=100")

    @pytest.mark.parametrize(
        "token", ["[0.72", "0.88]", "(0.72,", "**0.81**", "|0.81|", "/34", "0.81."]
    )
    def test_a_refresh_that_rewrites_only_the_core_keeps_every_wrapper(self, token):
        prefix, core, suffix = split_token(token)
        assert prefix + "9.99" + suffix == token.replace(core, "9.99")
        assert split_token(prefix + "9.99" + suffix) == (prefix, "9.99", suffix)

    @pytest.mark.parametrize(
        "line",
        [
            f"[0.84 {MARK_PR}, 0.87 {MARK_HIGH}]",
            f"(0.80 {MARK_LOW}, 0.87 {MARK_HIGH})",
            f"**0.84** {MARK_PR}",
            f"**0.84 {MARK_PR}**",
            f"The score was 0.84. {MARK_PR}",
            f"0.84, {MARK_PR}",
            f"|0.84{MARK_PR}|",
            f"| 0.84 {MARK_PR} |",
            f"0 {MARK_MISSED}/34 {MARK_ONSETS}",
        ],
    )
    def test_a_wrapped_number_is_compared_by_its_core(self, tmp_path, line):
        doc = _doc(tmp_path, line + "\n")
        assert check_numbers(doc, FIX) == []

    def test_a_wrapped_stale_number_is_still_reported_by_its_core(self, tmp_path):
        doc = _doc(tmp_path, f"**0.85** {MARK_PR}\n")
        msgs = check_numbers(doc, FIX)
        assert len(msgs) == 1 and "quoted 0.85 but results.json gives 0.84" in msgs[0]

    def test_an_en_dash_range_tells_the_writer_to_use_the_bracket_form(self, tmp_path):
        doc = _doc(tmp_path, f"0.72–0.88 {MARK_PR}\n")
        msgs = check_numbers(doc, FIX)
        assert len(msgs) == 1 and "bracket form" in msgs[0], msgs


class TestCodeIsInert:
    def test_a_marker_in_an_inline_code_span_is_documentation(self, tmp_path):
        text = "Write `0.84 <!--num nowhere.json x.y .2f-->` after a number.\n"
        assert check_numbers(_doc(tmp_path, text), FIX) == []

    def test_a_marker_in_a_fenced_block_is_documentation(self, tmp_path):
        text = "```\n0.99 <!--num nowhere.json x.y .2f-->\n```\n"
        assert check_numbers(_doc(tmp_path, text), FIX) == []

    def test_a_zone_tag_in_code_neither_opens_nor_closes_a_zone(self, tmp_path):
        text = (
            "Zones open with `<!--results-start-->`.\n"
            "```\n<!--results-start-->\n```\n"
            "Prose with 12 in it.\n"
        )
        assert check_numbers(_doc(tmp_path, text), FIX) == []

    def test_a_marker_in_code_inside_a_zone_is_not_counted_as_the_number(
        self, tmp_path
    ):
        body = "| `0.84 <!--num results.json x.y .2f-->` | 5 |"
        msgs = check_numbers(_doc(tmp_path, _zone(body)), FIX)
        assert len(msgs) == 1 and "unmarked number 5" in msgs[0], msgs

    def test_a_stray_backtick_does_not_hide_a_number_on_the_next_line(self, tmp_path):
        text = _zone("a ` b\n7 and ` c")
        msgs = check_numbers(_doc(tmp_path, text), FIX)
        assert len(msgs) == 1 and "unmarked number 7" in msgs[0], msgs

    @pytest.mark.parametrize("fence", ["```", "~~~"])
    def test_prose_between_two_fenced_blocks_is_still_checked(self, tmp_path, fence):
        # A greedy fence pattern read everything from the first fence to the last
        # as one block, so a stale number and a zone number between them went
        # unchecked.
        block = f"{fence}\ncode\n{fence}\n"
        text = block + f"0.85 {MARK_PR}\n" + _zone("| 3 |") + block
        msgs = check_numbers(_doc(tmp_path, text), FIX)
        assert len(msgs) == 2, msgs
        assert "quoted 0.85 but results.json gives 0.84" in msgs[0], msgs
        assert "unmarked number 3" in msgs[1], msgs

    def test_a_marker_and_a_zone_tag_in_a_tilde_fence_are_documentation(self, tmp_path):
        text = (
            "~~~\n0.99 <!--num nowhere.json x.y .2f-->\n<!--results-start-->\n~~~\n"
            "Prose with 12 in it.\n"
        )
        assert check_numbers(_doc(tmp_path, text), FIX) == []

    @pytest.mark.parametrize("outer, inner", [("~~~", "```"), ("```", "~~~")])
    def test_a_fence_is_closed_only_by_its_own_kind_of_fence(
        self, tmp_path, outer, inner
    ):
        text = f"{outer}\n{inner}\n{outer}\n0.85 {MARK_PR}\n"
        msgs = check_numbers(_doc(tmp_path, text), FIX)
        assert len(msgs) == 1 and "quoted 0.85" in msgs[0], msgs

    @pytest.mark.parametrize(
        "tag",
        [
            "<!-- results-start -->",
            "<!--results-start -->",
            "<!--results-end  -->",
            "<!--Results-End-->",
            "<!-- provisional -->",
            "<!--provisional -->",
            "<!--PROVISIONAL-->",
        ],
    )
    def test_a_tag_that_looks_like_a_zone_or_provisional_tag_is_reported(
        self, tmp_path, tag
    ):
        msgs = check_numbers(_doc(tmp_path, f"intro\n{tag}\n"), FIX)
        assert len(msgs) == 1 and "malformed zone tag" in msgs[0], msgs
        assert msgs[0].startswith("doc.md:2:"), msgs

    def test_a_spaced_zone_tag_does_not_leave_its_numbers_unchecked(self, tmp_path):
        text = "<!-- results-start -->| 3 |<!--results-end-->\n"
        msgs = check_numbers(_doc(tmp_path, text), FIX)
        assert any("malformed zone tag" in m for m in msgs), msgs

    def test_a_spaced_tag_in_code_is_documentation(self, tmp_path):
        text = "Write `<!-- results-start -->` or\n```\n<!-- provisional -->\n```\n"
        assert check_numbers(_doc(tmp_path, text), FIX) == []

    def test_the_exact_tags_are_not_reported_as_malformed(self, tmp_path):
        text = f"<!--provisional-->{_zone(f'0.84 {MARK_PR}')}"
        assert check_numbers(_doc(tmp_path, text), FIX) == []


class TestBacktickedNumbers:
    @pytest.mark.parametrize(
        "span", ["0.99", "-0.5", "+3", "1,000", "1e-3", "12%", ".5", "2026"]
    )
    def test_a_backticked_number_in_a_zone_is_an_unmarked_number(self, tmp_path, span):
        msgs = check_numbers(_doc(tmp_path, _zone(f"| PR-AUC | `{span}` |")), FIX)
        assert len(msgs) == 1 and "unmarked number" in msgs[0], msgs

    @pytest.mark.parametrize("span", [" 0.99 ", "  12%", "-0.5\t", " 2026 "])
    def test_a_padded_backticked_number_in_a_zone_is_an_unmarked_number(
        self, tmp_path, span
    ):
        msgs = check_numbers(_doc(tmp_path, _zone(f"| PR-AUC | `{span}` |")), FIX)
        assert len(msgs) == 1 and "unmarked number" in msgs[0], msgs

    def test_a_backticked_label_or_path_with_digits_stays_exempt(self, tmp_path):
        body = (
            "| `recall_at_1pct_fpr` | `pr_auc_by_horizon.4` | `policy.review_budget` "
            f"| `results/v1.0/risk.json` | `top10` | 0.84 {MARK_PR} |"
        )
        assert check_numbers(_doc(tmp_path, _zone(body)), FIX) == []

    def test_a_backticked_number_outside_a_zone_is_not_checked(self, tmp_path):
        assert check_numbers(_doc(tmp_path, "The seed is `2026`.\n"), FIX) == []


class TestZones:
    def test_prose_between_two_zones_is_not_a_results_zone(self, tmp_path):
        text = (
            _zone(f"0.84 {MARK_PR}")
            + "Between the zones 99 and 12% are prose.\n"
            + _zone(f"0.84 {MARK_PR}")
        )
        assert check_numbers(_doc(tmp_path, text), FIX) == []

    @pytest.mark.parametrize(
        "text, why",
        [
            ("<!--results-end-->x<!--results-start-->", "before any results-start"),
            (
                "<!--results-start-->a<!--results-start-->b"
                "<!--results-end--><!--results-end-->",
                "inside an open zone",
            ),
            ("<!--results-end-->| 3 |<!--results-start-->", "before any"),
            ("<!--results-start-->never closed", "never closed"),
        ],
    )
    def test_zone_tags_must_alternate_starting_with_a_start_tag(
        self, tmp_path, text, why
    ):
        msgs = check_numbers(_doc(tmp_path, text + "\n"), FIX)
        assert len(msgs) == 1 and "do not pair" in msgs[0] and why in msgs[0], msgs

    @pytest.mark.parametrize(
        "cell, quoted",
        [("345", "345"), ("12%", "12%"), ("12 %", "12"), ("3.25", "3.25")],
    )
    def test_a_whole_number_in_a_zone_is_reported_not_a_digit_of_it(
        self, tmp_path, cell, quoted
    ):
        msgs = check_numbers(_doc(tmp_path, _zone(f"| {cell} |")), FIX)
        assert len(msgs) == 1, msgs
        assert msgs[0].endswith(f"unmarked number {quoted} in a results zone")

    @pytest.mark.parametrize(
        "body, quoted",
        [("3 glued to the start tag", "3"), ("glued to the end tag 45", "45")],
    )
    def test_a_number_glued_to_a_zone_tag_is_inside_the_zone(
        self, tmp_path, body, quoted
    ):
        msgs = check_numbers(_doc(tmp_path, _zone(body)), FIX)
        assert len(msgs) == 1, msgs
        assert msgs[0].endswith(f"unmarked number {quoted} in a results zone"), msgs

    def test_a_one_character_zone_is_scanned_from_both_edges(self, tmp_path):
        msgs = check_numbers(_doc(tmp_path, _zone("7")), FIX)
        assert len(msgs) == 1 and "unmarked number 7" in msgs[0], msgs

    def test_a_multi_line_comment_in_a_zone_hides_its_digits(self, tmp_path):
        text = _zone(f"<!-- a note\nspanning 7 lines -->0.84 {MARK_PR}")
        assert check_numbers(_doc(tmp_path, text), FIX) == []

    def test_messages_carry_the_line_of_the_offending_text(self, tmp_path):
        text = f"intro\n<!-- a comment\nover two lines -->\n\n0.85 {MARK_PR}\n" + _zone(
            "| 3 |"
        )
        msgs = check_numbers(_doc(tmp_path, text), FIX)
        assert [m.split(" ")[0] for m in msgs] == ["doc.md:5:", "doc.md:6:"], msgs


class TestMarkerValues:
    def test_a_zero_is_quoted_as_zero_not_as_not_available(self, tmp_path):
        assert check_numbers(_doc(tmp_path, f"0 {MARK_MISSED}\n"), FIX) == []
        msgs = check_numbers(_doc(tmp_path, f"1 {MARK_MISSED}\n"), FIX)
        assert len(msgs) == 1 and "gives 0" in msgs[0], msgs

    def test_a_quoted_prefix_of_the_value_is_stale(self, tmp_path):
        for quoted in ("0.8", "10.84", "0.841"):
            msgs = check_numbers(_doc(tmp_path, f"{quoted} {MARK_PR}\n"), FIX)
            assert len(msgs) == 1 and "gives 0.84" in msgs[0], (quoted, msgs)

    def test_a_format_that_does_not_fit_the_value_is_reported(self, tmp_path):
        line = "0.84 <!--num results.json metrics.pr_auc.value s-->\n"
        msgs = check_numbers(_doc(tmp_path, line), FIX)
        assert len(msgs) == 1 and "cannot be formatted" in msgs[0], msgs

    @pytest.mark.parametrize(
        "dotted",
        [
            "metrics.pr_auc_by_horizon.x",
            "metrics.pr_auc_by_horizon.-1",
            "metrics.pr_auc_by_horizon.9",
            "metrics.lead_time.median",
            "metrics.pr_auc.value.x",
            "metrics.pr_auc.value.0",
        ],
    )
    def test_a_path_that_cannot_be_followed_is_no_such_path_not_a_crash(
        self, tmp_path, dotted
    ):
        doc = _doc(tmp_path, f"0.5 <!--num results.json {dotted} .2f-->\n")
        msgs = check_numbers(doc, FIX)
        assert len(msgs) == 1 and "no such path" in msgs[0], msgs

    @pytest.mark.parametrize(
        "line",
        [
            "0.84 <!--num results.json metrics.pr_auc.value-->",
            "0.84 <!-- num results.json metrics.pr_auc.value .2f-->",
            "0.84 <!--num results.json metrics.pr_auc.value .2f -->",
            "<!--num results.json metrics.pr_auc.value .2f-->",
        ],
    )
    def test_a_malformed_marker_is_reported_instead_of_ignored(self, tmp_path, line):
        msgs = check_numbers(_doc(tmp_path, line + "\n"), FIX)
        assert len(msgs) == 1 and "malformed marker" in msgs[0], msgs

    def test_a_malformed_marker_in_a_zone_is_reported_as_malformed(self, tmp_path):
        line = "0.84 <!--num results.json metrics.pr_auc.value-->"
        msgs = check_numbers(_doc(tmp_path, _zone(line)), FIX)
        assert any("malformed marker" in m for m in msgs), msgs


class TestSeveralResultsFiles:
    @staticmethod
    def _results(tmp_path):
        results = tmp_path / "res"
        results.mkdir()
        (results / "risk.json").write_text('{"metrics": {"pr_auc": 0.84}}')
        (results / "policy.json").write_text('{"metrics": {"neg_cost": -12.5}}')
        return results

    GOOD = (
        "0.84 <!--num risk.json metrics.pr_auc .2f--> and "
        "-12.5 <!--num policy.json metrics.neg_cost .1f-->\n"
        "0.84 <!--num risk.json metrics.pr_auc .2f-->\n"
    )

    def test_a_doc_quoting_numbers_from_two_results_files_is_checked_against_each(
        self, tmp_path
    ):
        results = self._results(tmp_path)
        assert check_numbers(_doc(tmp_path, self.GOOD), results) == []

    def test_a_stale_number_from_the_second_file_is_reported_with_its_file(
        self, tmp_path
    ):
        results = self._results(tmp_path)
        stale = self.GOOD.replace("-12.5", "-12.6")
        msgs = check_numbers(_doc(tmp_path, stale), results)
        assert len(msgs) == 1, msgs
        assert "quoted -12.6 but policy.json gives -12.5" in msgs[0], msgs


class TestEveryPublicDocIsChecked:
    def test_every_public_doc_quotes_numbers_that_equal_the_committed_results(self):
        assert (ROOT / "README.md").exists() and (ROOT / "docs/RESULTS.md").exists()
        assert numbers_problems() == []

    def test_a_stale_number_in_any_doc_is_reported(self, tmp_path):
        stale = _doc(tmp_path, f"0.99 {MARK_PR}\n", "ARCHITECTURE.md")
        msgs = numbers_problems([FIX / "good.md", stale], FIX)
        assert len(msgs) == 1 and "ARCHITECTURE.md:1" in msgs[0], msgs

    def test_the_repo_wide_check_scans_the_public_docs_by_default(
        self, tmp_path, monkeypatch
    ):
        stale = _doc(tmp_path, f"0.99 {MARK_PR}\n", "PRIVACY.md")
        monkeypatch.setattr(sys.modules[__name__], "_public_docs", lambda: [stale])
        assert len(numbers_problems(results_dir=FIX)) == 1

    def test_the_public_docs_include_every_markdown_file_under_docs(self):
        names = {path.relative_to(ROOT).as_posix() for path in _public_docs()}
        assert "README.md" in names
        assert {
            p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").rglob("*.md")
        } <= (names)


class TestProvisionalBanner:
    def test_the_banner_is_the_full_visible_line_until_the_evaluation_seed(self):
        assert banner_problems("d.md", f"intro\n{BANNER}\n", final=False) == []
        for broken in (
            "intro\n",
            "<!--provisional-->\n",
            "**Provisional: numbers are from the tuning seed; the final "
            "evaluation has not run.**\n",
            "<!--provisional-->**Provisional: numbers are from the tuning seed.**\n",
        ):
            assert len(banner_problems("d.md", broken, final=False)) == 1, broken

    def test_the_word_provisional_is_gone_once_the_results_use_the_evaluation_seed(
        self,
    ):
        assert banner_problems("d.md", "intro\n", final=True) == []
        for left_over in (
            BANNER,
            "<!--provisional-->",
            "Provisional: numbers are from the tuning seed.",
        ):
            assert len(banner_problems("d.md", left_over, final=True)) == 1

    def test_both_docs_carry_the_banner_exactly_while_the_seed_is_the_tuning_seed(
        self,
    ):
        risk = json.loads((RESULTS_DIR / "risk.json").read_text())
        final = risk["seed"] == EvalConfig().eval_seed
        problems = []
        for name in ("README.md", "docs/RESULTS.md"):
            text = (ROOT / name).read_text()
            problems += banner_problems(name, text, final)
        assert problems == [], (
            "the provisional banner follows risk.json's seed (the tuning seed: the "
            "full banner line; the evaluation seed: no banner at all): "
            f"{problems}"
        )


class TestResultsComeFromOneRun:
    def test_a_half_regenerated_results_set_is_not_published(self):
        problems = run_identity_problems()
        assert problems == [], (
            "the four eval.run results files must come from one run (the same "
            "seed, git_sha and config_hash): copy all four from the same archive "
            f"into results/v1.0: {problems}"
        )

    @pytest.mark.parametrize("key", ["seed", "git_sha", "config_hash"])
    @pytest.mark.parametrize("stale", ["risk", "forecast", "policy", "privacy"])
    def test_a_results_file_from_another_run_is_reported_whichever_file_it_is(
        self, tmp_path, stale, key
    ):
        # The other value is derived from the committed one, so the case stays a
        # difference when the release regenerates every file on the evaluation seed.
        other = None
        for name in ("risk", "forecast", "policy", "privacy"):
            payload = json.loads((RESULTS_DIR / f"{name}.json").read_text())
            if name == stale:
                other = payload[key] + 1 if key == "seed" else f"{payload[key]}-other"
                payload[key] = other
            (tmp_path / f"{name}.json").write_text(json.dumps(payload))
        problems = run_identity_problems(tmp_path)
        assert any(
            key in p and "differs" in p and f"'{stale}': {other!r}" in p
            for p in problems
        ), problems

    @pytest.mark.parametrize("key", ["seed", "git_sha", "config_hash"])
    def test_a_key_missing_from_every_results_file_fails_loudly(self, tmp_path, key):
        for name in ("risk", "forecast", "policy", "privacy"):
            payload = json.loads((RESULTS_DIR / f"{name}.json").read_text())
            del payload[key]
            (tmp_path / f"{name}.json").write_text(json.dumps(payload))
        with pytest.raises(KeyError, match=key):
            run_identity_problems(tmp_path)

    def test_a_seed_that_is_neither_frozen_seed_is_reported(self, tmp_path):
        for name in ("risk", "forecast", "policy", "privacy"):
            payload = json.loads((RESULTS_DIR / f"{name}.json").read_text())
            payload["seed"] = 7
            (tmp_path / f"{name}.json").write_text(json.dumps(payload))
        problems = run_identity_problems(tmp_path)
        assert len(problems) == 1 and "neither" in problems[0], problems


SYMBOL_RE = re.compile(r"^(core|data|models|model|train|export|eval|policy)(\.\w+)+$")
PATH_EXT = (".py", ".yaml", ".md", ".json", ".png", ".ipynb")
# Paths a doc may name although they are gitignored and so absent from a checkout.
ALLOWED_PREFIXES = ("runs/", "checkpoints/")
# Config sections a doc may name as `section.key`; a key is a dataclass field.
CONFIG_SECTIONS = {
    "model": ModelConfig,
    "data": DataConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
    "policy": PolicyConfig,
}
NESTED_CONFIG = {(DataConfig, "synthetic"): SyntheticConfig}


def _spans(text: str) -> list[str]:
    """Single-backtick spans of prose; fenced blocks are skipped."""
    return [s.strip() for s in SPAN.findall(FENCE_RE.sub("", text))]


def _normalize(ref: str) -> str:
    """A leading ./ names the same repo path."""
    while ref.startswith("./"):
        ref = ref[2:]
    return ref


def doc_symbols(text: str) -> list[str]:
    return [span for span in _spans(text) if SYMBOL_RE.match(span)]


def _path_like(span: str) -> bool:
    if re.search(r"[\s*{}<>]", span) or re.match(r"\w+://", span):
        return False
    return "/" in span or span.split("::")[0].split("#")[0].endswith(PATH_EXT)


def doc_paths(text: str) -> list[str]:
    spans = [s for s in _spans(text) if _path_like(s)]
    return [_normalize(s.split("::")[0].split("#")[0]) for s in spans]


def doc_nodes(text: str) -> list[tuple[str, str]]:
    """(file, name) of every non-test `path::name` citation. A method is named
    path::Class.method or path::Class::method; both give name Class.method."""
    nodes = []
    for span in _spans(text):
        span = _normalize(span)
        if _path_like(span) and "::" in span and not span.startswith("tests/"):
            path, rest = span.split("::", 1)
            nodes.append((path, rest.replace("::", ".")))
    return nodes


def _defs(body: list[ast.stmt]) -> dict[str, ast.stmt]:
    """Functions and classes defined directly in a module or class body."""
    return {
        n.name: n
        for n in body
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }


def node_problem(ref: str, name: str) -> str | None:
    """Why `ref::name` does not name a definition, or None. Every dotted part is
    looked up in the body of the one before it: a top-level def or class, then
    a method of that class."""
    if Path(ref).is_absolute():
        return "absolute path"
    if not ref.endswith(".py"):
        return "not a .py file"
    if not (ROOT / ref).is_file():
        return "no such file"
    body = ast.parse((ROOT / ref).read_text()).body
    parts = name.split(".")
    for i, part in enumerate(parts):
        node = _defs(body).get(part)
        if node is None:
            return f"{part} is not defined" + (f" in {parts[i - 1]}" if i else "")
        if i < len(parts) - 1 and not isinstance(node, ast.ClassDef):
            return f"{part} is not a class"
        body = node.body
    return None


def _resolve_config_key(dotted: str):
    section, *keys = dotted.split(".")
    cls = CONFIG_SECTIONS[section]
    previous = section
    for key in keys:
        if cls is None:
            raise AttributeError(f"{previous} is a leaf field and has no {key}")
        if key not in {f.name for f in dataclasses.fields(cls)}:
            raise AttributeError(f"{cls.__name__} has no field {key}")
        cls = NESTED_CONFIG.get((cls, key))
        previous = key
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
    if parts[0] in CONFIG_SECTIONS:  # model and policy are no importable module
        return _resolve_config_key(dotted)
    raise ImportError(dotted)


def _label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def missing_symbols(docs=None) -> list[str]:
    """Every symbol a doc names that no longer resolves (default: public docs)."""
    missing = []
    for path in _public_docs() if docs is None else docs:
        for sym in doc_symbols(path.read_text()):
            try:
                resolve_symbol(sym)
            except (ImportError, AttributeError):
                missing.append(f"{_label(path)}: {sym}")
    return missing


def _leaves_repo(ref: str) -> bool:
    """An absolute path, or one that climbs above the repo root."""
    return os.path.isabs(ref) or os.path.normpath(ref).split(os.sep)[0] == ".."


def missing_paths(docs=None) -> list[str]:
    """Every repo path a doc names that does not exist (default: public docs).
    Public docs name repo-relative paths only, so a path that leaves the repo is
    missing whether or not it exists on this machine."""
    return [
        f"{_label(p)}: {ref}"
        for p in (_public_docs() if docs is None else docs)
        for ref in doc_paths(p.read_text())
        if _leaves_repo(ref)
        or (not ref.startswith(ALLOWED_PREFIXES) and not (ROOT / ref).exists())
    ]


def missing_nodes(docs=None) -> list[str]:
    """Every non-test path::name citation that is not a definition there."""
    missing = []
    for path in _public_docs() if docs is None else docs:
        for ref, name in doc_nodes(path.read_text()):
            problem = node_problem(ref, name)
            if problem:
                missing.append(f"{_label(path)}: {ref}::{name} ({problem})")
    return missing


class TestDocReferences:
    def test_every_symbol_the_docs_name_still_exists(self):
        assert missing_symbols() == []

    def test_every_path_the_docs_name_exists(self):
        assert missing_paths() == []

    def test_every_non_test_node_the_docs_name_is_defined_there(self):
        assert missing_nodes() == []

    def test_a_stale_symbol_config_key_path_or_node_in_a_doc_is_reported(self):
        bad = [FIXR / "bad.md"]
        assert sorted(missing_symbols(bad)) == [
            "tests/fixtures/docs_refs/bad.md: core.privacy.NoSuchManager",
            "tests/fixtures/docs_refs/bad.md: data.synthetic.no_such_field",
        ]
        assert sorted(missing_paths(bad)) == [
            "tests/fixtures/docs_refs/bad.md: /etc/hosts.py",
            "tests/fixtures/docs_refs/bad.md: config/no_such_file.yaml",
            "tests/fixtures/docs_refs/bad.md: xruns/best.pt",
        ]
        nodes = missing_nodes(bad)
        assert [n.split(" (")[0].split(": ")[1] for n in nodes] == [
            "train/loop.py::missing_name",
            "core/privacy.py::PrivacyManager.no_such_method",
            "core/privacy.py::PrivacyManager.no_such_method",
            "README.md::Usage",
            "/etc/hosts.py::x",
        ]
        assert [n.rsplit(" (", 1)[1] for n in nodes] == [
            "missing_name is not defined)",
            "no_such_method is not defined in PrivacyManager)",
            "no_such_method is not defined in PrivacyManager)",
            "not a .py file)",
            "absolute path)",
        ]

    def test_a_doc_whose_references_are_all_real_reports_nothing(self):
        good = [FIXR / "good.md"]
        assert missing_symbols(good) == []
        assert missing_paths(good) == []
        assert missing_nodes(good) == []

    @pytest.mark.parametrize(
        "check",
        [missing_symbols, missing_paths, missing_nodes],
        ids=lambda f: f.__name__,
    )
    def test_the_repo_wide_checks_scan_the_public_docs_by_default(
        self, check, monkeypatch
    ):
        monkeypatch.setattr(
            sys.modules[__name__], "_public_docs", lambda: [FIXR / "bad.md"]
        )
        assert check() == check([FIXR / "bad.md"]) != []

    @pytest.mark.parametrize(
        "root", ["core", "data", "models", "train", "export", "eval"]
    )
    def test_a_symbol_is_any_dotted_name_under_a_package_root(self, root):
        assert doc_symbols(f"`{root}.a.b` and `{root}.c`") == [
            f"{root}.a.b",
            f"{root}.c",
        ]

    @pytest.mark.parametrize(
        "span",
        ["x.y", "core", "xcore.privacy", "core.privacy x", "core.privacy.", "core..p"],
    )
    def test_a_span_that_is_not_wholly_a_package_rooted_name_is_not_a_symbol(
        self, span
    ):
        assert doc_symbols(f"`{span}`") == []

    def test_symbol_extraction_ignores_prose_and_code_fences(self):
        text = (
            "see `core.privacy.PrivacyManager` and `x.y`\n"
            "```\n``data.nope.Q``\n```\n"
            "and ` data.synthetic.seed `"
        )
        # a package-rooted token in prose is returned (stripped of padding);
        # x.y has no package root; fenced text is skipped even when its
        # backticks pair up as spans
        assert doc_symbols(text) == [
            "core.privacy.PrivacyManager",
            "data.synthetic.seed",
        ]

    @pytest.mark.parametrize(
        "span",
        ["a/b", "docs/img", "a.py", "a.yaml", "a.md", "a.json", "a.png", "a.ipynb"],
    )
    def test_a_slash_or_a_known_extension_makes_a_span_a_path(self, span):
        assert doc_paths(f"`{span}`") == [span]

    @pytest.mark.parametrize("span", ["a.txt", "n", "a.pyc", "core.privacy"])
    def test_a_span_with_neither_slash_nor_known_extension_is_not_a_path(self, span):
        assert doc_paths(f"`{span}`") == []

    @pytest.mark.parametrize(
        "span",
        [
            "a b/c.py",
            "docs/*.md",
            "a/{b,c}.py",
            "a/{b.py",
            "a/}b.py",
            "runs/<task>/x.pt",
            "a/<b.py",
            "a/>b.py",
            "https://example.org/x.md",
        ],
    )
    def test_a_placeholder_url_or_spaced_span_is_not_a_path(self, span):
        assert doc_paths(f"`{span}`") == []

    def test_path_extraction_strips_suffixes_and_skips_placeholders(self):
        text = (
            "`tests/test_core.py::TestX::test_y` `README.md#usage` "
            "`config/synthetic_{easy,hard}.yaml` `runs/<task>/best.pt` "
            "`a b/c.py` `docs/*.md` `https://example.org/x.md` `core.privacy`\n"
            "```\n``fenced/path.py``\n```"
        )
        assert doc_paths(text) == ["tests/test_core.py", "README.md"]

    def test_a_citation_is_a_path_even_when_only_its_file_part_has_an_extension(
        self,
    ):
        assert doc_paths("`loop.py::fit` `a.md#x`") == ["loop.py", "a.md"]

    def test_a_leading_dot_slash_names_the_same_path(self):
        assert doc_paths("`./runs/risk/best.pt` `././a/b.py`") == [
            "runs/risk/best.pt",
            "a/b.py",
        ]

    def test_node_extraction_returns_only_non_test_citations(self):
        text = "`train/loop.py::fit` `tests/test_core.py::TestX` `eval/run.py`"
        assert doc_nodes(text) == [("train/loop.py", "fit")]

    def test_a_method_citation_gives_a_dotted_name_for_either_spelling(self):
        text = "`a/b.py::C.m` `a/b.py::C::m` `./a/b.py::f` `./tests/x.py::Y`"
        assert doc_nodes(text) == [
            ("a/b.py", "C.m"),
            ("a/b.py", "C.m"),
            ("a/b.py", "f"),
        ]

    def test_a_node_is_a_top_level_def_or_a_method_of_a_top_level_class(self):
        assert node_problem("train/loop.py", "fit") is None
        assert node_problem("core/privacy.py", "PrivacyManager") is None
        assert node_problem("core/privacy.py", "PrivacyManager.pseudonymize") is None
        assert node_problem("train/loop.py", "missing_name")
        assert node_problem("core/privacy.py", "PrivacyManager.gone")
        assert node_problem("train/loop.py", "fit.inner") == "fit is not a class"
        assert node_problem("train/nope.py", "fit") == "no such file"

    def test_only_definitions_made_directly_in_a_body_are_defined_there(self):
        source = (
            "def top():\n    def nested():\n        pass\n\n"
            "async def atop():\n    pass\n\n"
            "class Outer:\n    def method(self):\n        pass\n\n"
            "    class Inner:\n        pass\n"
        )
        defs = _defs(ast.parse(source).body)
        assert set(defs) == {"top", "atop", "Outer"}
        assert set(_defs(defs["Outer"].body)) == {"method", "Inner"}

    def test_a_nested_function_is_not_a_node_but_a_method_is_one_of_its_class(
        self, tmp_path, monkeypatch
    ):
        # ROOT is a module global read by node_problem, so a module in tmp_path is
        # looked up like a repo file.
        monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
        (tmp_path / "mod.py").write_text(
            "def outer():\n    def inner():\n        pass\n\n\n"
            "class K:\n    def meth(self):\n        pass\n\n"
            "    class Sub:\n        pass\n"
        )
        cited = (
            "`mod.py::outer` `mod.py::K` `mod.py::K.meth` `mod.py::K::meth` "
            "`mod.py::K.Sub` "
            "`mod.py::inner` `mod.py::meth` `mod.py::Sub` `mod.py::outer.inner` "
            "`mod.py::K.nope`"
        )
        doc = _doc(tmp_path, cited + "\n")
        assert missing_nodes([doc]) == [
            "doc.md: mod.py::inner (inner is not defined)",
            "doc.md: mod.py::meth (meth is not defined)",
            "doc.md: mod.py::Sub (Sub is not defined)",
            "doc.md: mod.py::outer.inner (outer is not a class)",
            "doc.md: mod.py::K.nope (nope is not defined in K)",
        ]

    @pytest.mark.parametrize("fence", ["```", "~~~"])
    def test_references_between_two_fenced_blocks_are_still_checked(
        self, tmp_path, fence
    ):
        block = f"{fence}\ncode\n{fence}\n"
        prose = "`core.privacy.NoSuch` `config/dead.yaml` `train/loop.py::gone`\n"
        doc = _doc(tmp_path, block + prose + block)
        assert missing_symbols([doc]) == [f"{doc}: core.privacy.NoSuch"]
        assert missing_paths([doc]) == [f"{doc}: config/dead.yaml"]
        assert missing_nodes([doc]) == [
            f"{doc}: train/loop.py::gone (gone is not defined)"
        ]

    def test_references_inside_a_tilde_fence_are_skipped(self, tmp_path):
        text = "~~~\n`core.nope.Q` `config/fenced.yaml` `train/loop.py::gone`\n~~~\n"
        doc = _doc(tmp_path, text)
        assert missing_symbols([doc]) == []
        assert missing_paths([doc]) == []
        assert missing_nodes([doc]) == []

    @pytest.mark.parametrize(
        "ref",
        [
            "/etc/hosts",
            "/usr",
            "../x",
            "../../etc/hosts",
            "docs/../../x",
            "runs/../../x",
        ],
    )
    def test_a_path_that_leaves_the_repo_is_missing_even_when_it_exists(
        self, tmp_path, ref
    ):
        # Public docs name repo-relative paths only. /etc/hosts and /usr exist on
        # the machine that runs the suite; that must not make them valid.
        doc = _doc(tmp_path, f"`{ref}`\n")
        assert missing_paths([doc]) == [f"{doc}: {ref}"]

    @pytest.mark.parametrize(
        "ref, ok",
        [("docs/../README.md", True), ("./docs/../docs", True), ("..x/y.md", False)],
    )
    def test_a_path_that_wanders_but_stays_in_the_repo_is_judged_by_existence(
        self, tmp_path, ref, ok
    ):
        doc = _doc(tmp_path, f"`{ref}`\n")
        assert (missing_paths([doc]) == []) is ok

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
        "key",
        [
            "data.nope",
            "data.synthetic.nope",
            "eval.nope",
            "train.nope",
            "core.nope.Q",
            "core.privacy.Nope",
        ],
    )
    def test_a_renamed_symbol_or_config_key_is_reported(self, key):
        with pytest.raises((ImportError, AttributeError)):
            resolve_symbol(key)

    def test_a_key_past_a_leaf_field_names_the_leaf_in_the_message(self):
        with pytest.raises(AttributeError, match="label_overlap is a leaf field"):
            resolve_symbol("data.label_overlap.x")

    def test_a_prefix_that_fails_to_import_for_any_reason_falls_back_to_a_shorter_one(
        self, monkeypatch
    ):
        module = importlib.import_module("core.privacy")
        real = importlib.import_module

        def broken_for_dotted_names(name, *args):
            if "." in name:
                raise ImportError("a module whose import fails")
            return real(name, *args)

        monkeypatch.setattr(importlib, "import_module", broken_for_dotted_names)
        assert resolve_symbol("core.privacy") is module

    def test_a_name_with_no_importable_root_is_an_import_error(self):
        with pytest.raises(ImportError):
            resolve_symbol("no_such_package_at_all.x")

    @pytest.mark.parametrize(
        "ref, ok",
        [
            ("runs/risk/best.pt", True),
            ("./runs/risk/best.pt", True),
            ("checkpoints/risk/x.pt", True),
            ("xruns/risk/best.pt", False),
            ("config/no_such_file.yaml", False),
            ("config/smoke.yaml", True),
            ("README.md", True),
        ],
    )
    def test_only_gitignored_output_prefixes_may_be_absent(self, tmp_path, ref, ok):
        doc = _doc(tmp_path, f"`{ref}`\n")
        assert (missing_paths([doc]) == []) is ok

    def test_a_file_whose_directory_exists_is_still_missing(self, tmp_path):
        doc = _doc(tmp_path, "`config/definitely_not_here.yaml` `docs/nothing.md`\n")
        assert len(missing_paths([doc])) == 2


FIXQ = ROOT / "tests" / "fixtures" / "docs_quickstart"
QUICKSTART_FENCE_RE = re.compile(
    r"^[ \t]*```bash quickstart[ \t]*\n(.*?)^[ \t]*```", re.DOTALL | re.MULTILINE
)
QUICKSTART_INFO_RE = re.compile(
    r"^[ \t]*```(.*quickstart.*)$", re.MULTILINE | re.IGNORECASE
)
# What a quickstart run must leave under its sandbox, as the README promises.
QUICKSTART_OUTPUTS = (
    "runs/smoke/train/best.pt",
    "m.onnx",
    "results/risk.json",
    "results/forecast.json",
    "results/policy.json",
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


def misspelled_quickstart_fences(md_text: str) -> list[str]:
    """Info strings that mention quickstart but are not exactly `bash quickstart`
    (such a fence would be skipped silently)."""
    return [
        info
        for info in QUICKSTART_INFO_RE.findall(md_text)
        if info.rstrip() != "bash quickstart"
    ]


def sandboxed_config(tmp: Path) -> Path:
    """config/smoke.yaml on the CPU with every output directory moved under tmp."""
    cfg = yaml.safe_load((ROOT / "config" / "smoke.yaml").read_text())
    cfg["device"] = "cpu"
    cfg["train"]["checkpoint_dir"] = f"{tmp}/runs/smoke/train"
    cfg["eval"]["out_dir"] = f"{tmp}/results"
    cfg["eval"]["img_dir"] = f"{tmp}/img"
    out = tmp / "smoke.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


def _git_status_paths(root: Path = ROOT) -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.splitlines()


def _write_snapshot(root: Path = ROOT) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """What a stray write would change: git status, plus size and mtime of every
    file in the directories git ignores or already has dirty (runs/, checkpoints/,
    results/, docs/img/) and of any ONNX file at the repo root."""
    stats = {}
    files = [
        f
        for sub in ("results", "docs/img", "runs", "checkpoints")
        for f in sorted((root / sub).rglob("*"))
    ]
    for file in files + sorted(root.glob("*.onnx")):
        if file.is_file():
            info = file.stat()
            stats[str(file)] = (info.st_size, info.st_mtime_ns)
    return _git_status_paths(root), stats


def stray_writes(before, after) -> list[str]:
    """What differs between two _write_snapshot results: a git status line that
    appeared or went away, and a file that is new, gone, or changed in size or
    mtime. Empty when the quickstart touched nothing in the repo."""
    status_before, stats_before = before
    status_after, stats_after = after
    changes = [
        f"git status: {line}" for line in sorted(set(status_before) ^ set(status_after))
    ]
    for file in sorted(stats_before.keys() | stats_after.keys()):
        if stats_before.get(file) != stats_after.get(file):
            changes.append(
                f"{file}: {stats_before.get(file)} -> {stats_after.get(file)}"
            )
    return changes


def _rewrite(token: str, cfg: Path, tmp: Path) -> str:
    token = _normalize(token)
    if token == "config/smoke.yaml":
        return str(cfg)
    if token.startswith("runs/"):
        return f"{tmp}/{token}"
    if token == "dignity_risk.onnx":
        return f"{tmp}/m.onnx"
    return token


def _sandboxed_argv(cmd: str, cfg: Path, tmp: Path) -> list[str]:
    """Token-wise rewrite (also of --flag=value) so a command only writes under tmp."""
    argv = []
    for token in shlex.split(cmd):
        if token.startswith("--") and "=" in token:
            flag, value = token.split("=", 1)
            token = f"{flag}={_rewrite(value, cfg, tmp)}"
        else:
            token = _rewrite(token, cfg, tmp)
        argv.append(token)
    assert argv[0] == "python", f"a quickstart command must run python: {cmd}"
    argv[0] = sys.executable
    return argv


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        # an identity of its own: a CI runner has none configured
        [
            "git",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.name=dignity-tests",
            "-c",
            "user.email=dignity-tests@example.invalid",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    """A committed tree shaped like this repo: .onnx files and runs/ are ignored,
    results/ and docs/img/ hold committed files."""
    root = tmp_path / "repo"
    files = {
        ".gitignore": "*.onnx\nruns/\ncheckpoints/\n",
        "README.md": "readme\n",
        "results/risk.json": '{"a": 1}\n',
        "docs/img/c.png": "png",
    }
    for name, body in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(body)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "--no-verify", "-m", "tree")
    return root


def _touch(path: Path) -> None:
    ns = path.stat().st_mtime_ns + 5_000_000_000
    os.utime(path, ns=(ns, ns))


def _grow(path: Path) -> None:
    old = path.stat().st_mtime_ns
    path.write_text(path.read_text() + "more\n")
    os.utime(path, ns=(old, old))  # same mtime, so only the size gives it away


def _write(rel: str, body: str = "x"):
    def act(root: Path) -> None:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(body)

    return act


class TestQuickstartStrayWrites:
    def test_a_tree_nothing_wrote_to_has_no_stray_write(self, repo):
        before = _write_snapshot(repo)
        assert stray_writes(before, _write_snapshot(repo)) == []

    def test_the_snapshot_of_this_repo_sees_the_committed_results(self):
        stats = _write_snapshot()[1]
        assert str(ROOT / "results" / "v1.0" / "risk.json") in stats

    @pytest.mark.parametrize(
        "act, culprit",
        [
            (_write("m.onnx"), "m.onnx"),
            (_write("runs/smoke/best.pt"), "best.pt"),
            (_write("checkpoints/risk/x.pt"), "x.pt"),
            (_write("results/new.json"), "new.json"),
            (_write("docs/img/new.png"), "new.png"),
            (_write("stray.txt"), "stray.txt"),
            (_write("README.md", "changed readme\n"), "README.md"),
            (_write("results/risk.json", '{"a": 22}\n'), "risk.json"),
            (lambda r: _touch(r / "results" / "risk.json"), "risk.json"),
            (lambda r: _grow(r / "docs" / "img" / "c.png"), "c.png"),
            (lambda r: (r / "results" / "risk.json").unlink(), "risk.json"),
        ],
        ids=[
            "an ONNX file at the repo root",
            "a new file under runs/",
            "a new file under checkpoints/",
            "a new file under results/",
            "a new file under docs/img/",
            "a new untracked file elsewhere",
            "a committed file elsewhere modified",
            "a committed results file changed in size",
            "a committed results file touched, same content",
            "a committed image grown, same mtime",
            "a committed results file deleted",
        ],
    )
    def test_each_kind_of_stray_write_is_detected(self, repo, act, culprit):
        before = _write_snapshot(repo)
        act(repo)
        changes = stray_writes(before, _write_snapshot(repo))
        assert any(culprit in change for change in changes), changes


class TestQuickstart:
    def test_quickstart_extraction_reads_only_marked_fences(self):
        assert quickstart_commands((FIXQ / "sample.md").read_text()) == [
            "python -m train.cli --config config/smoke.yaml",
            "python -m export.to_onnx --checkpoint runs/smoke/train/best.pt "
            "--output dignity_risk.onnx",
        ]

    def test_a_mistyped_quickstart_info_string_is_reported_not_skipped(self):
        assert misspelled_quickstart_fences("```bash quickstart\nx\n```\n") == []
        for info in (
            " bash quickstart",
            "bash  quickstart",
            "bash Quickstart",
            "sh quickstart",
        ):
            assert misspelled_quickstart_fences(f"```{info}\nx\n```\n") == [info], info
        assert misspelled_quickstart_fences((ROOT / "README.md").read_text()) == []

    def test_sandboxed_config_moves_every_output_under_tmp_and_pins_the_cpu(
        self, tmp_path
    ):
        cfg = yaml.safe_load(sandboxed_config(tmp_path).read_text())
        for out in (
            cfg["train"]["checkpoint_dir"],
            cfg["eval"]["out_dir"],
            cfg["eval"]["img_dir"],
        ):
            assert out.startswith(str(tmp_path))
        assert cfg["device"] == "cpu"

    @pytest.mark.parametrize(
        "cmd, expected",
        [
            ("python -m train.cli --config config/smoke.yaml", ["--config", "CFG"]),
            ("python -m train.cli --config=config/smoke.yaml", ["--config=CFG"]),
            ("python -m train.cli --config ./config/smoke.yaml", ["--config", "CFG"]),
            ("python -m x --output dignity_risk.onnx", ["--output", "TMP/m.onnx"]),
            ("python -m x --output=dignity_risk.onnx", ["--output=TMP/m.onnx"]),
            (
                "python -m x --checkpoint runs/a/best.pt",
                ["--checkpoint", "TMP/runs/a/best.pt"],
            ),
            (
                "python -m x --checkpoint=./runs/a/best.pt",
                ["--checkpoint=TMP/runs/a/best.pt"],
            ),
        ],
    )
    def test_every_output_spelling_is_redirected_under_tmp(
        self, tmp_path, cmd, expected
    ):
        cfg = tmp_path / "smoke.yaml"
        argv = _sandboxed_argv(cmd, cfg, tmp_path)
        want = [
            e.replace("CFG", str(cfg)).replace("TMP", str(tmp_path)) for e in expected
        ]
        assert argv[0] == sys.executable and argv[-len(want) :] == want

    @pytest.mark.parametrize(
        "cmd",
        [
            "pip install x",
            "python3 -m x",
            "dignity-train --config c.yaml",
            "rm -rf runs",
        ],
    )
    def test_a_command_that_is_not_plain_python_is_refused(self, tmp_path, cmd):
        with pytest.raises(AssertionError, match="must run python"):
            _sandboxed_argv(cmd, tmp_path / "c.yaml", tmp_path)

    @pytest.mark.slow
    @pytest.mark.timeout(900)  # three subprocesses, each capped at 300 s
    def test_readme_quickstart_runs_end_to_end_without_touching_committed_files(
        self, tmp_path, monkeypatch
    ):
        before = _write_snapshot()
        assert before[1], "the snapshot saw no files, so a comparison proves nothing"
        compared = []
        real_stray_writes = stray_writes

        def spy(*snapshots):
            compared.append(snapshots)
            return real_stray_writes(*snapshots)

        monkeypatch.setattr(sys.modules[__name__], "stray_writes", spy)
        cfg = sandboxed_config(tmp_path)
        cmds = quickstart_commands((ROOT / "README.md").read_text())
        assert cmds, "README has no `bash quickstart` fence"
        env = {
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
        for cmd in cmds:
            proc = subprocess.run(
                _sandboxed_argv(cmd, cfg, tmp_path),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert proc.returncode == 0, f"{cmd}\n{proc.stderr[-2000:]}"
        for output in QUICKSTART_OUTPUTS:
            produced = tmp_path / output
            assert produced.is_file() and produced.stat().st_size > 0, output
        stray = stray_writes(before, _write_snapshot())
        assert compared and compared[0][0] is before, "the comparison did not run"
        assert stray == [], "the quickstart wrote into the repo"


def _flat(text: str) -> str:
    """Prose with emphasis marks dropped and whitespace collapsed to single spaces."""
    return re.sub(r"\s+", " ", text.replace("**", ""))


def _between(text: str, start: str, end: str) -> str:
    """The text from `start` up to the next `end` after it."""
    first = text.index(start)
    return text[first : text.index(end, first)]


def _first_table(section: str) -> list[list[str]]:
    """Body rows of the first markdown table in `section`, as stripped cells."""
    rows: list[list[str]] = []
    for line in section.splitlines():
        if line.startswith("|"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif rows:
            break
    return rows[2:]


IDENT_CHAIN_RE = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")
NAME_PACKAGES = ("core", "data", "models", "train", "eval", "export", "scripts")
# Backticked words in ARCHITECTURE.md that the repo neither defines as a name nor
# writes as a string: library and builtin names, and a symbol of the prose.
ARCH_LITERALS = {
    "None": "the Python constant",
    "ValueError": "the builtin exception the code raises",
    "RobustScaler": "the scikit-learn scaler the pipeline wraps",
    "t": "the last row of a window, a symbol of the prose",
    "attn_weights": "the name of the backbone's second return value, a local variable",
}


def _walk_repo():
    for package in NAME_PACKAGES:
        for path in sorted((ROOT / package).rglob("*.py")):
            yield ast.parse(path.read_text())


def repo_names() -> set[str]:
    """Names the repo's packages define: top-level defs, classes and module-level
    assignments, plus every method, class-level field and `self.x` attribute of a
    class."""
    names: set[str] = set()

    def add(body: list[ast.stmt], in_class: bool) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(node.name)
                if isinstance(node, ast.ClassDef):
                    add(node.body, True)
                elif in_class:
                    names.update(
                        n.attr
                        for n in ast.walk(node)
                        if isinstance(n, ast.Attribute)
                        and isinstance(n.ctx, ast.Store)
                        and isinstance(n.value, ast.Name)
                        and n.value.id == "self"
                    )
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                names.update(
                    n.id
                    for n in ast.walk(target)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                )

    for tree in _walk_repo():
        add(tree.body, False)
    return names


def repo_strings() -> set[str]:
    """Identifier-like string constants of the repo's packages: dictionary keys,
    column names, actions, metric names and the like."""
    return {
        n.value
        for tree in _walk_repo()
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and IDENT_CHAIN_RE.fullmatch(n.value)
    }


def bare_identifiers(text: str) -> list[str]:
    """Backticked spans that are an identifier or a dotted chain of them and that
    no other checker owns: not a path, not a config key, not a package-rooted
    symbol."""
    return [
        span
        for span in _spans(text)
        if IDENT_CHAIN_RE.fullmatch(span)
        and not SYMBOL_RE.match(span)
        and span.split(".")[0] not in CONFIG_SECTIONS
        and not _path_like(span)
    ]


def repo_class_members() -> dict[str, set[str]]:
    """Class name -> its members: methods, class-level fields and `self.x`
    attributes, plus those of base classes the repo defines."""
    own: dict[str, set[str]] = {}
    bases: dict[str, set[str]] = {}
    for tree in _walk_repo():
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            members = own.setdefault(cls.name, set())
            bases.setdefault(cls.name, set()).update(
                b.id for b in cls.bases if isinstance(b, ast.Name)
            )
            for item in cls.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    members.add(item.name)
                    members.update(
                        n.attr
                        for n in ast.walk(item)
                        if isinstance(n, ast.Attribute)
                        and isinstance(n.ctx, ast.Store)
                        and isinstance(n.value, ast.Name)
                        and n.value.id == "self"
                    )
                elif isinstance(item, ast.Assign | ast.AnnAssign):
                    targets = (
                        item.targets if isinstance(item, ast.Assign) else [item.target]
                    )
                    members.update(
                        n.id
                        for t in targets
                        for n in ast.walk(t)
                        if isinstance(n, ast.Name)
                    )

    def full(name: str, seen: frozenset = frozenset()) -> set[str]:
        if name in seen:
            return set()
        out = set(own[name])
        for base in bases[name] & own.keys():
            out |= full(base, seen | {name})
        return out

    return {name: full(name) for name in own}


def missing_identifiers(text: str) -> list[str]:
    """Bare identifiers of `text` with a part that is neither a name the repo defines
    nor a string it writes nor a listed literal. A chain whose first part is a repo
    class is resolved as a chain: its second part must be a member of that class."""
    known = repo_names() | repo_strings() | set(ARCH_LITERALS)
    members = repo_class_members()

    def defined(span: str) -> bool:
        first, *rest = span.split(".")
        if rest and first in members and rest[0] not in members[first]:
            return False
        return span in known or all(p in known for p in span.split("."))

    return sorted({span for span in bare_identifiers(text) if not defined(span)})


def _onnx_graph(tmp_path):
    """A tiny exported risk model, loaded back."""
    import onnx

    from export.to_onnx import export_to_onnx
    from models.dignity import Dignity

    model = Dignity(task="risk", input_size=3, hidden_size=8, n_layers=1)
    path = tmp_path / "tiny.onnx"
    assert export_to_onnx(model, str(path), input_shape=(1, 12, 3))
    return onnx.load(str(path))


class TestArchitectureDoc:
    """docs/ARCHITECTURE.md describes the code as it is: its tables, constants,
    shapes and scope sentences are parsed and compared with the code, and every
    backticked name must be one the repo defines."""

    ARCH = ROOT / "docs" / "ARCHITECTURE.md"

    def _frame(self):
        from core.config import SyntheticConfig
        from data.source.synthetic import SyntheticGenerator

        config = SyntheticConfig(n_blocks=2, block_len=20)
        return SyntheticGenerator(config).generate()

    def test_data_contract_table_is_exactly_the_generated_frame(self):
        frame = self._frame()
        text = self.ARCH.read_text()
        rows = _first_table(_between(text, "## Data contract", "### Labels"))
        documented = {cells[0].strip("`"): cells[1] for cells in rows}
        assert list(documented) == list(frame.columns)
        for col, cell in documented.items():
            actual = str(frame[col].dtype)
            if cell.startswith("str"):
                # object under pandas 2, str under pandas 3; the cell says so
                assert actual in ("object", "str", "string"), col
                assert actual != "object" or "object under pandas 2" in cell, col
            else:
                assert cell == actual, col

    def test_anomaly_ids_count_every_injected_anomaly_once(self):
        from core.config import SyntheticConfig
        from data.source.synthetic import SyntheticGenerator

        config = SyntheticConfig(n_blocks=40, block_len=160, contamination=1.0)
        frame = SyntheticGenerator(config).generate()
        ids = frame["anomaly_id"].to_numpy()
        injected = ids[ids >= 0]
        assert (ids[~frame["is_anomalous"].to_numpy()] == -1).all()
        assert (np.diff(injected) >= 0).all()
        assert sorted(set(injected.tolist())) == list(range(injected.max() + 1))

    def test_doc_names_every_implemented_anomaly_type(self):
        from data.source.anomalies import ANOMALY_TYPES

        text = self.ARCH.read_text()
        assert all(f"`{t}`" in text for t in ANOMALY_TYPES)

    def test_each_anomaly_row_names_the_columns_its_injector_changes(self):
        # Swapping two rows of the mechanism table must fail: the row of a type
        # lists the raw columns that injector changes at full severity, each with
        # the direction it moves, and every claim is measured on a real block.
        from core.config import SyntheticConfig
        from data.source.anomalies import ANOMALY_INJECTORS, ANOMALY_TYPES
        from data.source.synthetic import RAW_COLUMNS, SyntheticGenerator

        block = SyntheticGenerator(
            SyntheticConfig(n_blocks=1, block_len=120, contamination=0.0)
        ).generate()
        start, duration = 40, 20
        span = slice(start, start + duration)
        text = self.ARCH.read_text()
        rows = _first_table(_between(text, "`data.source.anomalies.", "### Labels"))
        by_type = {cells[0].strip("`"): cells for cells in rows}
        assert set(by_type) == set(ANOMALY_TYPES)
        for kind in ANOMALY_TYPES:
            out = ANOMALY_INJECTORS[kind](
                block, start, duration, 1.0, np.random.default_rng(0)
            )
            changed = {
                c
                for c in RAW_COLUMNS
                if not np.allclose(out[c].to_numpy(float), block[c].to_numpy(float))
            }
            documented = dict(re.findall(r"`(\w+)` (\w+)", by_type[kind][1]))
            assert set(documented) == changed, kind
            for col, word in documented.items():
                new = out[col].to_numpy(float)
                old = block[col].to_numpy(float)
                if word == "up":
                    assert (new >= old - 1e-9).all() and (new > old + 1e-9).any(), kind
                elif word == "down":
                    assert (new <= old + 1e-9).all() and (new < old - 1e-9).any(), kind
        price = {
            kind: ANOMALY_INJECTORS[kind](
                block, start, duration, 1.0, np.random.default_rng(0)
            )["price"].to_numpy(float)
            for kind in ("wash_trading", "pump_and_dump", "flash_crash")
        }
        old = block["price"].to_numpy(float)
        assert "damped" in by_type["wash_trading"][1]
        wash = np.abs(np.diff(np.log(price["wash_trading"])))[
            start - 1 : start - 1 + duration
        ]
        assert (
            wash < np.abs(np.diff(np.log(old)))[start - 1 : start - 1 + duration]
        ).all()
        assert "rises" in by_type["pump_and_dump"][1]
        assert price["pump_and_dump"][start] > old[start]
        assert "drops" in by_type["flash_crash"][1]
        assert price["flash_crash"][start] < old[start]
        assert "first row, where volume peaks" in by_type["flash_crash"][2]
        crash = ANOMALY_INJECTORS["flash_crash"](
            block, start, duration, 1.0, np.random.default_rng(0)
        )["volume"].to_numpy(float)
        ratio = crash[span] / block["volume"].to_numpy(float)[span]
        assert ratio[0] > ratio[1:].max() and (ratio[1:] > 1).all()
        assert (
            "volume stays raised for the rest of the span" in by_type["flash_crash"][2]
        )

    def test_fee_spike_and_layering_rows_state_the_measured_multipliers(self):
        from core.config import SyntheticConfig
        from data.source.anomalies import ANOMALY_INJECTORS
        from data.source.synthetic import SyntheticGenerator

        block = SyntheticGenerator(
            SyntheticConfig(n_blocks=1, block_len=120, contamination=0.0)
        ).generate()
        start, duration = 40, 20
        span = slice(start, start + duration)
        rng = np.random.default_rng(0)
        spike = ANOMALY_INJECTORS["fee_spike"](block, start, duration, 1.0, rng)
        fee = spike["fee_rate"].to_numpy(float) / block["fee_rate"].to_numpy(float)
        tx = spike["tx_count"].to_numpy(float) / block["tx_count"].to_numpy(float)
        assert np.allclose(fee[span], 10.0) and np.allclose(tx[span], 2.0)
        layered = ANOMALY_INJECTORS["layering"](block, start, duration, 1.0, rng)
        rows = np.flatnonzero(
            ~np.isclose(layered["volume"].to_numpy(), block["volume"].to_numpy())
        )
        assert rows.tolist() == list(range(start, start + duration, 4))
        text = _flat(self.ARCH.read_text())
        assert "`fee_rate` is multiplied by up to 10 and `tx_count` by up to 2" in text
        assert "Every fourth row gets an added volume spike" in text

    def test_doc_makes_no_trading_or_agent_claim(self):
        # Regression guard: green before the rewrite. Word-bounded so that
        # "support" and UnsupportedCheckpointError stay writable; the REMOVED
        # list already covers the forecasting phrases.
        text = self.ARCH.read_text().lower()
        assert re.search(r"\b(ppo|a3c)\b", text) is None
        assert "trading agent" not in text
        # the three tasks are about risk: no trading or price-forecasting framing
        assert re.search(r"\b(trading|quant|alpha)\b", text) is None
        assert re.search(r"price (forecast|prediction)|predict prices", text) is None

    def test_architecture_doc_has_no_pre_v1_names(self):
        # Guard: red on the old document, which named load_crypto_prices,
        # create_dignity_model, create_dataloaders and load_and_process; it keeps
        # every name that is defined nowhere in the code out of the new one.
        text = self.ARCH.read_text()
        for stale in (
            "compute_volatility",
            "compute_entropy",
            "compute_momentum",
            "compute_all_signals",
            "detect_regime",
            "load_crypto_prices",
            "create_dignity_model",
            "create_dataloaders",
            "load_and_process",
            "test_size",
        ):
            assert stale not in text, stale

    def test_every_signal_the_pipeline_derives_is_documented(self):
        from data.pipeline import SIGNAL_SOURCES

        text = self.ARCH.read_text()
        for name in SIGNAL_SOURCES:
            assert f"`{name}`" in text, name

    def test_signal_constants_are_the_ones_the_pipeline_computes_with(self):
        from data.pipeline import SIGNAL_SOURCES, TransactionPipeline

        text = _flat(self.ARCH.read_text())
        (window,) = re.findall(r"standard deviation of the (\d+) rows before", text)
        (lag,) = re.findall(r"\(x\[i\] - x\[i-(\d+)\]\) / x\[i-\1\]", text)
        up, down = re.findall(r"1 above `\+([\d.]+)`, -1 below `-([\d.]+)`", text)[0]
        assert up == down
        window, lag, threshold = int(window), int(lag), float(up)
        rng = np.random.default_rng(7)
        n = window + 40
        frame = pd.DataFrame(
            {
                "volume": rng.uniform(1, 100, n),
                "price": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))),
            }
        )
        out = TransactionPipeline().compute_signals(frame)
        assert set(SIGNAL_SOURCES) <= set(out.columns)
        i = window + 15
        for name, source in (("volume_volatility", "volume"), ("volatility", "price")):
            values = frame[source].to_numpy()
            assert out[name].iloc[0] == 0
            assert out[name].iloc[i] == pytest.approx(
                np.std(values[i - window : i]), rel=1e-9
            )
            assert out[name].iloc[3] == pytest.approx(np.std(values[0:3]), rel=1e-9)
        price = frame["price"].to_numpy()
        momentum = out["momentum"].to_numpy()
        assert (momentum[:lag] == 0).all()
        assert momentum[lag] == pytest.approx((price[lag] - price[0]) / price[0])
        assert momentum[i] == pytest.approx(
            (price[i] - price[i - lag]) / price[i - lag]
        )
        steps = np.array(
            [
                1.0,
                1 + 1.5 * threshold,
                1 - 1.5 * threshold,
                1 + 0.5 * threshold,
                1 - 0.5 * threshold,
                1.0,
            ]
        )
        series = pd.DataFrame({"volume": np.ones(6), "price": 100 * np.cumprod(steps)})
        direction = TransactionPipeline().compute_signals(series)["directional_change"]
        assert direction.tolist() == [0, 1, -1, 0, 0, 0]

    def test_sections_that_later_docs_link_to_exist_in_pipeline_order(self):
        text = self.ARCH.read_text()
        headings = re.findall(r"^## (.+)$", text, re.MULTILINE)
        wanted = [
            "Data contract",
            "Pipeline",
            "Signals",
            "Backbone",
            "Heads",
            "Training and evaluation",
            "Checkpoint and export",
        ]
        assert [h for h in headings if h in wanted] == wanted

    def test_every_triage_action_is_named(self):
        from models.head.policy import ACTIONS

        text = self.ARCH.read_text()
        for action in ACTIONS:
            assert f"`{action}`" in text, action

    def test_cost_table_is_the_shipped_policy_cost_table(self):
        text = self.ARCH.read_text()
        rows = _first_table(_between(text, "| Action |", "### Presets"))
        documented = {
            cells[0].strip("`"): {"anomaly": float(cells[1]), "normal": float(cells[2])}
            for cells in rows
        }
        assert documented == PolicyConfig().costs

    def test_scope_limits_of_v1_are_stated(self):
        # data.source is synthetic-only and nothing releases at inference time;
        # a doc that implied otherwise would describe paths that do not exist.
        text = self.ARCH.read_text()
        assert "`data.source`" in text and "`CryptoSource`" in text
        assert "DGNTY-27" in text
        assert "DGNTY-35" in text

    def test_scope_limits_of_v1_hold_in_the_code_and_are_stated_as_sentences(self):
        text = _flat(self.ARCH.read_text())
        assert "`data.source` is `synthetic` only in v1.0" in text
        assert "v1.0 has no inference-time privacy release" in text
        assert (
            "nothing in the export path adds noise to or generalizes a live input"
            in text
        )
        with pytest.raises(ValueError, match="crypto"):
            DataConfig(source="crypto")
        export_source = (ROOT / "export" / "to_onnx.py").read_text().lower()
        assert "privacy" not in export_source

    def test_cryptosource_is_unsupported_and_two_features_are_never_derived(
        self, tmp_path
    ):
        # F17, F18: two scope sentences that an inversion would make false. Each is
        # pinned as a sentence and tied to the code that makes it true.
        from data.pipeline import TransactionPipeline

        text = _flat(self.ARCH.read_text())
        assert (
            "`CryptoSource` in `data/source/crypto.py` is present but unsupported "
            "until v1.1 (DGNTY-41)" in text
        )
        assert (
            "`price_change`, `regime` and `entropy` are in the `DataConfig` default "
            "list" in text
        )
        assert "but are never derived, so they vanish" in text
        assert (
            "`SignalProcessor` also defines `entropy`, `regime_detection` and "
            "`process_sequence`; the pipeline does not call them" in text
        )
        crypto = tmp_path / "crypto.yaml"
        crypto.write_text("data:\n  source: crypto\n")
        with pytest.raises(ValueError, match="crypto"):
            DignityConfig.from_yaml(str(crypto))
        assert {"entropy", "regime", "price_change"} <= set(DataConfig().features)
        frame = pd.DataFrame(
            {"volume": np.arange(1.0, 61), "price": np.linspace(100, 130, 60)}
        )
        columns = set(TransactionPipeline().compute_signals(frame).columns)
        assert columns.isdisjoint({"entropy", "regime", "price_change"})
        pipeline_source = (ROOT / "data" / "pipeline.py").read_text()
        assert "regime_detection" not in pipeline_source
        assert "entropy" not in pipeline_source

    def test_anomaly_table_lead_in_names_the_test_that_checks_it(self):
        text = _flat(self.ARCH.read_text())
        cited = (
            "tests/test_docs.py::TestArchitectureDoc::"
            "test_each_anomaly_row_names_the_columns_its_injector_changes"
        )
        assert f"`{cited}`" in text
        assert hasattr(type(self), cited.rsplit("::", 1)[1])

    def test_label_policy_and_selection_sentences_are_what_the_code_does(self):
        # D16, D21, D22, D33, D40: each sentence is pinned and its code is run
        from models.head.policy import decide_actions
        from train.loop import _selection_metric

        text = _flat(self.ARCH.read_text())
        assert "strictly greater than `data.label_overlap`" in text
        assert "`block` never counts against the review budget" in text
        assert (
            "Policy ignores `train.select_metric`, `train.pos_weight` and the "
            "chance guard" in text
        )
        assert "if validation AUROC is below `train.chance_guard_auroc`" in text
        assert "`model.n_layers` unidirectional layers" in text
        from data.labels import risk_labels

        flags = np.array([True, False, False, False, False, False])
        at_limit = risk_labels(flags, np.array([0]), 4, 0.25)
        below = risk_labels(flags, np.array([0]), 4, 0.2)
        assert at_limit.tolist() == [0] and below.tolist() == [1]
        probs = np.array(
            [[0.1, 0.1, 0.8], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8], [0.2, 0.6, 0.2]]
        )
        assert decide_actions(probs, 0.25).tolist() == [2, 2, 2, 1]
        config = DignityConfig(model=ModelConfig(task="policy"))
        config.train.select_metric = "auroc"
        assert _selection_metric(config) == "neg_cost"

    def test_head_output_shapes_are_the_documented_ones(self):
        import torch

        from models.dignity import Dignity

        text = self.ARCH.read_text()
        rows = _first_table(_between(text, "| Task | Head |", "Pinned by"))
        documented = {
            cells[0].strip("`"): re.search(r"`\[B, ([^\]]+)\]`", cells[2]).group(1)
            for cells in rows
        }
        assert set(documented) == {"risk", "forecast", "policy"}
        x = torch.randn(2, 12, 3)
        kwargs = {"risk": {}, "forecast": {"horizon": 4}, "policy": {}}
        for task, width in documented.items():
            model = Dignity(task=task, input_size=3, hidden_size=8, **kwargs[task])
            predictions, attention = model.eval()(x)
            expected = 4 if width == "horizon" else int(width)
            assert tuple(predictions.shape) == (2, expected), task
            assert tuple(attention.shape) == (2, 12), task

    def test_backbone_shapes_and_kernel_caveat_are_the_documented_ones(self):
        import torch

        from models.dignity import Dignity

        text = _flat(self.ARCH.read_text())
        assert "maps `x [B, T, F]` to `(context [B, H], attn_weights [B, T])`" in text
        assert (
            "An even kernel adds one step per convolution, so `attn_weights` is "
            "`[B, T+2]`" in text
        )
        x = torch.randn(2, 12, 3)
        for kernel, steps in ((3, 12), (5, 12), (4, 14)):
            backbone = Dignity(
                task="risk", input_size=3, hidden_size=8, cnn_kernel_size=kernel
            ).backbone.eval()
            context, attention = backbone(x)
            assert tuple(context.shape) == (2, 8), kernel
            assert tuple(attention.shape) == (2, steps), kernel
            assert backbone.lstm.bidirectional is False
        for path in sorted((ROOT / "config").glob("*.yaml")):
            assert DignityConfig.from_yaml(str(path)).model.cnn_kernel_size % 2 == 1

    def test_fit_signature_and_record_fields_are_the_documented_ones(self):
        import inspect

        from data.pipeline import SplitData, WindowSet
        from train.loop import FitResult, fit

        text = _flat(self.ARCH.read_text())
        (signature,) = re.findall(r"`train/loop\.py::fit\(([^)]*)\)`", text)
        documented = [p.split("=")[0].strip() for p in signature.split(",")]
        assert documented == list(inspect.signature(fit).parameters)

        def named(pattern: str) -> list[str]:
            (found,) = re.findall(pattern, text)
            return [s for s in SPAN.findall(found) if IDENT_CHAIN_RE.fullmatch(s)]

        def fields(cls) -> list[str]:
            return [f.name for f in dataclasses.fields(cls)]

        assert named(r"`train\.loop\.FitResult`: (.*?)\. ") == fields(FitResult)
        assert named(r"`SplitData` holds (.*?)\. Each") == fields(SplitData)[:3] + [
            "feature_names",
            "scaler_center",
            "scaler_scale",
            "epsilon_spent",
            "privacy_mode",
        ]
        assert named(r"`WindowSet` carries (.*?) \(the dominant") == fields(WindowSet)

    def test_checkpoint_sentences_are_the_documented_format(self):
        from train.checkpoint import FORMAT_VERSION, CheckpointMeta

        text = _flat(self.ARCH.read_text())
        assert re.findall(r"`train\.checkpoint\.FORMAT_VERSION` is (\d+)", text) == [
            str(FORMAT_VERSION)
        ]
        (listed,) = re.findall(
            r"`train\.checkpoint\.CheckpointMeta`: (.*?) \(the whole", text
        )
        documented = SPAN.findall(listed)
        assert documented == [f.name for f in dataclasses.fields(CheckpointMeta)]
        assert "`load_checkpoint` returns it as a `CheckpointMeta`" in text

    def test_export_sentences_are_what_the_export_code_does(self, tmp_path):
        import torch

        from export.to_onnx import main

        text = _flat(self.ARCH.read_text())
        model = _onnx_graph(tmp_path)
        graph = model.graph
        ((opset, input_name, first, second),) = re.findall(
            r"writes opset (\d+) with input `(\w+)` `\[batch, sequence, features\]` "
            r"and outputs `(\w+)` and `(\w+)`",
            text,
        )
        assert [o.version for o in model.opset_import if o.domain == ""] == [int(opset)]
        initializers = {i.name for i in graph.initializer}
        inputs = [i for i in graph.input if i.name not in initializers]
        assert [i.name for i in inputs] == [input_name]
        assert [o.name for o in graph.output] == [first, second]
        dims = {
            t.name: [bool(d.dim_param) for d in t.type.tensor_type.shape.dim]
            for t in [*inputs, *graph.output]
        }
        assert dims[input_name][:2] == [True, True]
        assert dims[first][0] and not any(dims[first][1:])
        assert dims[second][:2] == [True, True]
        assert (
            "the batch axis of every tensor and the sequence axis of the input and "
            "`attention_weights` are dynamic" in text
        )
        (ok, failed, unsupported) = re.findall(
            r"returns (\d+) on success, (\d+) when verification fails and (\d+) for "
            r"an unsupported checkpoint",
            text,
        )[0]
        returned = {
            n.value.value
            for n in ast.walk(
                next(
                    f
                    for f in ast.parse((ROOT / "export/to_onnx.py").read_text()).body
                    if isinstance(f, ast.FunctionDef) and f.name == "main"
                )
            )
            if isinstance(n, ast.Return) and isinstance(n.value, ast.Constant)
        }
        assert returned == {int(ok), int(failed), int(unsupported)}
        old = tmp_path / "pre_v1.pt"
        torch.save({"model_state_dict": {}}, old)
        assert main(
            ["--checkpoint", str(old), "--output", str(tmp_path / "x.onnx")]
        ) == int(unsupported)

    def test_onnx_consumer_paragraph_tells_the_truth_about_released_inputs(self):
        # The exported model was fitted to released inputs. Under a privacy block
        # volume is released in log1p units and the generalization classes are
        # fitted on train, and the checkpoint stores neither: v1.0 ships no way to
        # prepare new private inputs for it.
        from train.checkpoint import CheckpointMeta

        text = _flat(self.ARCH.read_text())
        assert "privacy release, then signals, then" in text
        assert "`(x - meta.scaler_center) / meta.scaler_scale`" in text
        assert "`volume` is in log1p units, clipped to the block's `bounds`" in text
        assert "the checkpoint does not store them" in text
        assert "no way to prepare new private inputs" in text
        assert "needs only the signals and the scaler" in text
        fields = {f.name for f in dataclasses.fields(CheckpointMeta)}
        assert not {f for f in fields if "general" in f or f == "privacy"}
        for path in sorted((ROOT / "config").glob("*.yaml")):
            privacy = DignityConfig.from_yaml(str(path)).privacy
            if privacy is not None:
                assert privacy.features["volume"].transform == "log1p", path.name

    def test_a_privacy_trained_scaler_is_fitted_in_log1p_units_of_volume(self):
        from train.loop import prepare_data

        template = DignityConfig.from_yaml(str(ROOT / "config/train_risk.yaml"))
        config = DignityConfig.from_yaml(str(ROOT / "config/smoke.yaml"))
        privacy = dataclasses.replace(template.privacy, deterministic_rng=True)
        config = dataclasses.replace(
            config,
            privacy=privacy,
            data=dataclasses.replace(config.data, features=template.data.features),
        )
        frame, split = prepare_data(config, privacy_seed=0)
        centre = split.scaler_center[split.feature_names.index("volume")]
        raw_median = float(np.median(frame["volume"]))
        assert abs(centre - np.log1p(raw_median)) < abs(centre - raw_median)
        assert privacy.features["volume"].bounds[1] < raw_median

    def test_doc_quotes_no_metric_number_and_has_no_results_zone(self):
        text = self.ARCH.read_text()
        assert "<!--" not in text
        prose = strip_code(text)
        metric = r"\b(AUROC|PR-AUC|recall|Brier|prevalence|precision|cost)\b"
        assert re.search(metric + r"[^.\n]{0,40}\b\d\.\d+", prose, re.I) is None
        assert numbers_problems([self.ARCH]) == []

    def test_every_bare_name_the_doc_backticks_is_defined_in_the_repo(self):
        assert missing_identifiers(self.ARCH.read_text()) == []

    def test_a_renamed_name_in_the_doc_is_reported(self):
        text = self.ARCH.read_text()
        renamed = text.replace("`_fit_scaler`", "`_fit_nothing`")
        assert missing_identifiers(renamed) == ["_fit_nothing"]

    def test_a_dotted_chain_is_resolved_as_a_chain_not_part_by_part(self):
        # `costs` is a field of PolicyConfig, not of SplitData
        assert missing_identifiers("`SplitData.costs`") == ["SplitData.costs"]
        assert missing_identifiers("`SplitData.epsilon_spent`") == []
        assert missing_identifiers("`SignalProcessor.price_momentum`") == []

    def test_a_renamed_config_key_in_the_doc_is_reported(self, tmp_path):
        doc = tmp_path / "doc.md"
        doc.write_text(
            "`model.dropout` `policy.review_budget` "
            "`model.nope_dropout` `policy.review_nope`"
        )
        assert missing_symbols([doc]) == [
            f"{doc}: model.nope_dropout",
            f"{doc}: policy.review_nope",
        ]
        assert missing_symbols([self.ARCH]) == []


class TestBaseConfigDoc:
    """config/base.yaml is the configuration reference: every key, live or in the
    commented blocks, carries a comment of at least three words, the load-bearing
    comments are pinned to the code, and the file still loads to the shipped
    values."""

    BASE = ROOT / "config/base.yaml"
    SHIPPED_FEATURES = [
        "volume",
        "fee_rate",
        "tx_count",
        "volatility",
        "price_change",
        "momentum",
        "directional_change",
        "volume_volatility",
        "regime",
    ]

    def test_every_config_field_is_documented_in_base_yaml(self):
        raw = yaml.safe_load(self.BASE.read_text())
        # data.synthetic stays commented out on purpose (a live block would make
        # every run that sets only part of it inherit the whole thing), so its
        # fields are checked against the commented block instead
        assert set(raw) == {f.name for f in dataclasses.fields(DignityConfig)} - {
            "privacy"
        }
        sections = {
            "model": ModelConfig,
            "data": DataConfig,
            "train": TrainConfig,
            "policy": PolicyConfig,
            "eval": EvalConfig,
        }
        for section, cls in sections.items():
            expected = {f.name for f in dataclasses.fields(cls)}
            if section == "data":
                expected -= {"synthetic"}
            assert expected <= set(raw[section]), section
        text = self.BASE.read_text()
        block = text[text.index("# Event-record generator") : text.index("\ntrain:")]
        commented = set(re.findall(r"^#\s+(\w+):", block, re.M))
        wanted = {f.name for f in dataclasses.fields(SyntheticConfig)}
        assert wanted <= commented, wanted - commented

    @staticmethod
    def _words(lines: list[str], i: int) -> int:
        """Words of the comment on line i plus the comment lines right above it."""
        words = lines[i].partition("#")[2].split()
        j = i - 1
        while j >= 0 and lines[j].lstrip().startswith("#"):
            words += lines[j].split()[1:]
            j -= 1
        return len(words)

    def test_every_key_in_base_yaml_carries_a_comment_of_three_words(self):
        # A comment can still say the wrong thing: the load-bearing ones are
        # pinned below, and the rest are reviewed against the code with the file.
        lines = self.BASE.read_text().splitlines()
        short = [
            line
            for i, line in enumerate(lines)
            if re.match(r"\s*[a-z_]+:", line) and self._words(lines, i) < 3
        ]
        assert short == []

    def test_every_key_of_the_commented_synthetic_block_carries_a_comment(self):
        block = self._synthetic_block()
        text = "\n".join(block)
        for field in dataclasses.fields(SyntheticConfig):
            chunk = re.search(
                rf"^ +{field.name}:(.*?)(?=^ {{4}}\w+:|\Z)", text, re.M | re.S
            )
            assert chunk is not None, field.name
            comment = " ".join(
                line.partition(" #")[2] for line in chunk.group(0).splitlines()
            )
            assert len(comment.split()) >= 3, field.name

    def _synthetic_block(self) -> list[str]:
        lines = self.BASE.read_text().splitlines()
        block = []
        for line in lines[lines.index("#   synthetic:") :]:
            if not line.startswith("#   "):
                break
            block.append(line[2:])
        return block

    def test_commented_synthetic_block_holds_the_generator_defaults(self):
        # the header says "every field at default": the values must be, too
        shown = yaml.safe_load("\n".join(self._synthetic_block()))
        default = json.loads(json.dumps(dataclasses.asdict(SyntheticConfig())))
        assert shown == {"synthetic": default}

    def test_load_bearing_comments_say_what_the_code_does(self):
        lines = self.BASE.read_text().splitlines()
        pinned = {
            "input_size": "IGNORED",
            "source": 'only "synthetic" in v1.0',
            "label_overlap": "> this",
            "use_amp": "takes effect only on CUDA",
            "eval_seed": "never tune on it",
            "severity_range": "drawn per anomaly",
            "epsilons": "n_train_seeds risk models per value, plus a no-privacy",
        }
        for key, phrase in pinned.items():
            (line,) = [ln for ln in lines if re.match(rf"#?\s+{key}:", ln)]
            assert phrase in line, key
        assert EvalConfig().eval_seed == 2026

    def test_base_yaml_loads_to_the_shipped_defaults(self):
        loaded = DignityConfig.from_yaml(str(self.BASE))
        default = DignityConfig()
        # data.features is excluded by name: the shipped list is pinned in
        # test_shipped_feature_list_resolves_to_seven_features and differs from
        # the DataConfig default, which names entropy and lacks volume_volatility
        loaded = dataclasses.replace(
            loaded,
            data=dataclasses.replace(loaded.data, features=default.data.features),
        )
        assert loaded == default

    def test_shipped_feature_list_resolves_to_seven_features(self):
        from train.loop import prepare_data

        config = DignityConfig.from_yaml(str(self.BASE))
        assert config.data.features == self.SHIPPED_FEATURES
        synthetic = dataclasses.replace(config.data.synthetic, n_blocks=12)
        config = dataclasses.replace(
            config, data=dataclasses.replace(config.data, synthetic=synthetic)
        )
        _, split = prepare_data(config)
        assert split.feature_names == [
            f for f in self.SHIPPED_FEATURES if f not in ("price_change", "regime")
        ]

    def test_the_architecture_link_in_base_yaml_names_a_heading(self):
        (anchor,) = re.findall(r"docs/ARCHITECTURE\.md#([\w-]+)", self.BASE.read_text())
        assert anchor in heading_slugs(ROOT / "docs" / "ARCHITECTURE.md")

    def test_commented_privacy_example_is_the_shipped_risk_block(self):
        # the example must not drift from what config/train_risk.yaml runs with
        lines = self.BASE.read_text().splitlines()
        start = lines.index("# privacy:")
        block = []
        for line in lines[start:]:
            if not (line == "# privacy:" or line.startswith("#   ")):
                break
            block.append(line[2:])
        example = yaml.safe_load("\n".join(block))["privacy"]
        shipped = yaml.safe_load((ROOT / "config/train_risk.yaml").read_text())
        assert example == shipped["privacy"]
        PrivacyConfig(**example)


# Trading framing: Dignity is a risk project (detection, early warning, triage).
# Matching is case-insensitive and word-bounded, so an identifier such as
# `wash_trading` (underscores are word characters) is never a hit by itself.
FRAMING_WORDS = [
    "trade",
    "trades",
    "traded",
    "trader",
    "traders",
    "trading",
    "buy",
    "buys",
    "buying",
    "bought",
    "sell",
    "sells",
    "selling",
    "sold",
    "investor",
    "investors",
    "investing",
    "investment",
    "portfolio",
    "portfolios",
    "profit",
    "profits",
    "profitable",
    "pnl",
    "p&l",
    "alpha",
    "quant",
    "quants",
]
FRAMING_PHRASES = (
    "go long",
    "going long",
    "go short",
    "going short",
    "long position",
    "short position",
    "quantitative trading",
    "trading strategy",
    "trading agent",
    "price move",
    "price moves",
    "price movement",
    "price target",
    "next price",
    "predict the price",
    "predicts the price",
    "predict prices",
    "price prediction",
    "price forecast",
    "price forecasting",
    "high return",
    "high returns",
)
FRAMING_PATTERNS = (
    # plural "returns" qualified by what is earned: "earns returns", "expected returns"
    r"(?:earn(?:s|ing)?|portfolio|strategy|excess|expected"
    r"|maximi[sz](?:e|es|ed|ing)|chas(?:e|es|ed|ing))\s+returns",
    # "strategy" followed within three words by a return or a profit
    r"strategy(?:\s+\w+){0,2}\s+(?:returns?|profits?)",
    r"forecast\w*\s+(?:the\s+)?price",
)
FRAMING_RE = re.compile(
    "|".join(
        [rf"\b{re.escape(w)}\b" for w in FRAMING_WORDS]
        + [
            r"\b" + r"\s+".join(map(re.escape, p.split())) + r"\b"
            for p in FRAMING_PHRASES
        ]
        + [rf"\b{p}\b" for p in FRAMING_PATTERNS]
    ),
    re.I,
)
# The only forms removed before matching: anomaly types the generator injects,
# in identifier and prose form, and "trade-off". "layering" and "pump and dump"
# never match the pattern; they are listed because the maintainer named them.
FRAMING_ALLOWED = (
    re.compile(r"wash[_ -]trading", re.I),  # anomaly type, six injectors
    re.compile(r"pump[_ -]and[_ -]dump", re.I),  # anomaly type
    re.compile(r"\blayering\b", re.I),  # anomaly type
    re.compile(r"\btrade-?offs?\b", re.I),  # a compromise, not a transaction
)


def framing_hits(text: str) -> list[str]:
    """Trading-framed phrases of `text`, after the allowed forms are removed.

    A tripwire over the fixed vocabulary above, not a proof that a text is free
    of such framing: a denylist cannot be complete, and review does not extend it.
    """
    for allowed in FRAMING_ALLOWED:
        text = allowed.sub(" ", text)
    return [m.group(0) for m in FRAMING_RE.finditer(text)]


FRAMING_REJECTED = {
    "trade": "Traders place a trade each day.",
    "trades": "The policy trades often.",
    "traded": "The asset was traded yesterday.",
    "trader": "The trader reads the score.",
    "traders": "Traders read the score.",
    "trading": "A trading desk reads the score.",
    "buy": "The policy tells you to buy.",
    "buys": "The policy buys early.",
    "buying": "The policy is buying early.",
    "sell": "The policy tells you to sell.",
    "sells": "The policy sells early.",
    "selling": "The policy is selling early.",
    "go long": "Go long when risk is low.",
    "going long": "Going long when risk is low.",
    "go short": "Go short on the asset.",
    "going short": "Going short on the asset.",
    "long position": "Hold a long position until the alert.",
    "short position": "Hold a short position until the alert.",
    "investor": "An investor reads the score.",
    "investors": "Investors benefit.",
    "investing": "Investing in the model pays.",
    "investment": "The investment grows.",
    "portfolio": "The portfolio is rebalanced.",
    "profit": "The policy makes a profit.",
    "profits": "The policy keeps its profits.",
    "profitable": "The policy is profitable.",
    "pnl": "Maximizes PnL.",
    "p&l": "Reports the P&L.",
    "alpha": "The model has alpha.",
    "quant": "A quant wrote it.",
    "quants": "Quants wrote it.",
    "quantitative trading": "A quantitative trading model.",
    "price move": "Anticipates the price move.",
    "price moves": "Anticipates price moves.",
    "price movement": "Anticipates price movement.",
    "price target": "Sets a price target.",
    "next price": "Predicts the next price.",
    "predict the price": "Models predict the price.",
    "predicts the price": "The model predicts the price.",
    "predict prices": "Models predict prices.",
    "price prediction": "A price prediction head.",
    "price forecast": "A price forecast head.",
    "price forecasting": "A price forecasting head.",
    "high return": "The strategy earns a high return.",
    "earn returns": "Users earn returns.",
    "earns returns": "The model earns returns.",
    "earning returns": "Earning returns is not the goal.",
    "portfolio returns": "Reports portfolio returns.",
    "strategy returns": "Reports strategy returns.",
    "excess returns": "Reports excess returns.",
    "expected returns": "Reports expected returns.",
    "trading strategy": "A trading strategy built on the score.",
    "strategy return": "The strategy return is reported.",
    "strategy returns, three words": "The strategy used here returns nothing.",
    "strategy profit": "The strategy profit is reported.",
    "strategy profits, three words": "The strategy we ship profits nobody.",
}
FRAMING_PASSING = {
    "trade-off": "There is a trade-off between recall and review load.",
    "trade-offs": "The trade-offs are listed below.",
    "tradeoff": "That is a tradeoff.",
    "tradeoffs": "Those are tradeoffs.",
    "wash_trading": "The `wash_trading` anomaly type.",
    "wash trading": "Wash trading is an injected anomaly.",
    "pump_and_dump": "The `pump_and_dump` anomaly type.",
    "pump and dump": "A pump and dump is an injected anomaly.",
    "layering": "Layering is an injected anomaly.",
    "price": "The `price` column holds the raw series.",
    "price column": "The price column is raw input.",
    "returns a tuple": "The head returns a tuple.",
    "generator returns": "The generator returns a frame.",
    "returns on its own": "The call returns on its own.",
    "asset returns": "Each asset returns a dict.",
    "log returns": "The log returns a float.",
    "market returns": "The market returns nothing here.",
    "generates returns": "The sampler generates returns a list.",
    "stock returns": "The stock returns no rows.",
    "expected return type": "The expected return type is a tuple.",
    "the function returns": "Check what the function returns.",
    "forecast head": "The forecast head scores early warning.",
    "forecast labels": "The forecast labels come from the event record.",
    "forecast horizon": "The forecast horizon is in windows.",
    "cost": "The cost table is fixed.",
    "expected cost": "The loss is the expected cost.",
    "review budget": "The review budget caps alerts.",
    "risk": "The risk score is a probability.",
    "triage": "The policy is a triage rule.",
}


def config_comments(path: Path) -> str:
    """The comment text of a YAML file, live and commented-out keys alike."""
    return "\n".join(
        line[line.index("#") :] for line in path.read_text().splitlines() if "#" in line
    )


class TestFramingGuard:
    """Dignity is a risk project; no public text frames it as trading, price
    forecasting or quant work (maintainer rule)."""

    def test_public_docs_and_config_comments_do_not_frame_the_project_as_trading(self):
        hits = {
            str(path.relative_to(ROOT)): framing_hits(text)
            for path, text in [
                *((d, d.read_text()) for d in _public_docs()),
                *(
                    (c, config_comments(c))
                    for c in sorted((ROOT / "config").glob("*.yaml"))
                ),
            ]
            if framing_hits(text)
        }
        assert hits == {}

    def test_a_trading_sentence_added_to_the_architecture_doc_is_reported(self):
        text = (ROOT / "docs" / "ARCHITECTURE.md").read_text()
        for sentence in (
            "Traders use the forecast to anticipate the next price move.",
            "The policy decides whether to buy or sell.",
        ):
            assert framing_hits(text) == []
            assert framing_hits(text + "\n" + sentence) != [], sentence

    def test_a_trading_header_in_a_config_comment_is_reported(self):
        base = ROOT / "config" / "base.yaml"
        comments = config_comments(base)
        assert comments.startswith("# Base Dignity Configuration")
        framed = comments.replace(
            "# Base Dignity Configuration", "# Trading-agent model architecture", 1
        )
        assert framing_hits(comments) == []
        assert framing_hits(framed) == ["Trading"]

    @pytest.mark.parametrize(
        "phrase",
        [
            "a quant strategy",
            "the alpha of the model",
            "maximise returns",
            "a portfolio of assets",
            "chasing profit",
            "buying and selling",
            "a price target",
            "price movement",
            "the trader",
        ],
    )
    def test_each_forbidden_trading_term_is_reported(self, phrase):
        assert framing_hits(phrase) != []

    def test_anomaly_type_names_are_allowed_but_do_not_mask_trading_framing(self):
        allowed = (
            "`wash_trading`, wash trading, wash-trading, `pump_and_dump`, "
            "pump and dump, `layering` and the generator's `price` column"
        )
        assert framing_hits(allowed) == []
        assert framing_hits(allowed + " for traders") == ["traders"]
        assert framing_hits("wash trading helps trading") == ["trading"]

    def test_the_docs_do_use_the_allowlisted_anomaly_names(self):
        # the allowlist is not dead weight: the docs name the anomaly types
        text = (ROOT / "docs" / "ARCHITECTURE.md").read_text()
        assert "wash_trading" in text and "pump_and_dump" in text

    @staticmethod
    def _in_markdown(sentence: str, tmp_path: Path) -> list[str]:
        doc = tmp_path / "doc.md"
        doc.write_text(f"# Title\n\n{sentence}\n")
        return framing_hits(doc.read_text())

    @staticmethod
    def _in_yaml_comment(sentence: str, tmp_path: Path) -> list[str]:
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(f"key: 1  # {sentence}\n")
        return framing_hits(config_comments(cfg))

    @pytest.mark.parametrize("term", list(FRAMING_REJECTED))
    def test_the_fixed_vocabulary_is_reported_in_docs_and_config_comments(
        self, term, tmp_path
    ):
        sentence = FRAMING_REJECTED[term]
        assert self._in_markdown(sentence, tmp_path) != [], term
        assert self._in_yaml_comment(sentence, tmp_path) != [], term

    @pytest.mark.parametrize("term", list(FRAMING_REJECTED))
    def test_the_fixed_vocabulary_is_reported_in_any_letter_case(self, term):
        assert framing_hits(FRAMING_REJECTED[term].upper()) != [], term

    @pytest.mark.parametrize("form", list(FRAMING_PASSING))
    def test_the_allowed_forms_are_not_reported(self, form, tmp_path):
        sentence = FRAMING_PASSING[form]
        assert self._in_markdown(sentence, tmp_path) == [], form
        assert self._in_yaml_comment(sentence, tmp_path) == [], form

    def test_a_noun_subject_before_the_function_verb_returns_passes_but_the_ruled_prefixes_do_not(
        self,
    ):
        assert framing_hits("The sampler returns a list.") == []
        assert framing_hits("The log returns a float.") == []
        for phrase in (
            "earns returns",
            "expected returns",
            "portfolio returns",
            "excess returns",
            "strategy returns",
            "maximise returns",
            "chasing returns",
        ):
            assert framing_hits(f"It reports {phrase} daily.") != [], phrase

    def test_a_phrase_wrapped_across_lines_is_still_reported(self):
        assert framing_hits("Go\nlong when risk is low.") != []


class TestBaseConfigRunDirectories:
    def test_train_header_names_the_directories_each_runner_nests_under_it(self):
        comments = _flat(
            config_comments(ROOT / "config" / "base.yaml").replace("#", "")
        )
        assert "eval.run nests eval_<seed>/<task>/train_seed_<s> under it" in comments
        assert (
            "the privacy sweep nests privacy_sweep/eps_<e>/train_seed_<s> and "
            "privacy_sweep/reference/train_seed_<s>" in comments
        )
        run = (ROOT / "eval" / "run.py").read_text()
        sweep = (ROOT / "eval" / "privacy_sweep.py").read_text()
        assert 'f"eval_{seed_name}"' in run and 'f"train_seed_{s}"' in run
        assert "config.model.task" in run
        assert '"privacy_sweep"' in sweep
        assert 'f"eps_{epsilon}"' in sweep and '"reference"' in sweep
        assert 'f"train_seed_{config.seed + r}"' in sweep


# --- docs/RESULTS.md (M6.6) ---------------------------------------------------

RESULTS_MD = ROOT / "docs" / "RESULTS.md"
ZONE_BLOCK_RE = re.compile(r"<!--results-start-->(.*?)<!--results-end-->", re.DOTALL)
TABLE_SEPARATOR_RE = re.compile(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?")
SOURCE_CELL_RE = re.compile(
    r"`(python -m eval\.run[^`]*)` -> `results/v1\.0/(\w+)\.json`"
)
RESULTS_HEADERS = (
    "Metric | Model | Best baseline | Source",
    "Type | Recall [CI] | n_windows | Source",
)
ZONE_SECTIONS = (
    "## Risk",
    "## Forecast",
    "## Policy",
    "## Privacy curve",
    "## Per-type results",
)


def results_zones(text: str) -> list[str]:
    """The text of every results zone, with the zone tags of code left out."""
    live = strip_code(text)
    return [text[m.start(1) : m.end(1)] for m in ZONE_BLOCK_RE.finditer(live)]


def section_of(text: str, heading: str) -> str:
    """The text from `heading` up to the next level-two heading (or the end)."""
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start : len(text) if end == -1 else end]


def zone_tables(zone: str) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """(header, data rows) of every table in a zone, and the pipe lines that
    belong to no table. A header is the row that sits directly above a separator
    row; the first body row below it is never exempt."""
    lines = zone.splitlines()
    tables, claimed, i = [], set(), 0
    while i < len(lines) - 1:
        if lines[i].startswith("|") and TABLE_SEPARATOR_RE.fullmatch(
            lines[i + 1].strip()
        ):
            rows, j = [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append(lines[j])
                j += 1
            tables.append((lines[i], rows))
            claimed.update(range(i, j))
            i = j
        else:
            i += 1
    loose = [
        line
        for k, line in enumerate(lines)
        if line.startswith("|") and k not in claimed
    ]
    return tables, loose


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _main_parser(module):
    """The argparse parser of module.main, captured instead of hard-coded."""
    import argparse
    import contextlib
    from unittest import mock

    class CapturedError(Exception):
        pass

    captured = []

    def spy(self, args=None, namespace=None):
        captured.append(self)
        raise CapturedError

    with (
        mock.patch.object(argparse.ArgumentParser, "parse_args", spy),
        contextlib.suppress(CapturedError),
    ):
        module.main([])
    return captured[0]


def eval_run_parser():
    from eval import run as eval_run

    return _main_parser(eval_run)


def gates_parser():
    from eval import gates

    return _main_parser(gates)


def command_problems(command: str, file_name: str, root: Path = ROOT) -> list[str]:
    """Why a Source cell's command is not one eval.run accepts and that writes
    the cited results file."""
    import contextlib
    import io

    argv = shlex.split(command)
    if argv[:3] != ["python", "-m", "eval.run"]:
        return [f"{command}: does not start with python -m eval.run"]
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            args = eval_run_parser().parse_args(argv[3:])
    except SystemExit:
        return [f"{command}: eval.run rejects it ({err.getvalue().strip()[-80:]})"]
    problems = []
    if args.config is None or not (root / args.config).is_file():
        problems.append(f"{command}: --config {args.config} is not a file")
    tasks = {t.strip() for t in args.tasks.split(",")}
    for task in sorted(tasks - {"risk", "forecast", "policy"}):
        problems.append(f"{command}: eval.run rejects the unknown task {task!r}")
    if file_name in ("risk", "forecast", "policy") and file_name not in tasks:
        problems.append(f"{command}: --tasks does not include {file_name}")
    if file_name == "privacy" and not args.privacy_sweep:
        problems.append(f"{command}: privacy.json needs --privacy-sweep")
    return problems


def zone_problems(zone: str, root: Path = ROOT) -> list[str]:
    """Rule violations of every data row of every table in one results zone."""
    tables, loose = zone_tables(zone)
    problems = [f"pipe line outside any table: {line}" for line in loose]
    for header, rows in tables:
        if " | ".join(_cells(header)) not in RESULTS_HEADERS:
            problems.append(f"unknown table header: {header}")
        for row in rows:
            source = SOURCE_CELL_RE.fullmatch(_cells(row)[-1])
            if not source:
                problems.append(f"no command and results file in the last cell: {row}")
                continue
            command, file_name = source.groups()
            problems += command_problems(command, file_name, root)
            markers = [m[1:] for m in NUM_RE.findall(row)]
            if len(markers) != row.count("<!--num"):
                problems.append(f"a marker the grammar does not read: {row}")
            for name, dotted, _ in markers:
                if name != file_name:
                    problems.append(
                        f"{dotted} reads {name}.json, row cites {file_name}"
                    )
            indexes = {
                i
                for _, dotted, _ in markers
                for i in re.findall(r"\bpoints\.(\d+)\.", dotted)
            }
            if len(indexes) > 1:
                problems.append(f"privacy row mixes points {sorted(indexes)}: {row}")
    return problems


def source_seed_problems(text: str, results_dir: Path = RESULTS_DIR) -> list[str]:
    """Every Source cell's --seed word is tuning while risk.json is on the tuning
    seed and eval after."""
    expected = "tuning" if results_on_tuning_seed(results_dir) else "eval"
    wrong = []
    for command, _ in SOURCE_CELL_RE.findall(text):
        argv = shlex.split(command)
        word = argv[argv.index("--seed") + 1] if "--seed" in argv else None
        if word != expected:
            wrong.append(command)
    return [
        f"{wrong.count(command)} Source cell(s) carry `{command}`: its --seed "
        f"should be {expected}"
        for command in dict.fromkeys(wrong)
    ]


def _results_json(name: str, results_dir: Path = RESULTS_DIR) -> dict:
    return json.loads((results_dir / f"{name}.json").read_text())


def results_on_tuning_seed(results_dir: Path = RESULTS_DIR) -> bool:
    """Whether the committed results are tuning-seed results, decided from the seed
    risk.json records against core.config.EvalConfig. Every sentence that holds
    only while this is true is checked in both directions: required while it is,
    forbidden once the seed is the evaluation seed."""
    return _results_json("risk", results_dir)["seed"] == EvalConfig().tuning_seed


OutcomePhrases = collections.namedtuple("OutcomePhrases", "required forbidden headline")


def _where(flags: list[bool]) -> str:
    """none / all / some of the horizons (or seeds) a condition holds at."""
    return "none" if not any(flags) else "all" if all(flags) else "some"


# The words RESULTS.md uses for "at how many horizons", defined without a gap: none
# is 0, some is 1 up to fewer than half, half is exactly half, most is more than half
# but not all, every is all. The Forecast section states the rule in these words.
HORIZON_PHRASES = {
    "none": "at no horizon",
    "some": "at some horizons, fewer than half of them",
    "half": "at exactly half of the horizons",
    "most": "at most horizons",
    "every": "at every horizon",
}
HORIZON_WORDS_RULE = (
    "none means no horizon, some means at least one but fewer than half, half means "
    "exactly half, most means more than half but not all, and every means all of them"
)


def horizon_word(n: int, total: int) -> str:
    """The word for n of `total` horizons: every case is exactly one word."""
    if n == 0:
        return "none"
    if n == total:
        return "every"
    if 2 * n == total:
        return "half"
    return "most" if 2 * n > total else "some"


def outcome_phrases(results_dir: Path = RESULTS_DIR) -> dict[str, OutcomePhrases]:
    """Per task, the outcome phrases the committed JSON supports, the ones it
    contradicts and the headline subset the opening paragraph must repeat, all
    recomputed from the numbers (never read from the doc)."""
    out = {}

    risk = _results_json("risk", results_dir)
    required, forbidden = [], []
    for level, model, baselines in (
        ("window", risk["metrics"]["pr_auc"], ("majority", "window_logistic")),
        ("block", risk["metrics"]["block_level"]["pr_auc"], ("block_tree",)),
    ):
        best = max(baselines, key=lambda b: risk["baselines"][b]["pr_auc"]["value"])
        base = risk["baselines"][best]["pr_auc"]["value"]
        passed = model["value"] > base and model["ci_low"] > base
        yes = f"the {level} gate passes against `{best}`"
        no = f"the {level} gate does not pass against `{best}`"
        required.append(yes if passed else no)
        forbidden.append(no if passed else yes)
    out["risk"] = OutcomePhrases(required, forbidden, list(required))

    forecast = _results_json("forecast", results_dir)
    metrics = forecast["metrics"]
    model = metrics["pr_auc_by_horizon"]
    base = forecast["baselines"]["risk_score_forecast"]["pr_auc_by_horizon"]
    chance = metrics["prevalence_by_horizon"]
    cleared = [m["ci_low"] > b["value"] for m, b in zip(model, base, strict=True)]
    below = [m["value"] < b["value"] for m, b in zip(model, base, strict=True)]
    where = {
        "none": "at no horizon",
        "all": "at every horizon",
        "some": "at some horizons and not at others",
    }
    key = _where(cleared)
    required = [
        f"the model's PR-AUC interval clears `risk_score_forecast` {where[key]}"
    ]
    forbidden = [
        f"the model's PR-AUC interval clears `risk_score_forecast` {w}"
        for k, w in where.items()
        if k != key
    ]
    headline = [required[0]]
    key = _where(below)
    template = "its point estimate is below the baseline's {}"
    required.append(template.format(where[key]))
    forbidden += [template.format(w) for k, w in where.items() if k != key]
    headline.append(required[-1])
    # the chance line, in both directions
    contains = [
        m["ci_low"] <= c <= m["ci_high"] for m, c in zip(model, chance, strict=True)
    ]
    above = [b["ci_low"] > c for b, c in zip(base, chance, strict=True)]
    for template, flags in (
        ("the model's PR-AUC interval contains the prevalence {}", contains),
        ("the baseline's PR-AUC interval lies wholly above the prevalence {}", above),
    ):
        key = _where(flags)
        required.append(template.format(where[key]))
        forbidden += [template.format(w) for k, w in where.items() if k != key]
        headline.append(required[-1])
    # at how many horizons the baseline's lower bound is above the model's estimate
    n_ahead = sum(b["ci_low"] > m["value"] for m, b in zip(model, base, strict=True))
    key = horizon_word(n_ahead, len(model))
    template = "the baseline's lower bound is above the model's point estimate {}"
    required.append(template.format(HORIZON_PHRASES[key]))
    forbidden += [template.format(w) for k, w in HORIZON_PHRASES.items() if k != key]
    headline.append(required[-1])
    required += [
        "On the headline seed, at the alarm threshold, the model warns ahead of",
        "so on the headline seed the model's median lead time is",
        HORIZON_WORDS_RULE,
    ]
    amounts = {
        "none": "warns ahead of no onset",
        "all": "warns ahead of every onset",
        "some": "warns ahead of some onsets",
    }
    for who, lead in (
        ("the model", metrics["lead_time"]),
        (
            "`risk_score_forecast`",
            forecast["baselines"]["risk_score_forecast"]["lead_time"],
        ),
    ):
        missed, onsets = lead["n_missed"], lead["n_onsets"]
        key = "none" if missed == onsets else "all" if missed == 0 else "some"
        required.append(f"{who} {amounts[key]}")
        forbidden += [f"{who} {a}" for k, a in amounts.items() if k != key]
        if who == "the model":
            headline.append(required[-1])
    median = "the model's median lead time is n/a"
    (required if metrics["lead_time"]["median"] is None else forbidden).append(median)
    # the lead time of every training seed, not only the headline one
    values = metrics["seed_spread"]["lead_time_median"]["values"]
    seeds = {
        "none": "on none of the training seeds",
        "all": "on every one of the training seeds",
        "some": "on some of the training seeds",
    }
    key = _where([v is not None for v in values])
    template = "the model warns ahead of at least one onset {}"
    required.append(template.format(seeds[key]))
    forbidden += [template.format(w) for k, w in seeds.items() if k != key]
    # the n/a explanation is true only while some seed's median is null
    explanation = "(n/a: no onset warned ahead on that seed)"
    (required if None in values else forbidden).append(explanation)
    # the file records the medians, and counts per seed only if the JSON has them
    spread_text = json.dumps(metrics["seed_spread"])
    counted = "n_onsets" in spread_text or "n_missed" in spread_text
    (forbidden if counted else required).append("records no per-seed onset counts")
    out["forecast"] = OutcomePhrases(required, forbidden, headline)

    policy = _results_json("policy", results_dir)
    m = policy["metrics"]["expected_cost"]
    b = policy["baselines"]["threshold_policy"]["expected_cost"]
    verdicts = {
        "clears": "the triage head's expected cost clears the threshold policy's",
        "lower": "the triage head's expected cost is lower than the threshold "
        "policy's, but its interval does not clear it",
        "higher": "the triage head's expected cost is higher than the threshold "
        "policy's",
    }
    key = (
        "clears"
        if m["ci_high"] < b["value"]
        else ("lower" if m["value"] < b["value"] else "higher")
    )
    required = [verdicts[key]]
    forbidden = [v for k, v in verdicts.items() if k != key]
    conclusion = "so the triage head beats the threshold policy"
    (required if key == "clears" else forbidden).append(conclusion)
    (forbidden if key == "clears" else required).append(
        "so the triage head does not beat the threshold policy"
    )
    headline = list(required)
    # the rule the verdict applies, as the doc must state it
    required.append(
        "the upper end of its interval lies below the threshold policy's expected cost"
    )
    overlap = m["ci_low"] <= b["ci_high"] and b["ci_low"] <= m["ci_high"]
    yes, no = "the two intervals overlap", "the two intervals do not overlap"
    required.append(yes if overlap else no)
    forbidden.append(no if overlap else yes)
    spread = policy["metrics"]["seed_spread"]
    pairs = list(
        zip(
            spread["expected_cost"]["values"],
            spread["baseline_expected_cost"]["values"],
            strict=True,
        )
    )
    per_seed = {
        "more": "on every training seed the triage head costs more than the "
        "threshold policy",
        "less": "on every training seed the triage head costs less than the "
        "threshold policy",
        "mixed": "the triage head costs more than the threshold policy on some "
        "training seeds and less on others",
    }
    if all(a > c for a, c in pairs):
        key = "more"
    elif all(a < c for a, c in pairs):
        key = "less"
    else:
        key = "mixed"
    required.append(per_seed[key])
    forbidden += [v for k, v in per_seed.items() if k != key]
    out["policy"] = OutcomePhrases(required, forbidden, headline)

    privacy = _results_json("privacy", results_dir)["metrics"]
    prevalence = privacy["reference"]["risk"]["prevalence"]["value"]
    points = sorted(privacy["points"], key=lambda p: p["epsilon"])
    intervals = [p["risk"]["pr_auc"] for p in points]
    split = (
        "not distinguishable from chance at every grid value except the two "
        "largest, where it is above chance"
    )
    holds = all(i["ci_low"] <= prevalence <= i["ci_high"] for i in intervals[:-2])
    holds = holds and all(i["ci_low"] > prevalence for i in intervals[-2:])
    required = [split if holds else "(the privacy outcome no longer holds: rewrite it)"]
    forbidden = ["above chance at every grid value"] + ([] if holds else [split])
    out["privacy"] = OutcomePhrases(required, forbidden, list(required))
    return out


def phrase_problems(text: str, required: list[str], forbidden: list[str]) -> list[str]:
    flat = _flat(text)
    return [
        f"missing, so write a sentence that says: {p}"
        for p in required
        if p not in flat
    ] + [
        f"contradicts the results, so rewrite the sentence that says: {p}"
        for p in forbidden
        if p in flat
    ]


# A sentence that holds only while the committed results are tuning-seed results.
# Each is required while results_on_tuning_seed() is true and forbidden after, so
# the final evaluation fails on exactly the sentences a person must rewrite.
FOUR_RESULTS = (
    "`results/v1.0/risk.json`, `results/v1.0/forecast.json`, "
    "`results/v1.0/policy.json` and `results/v1.0/privacy.json`"
)
TUNING_ONLY_SENTENCES = (
    f"The numbers of the four `eval.run` results ({FOUR_RESULTS}) are tuning-seed "
    "test scores, and the tuning seed's test split was read during development, "
    "so they are not an unbiased estimate of the final evaluation, which "
    "replaces them.",
    "The numbers of the four `eval.run` results are tuning-seed scores.",
    "On the tuning seed, the shipped allocation leaves the risk model's PR-AUC "
    "not distinguishable from chance",
)
# Wording that names the tuning seed as the source of the numbers, wherever it is.
TUNING_ONLY_WORDING = ("tuning-seed", "On the tuning seed", "not an unbiased estimate")


def seed_wording_problems(text: str, results_dir: Path = RESULTS_DIR) -> list[str]:
    flat = _flat(text)
    if results_on_tuning_seed(results_dir):
        return [
            f"missing while the results are on the tuning seed: {s}"
            for s in TUNING_ONLY_SENTENCES
            if s not in flat
        ]
    return [
        f"holds only on the tuning seed, but the results carry the evaluation "
        f"seed; rewrite it: {s}"
        for s in TUNING_ONLY_SENTENCES
        if s in flat
    ] + [
        f"names the tuning seed as the source of the numbers, but the results "
        f"carry the evaluation seed; rewrite it: {w}"
        for w in TUNING_ONLY_WORDING
        if w in flat
    ]


def eval_seed_literal_problems(text: str) -> list[str]:
    """The evaluation seed is written in code, or through a marker (which a
    refresh keeps true); an unmarked literal, or any in a zone, is reported."""
    seed = str(EvalConfig().eval_seed)
    # the seed as a whole token: a marked value such as 0.2026 or 12026 is no hit
    literal = re.compile(rf"(?<![\w.,]){seed}(?![\w]|[.,]\d)")

    problems = [
        f"a results zone names {seed} outside a marker: write it in code outside "
        "the zone, or through a marker"
        for z in results_zones(text)
        if literal.search(NUM_RE.sub("", z))
    ]
    if literal.search(NUM_RE.sub("", strip_code(text))):
        problems.append(
            f"{seed} is written outside code and outside a marker: put it in code "
            "or give it a marker such as <!--num risk.json seed d-->"
        )
    return problems


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?])\s+(?=(?:- )?[A-Zε`!\[(])")


def sentences_of(text: str) -> list[str]:
    """The whole sentences of a Markdown passage: emphasis, images and list
    dashes dropped, whitespace collapsed, split where a sentence ends."""
    flat = _flat(re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text))
    return [re.sub(r"^- ", "", s) for s in SENTENCE_SPLIT_RE.split(flat)]


def privacy_verbatim_problems(results_text: str, privacy_text: str) -> list[str]:
    """Every sentence RESULTS.md reuses from the curve section of PRIVACY.md is
    reused whole: the scope, the headline paragraph, the AUROC qualifier and the
    shipped-config sentences."""
    curve = section_of(privacy_text, "## Privacy-versus-utility curve")
    sentences = sentences_of(curve.split("\n", 1)[1])

    def at(fragment: str) -> int:
        return next(i for i, s in enumerate(sentences) if fragment in s)

    wanted = (
        sentences[at("The curve answers one question") : at("is not consumed") + 1]
        + sentences[
            at("On the tuning seed, the shipped allocation") : at(
                "they are not offered as a private setting"
            )
            + 1
        ]
        + [
            sentences[at("The statement is about PR-AUC")],
            sentences[at("The file also carries AUROC")],
            sentences[at("The shipped config is not a point on this curve")],
            sentences[at("Its ε of 1.0 is a grid value")],
        ]
    )
    section = _flat(section_of(results_text, "## Privacy curve"))
    return [f"not reused whole: {s}" for s in wanted if s not in section]


def eval_gates_command_problems(text: str) -> list[str]:
    """The Reproduce section carries the one command that writes the gate file."""
    commands = [
        line
        for line in section_of(text, "## Reproduce").splitlines()
        if line.startswith("python -m eval.gates")
    ]
    if len(commands) != 1:
        return [f"expected one python -m eval.gates command, found {len(commands)}"]
    import contextlib
    import io

    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            args = gates_parser().parse_args(shlex.split(commands[0])[3:])
    except SystemExit:
        return [f"{commands[0]}: eval.gates rejects it"]
    problems = []
    if args.seed != EvalConfig().tuning_seed:
        problems.append(f"{commands[0]}: --seed is not the tuning seed")
    if args.out != Path("results/v1.0/data_gate_tuning.json"):
        problems.append(f"{commands[0]}: --out is not the gate file")
    return problems


def _triple(base: str) -> list[str]:
    return [f"{base}.value", f"{base}.ci_low", f"{base}.ci_high"]


def _expected_cells(heading: str, cells: list[str]):
    """(model paths, baseline name, baseline paths) a row's label promises; the
    paths are the dotted paths of the markers in the Model and baseline cells."""
    label = NUM_RE.sub("", cells[0]).strip()
    if heading == "## Risk":
        found = re.fullmatch(r"`(\w+)`( per block)?", label)
        metric, block = found[1], bool(found[2])
        model = f"metrics.block_level.{metric}" if block else f"metrics.{metric}"
        name = "block_tree" if block else "window_logistic"
        return _triple(model), name, _triple(f"baselines.{name}.{metric}")
    if heading == "## Forecast":
        found = re.fullmatch(r"`(pr_auc_by_horizon\.\d+)`", label)
        name = "risk_score_forecast"
        if found:
            return (
                _triple(f"metrics.{found[1]}"),
                name,
                _triple(f"baselines.{name}.{found[1]}"),
            )
        found = re.fullmatch(r"`lead_time\.(median|n_missed|n_onsets)`", label)
        return (
            [f"metrics.lead_time.{found[1]}"],
            name,
            [f"baselines.{name}.lead_time.{found[1]}"],
        )
    if heading == "## Policy":
        metric = re.fullmatch(r"`(\w+)`", label)[1]
        name = "threshold_policy"
        return _triple(f"metrics.{metric}"), name, _triple(f"baselines.{name}.{metric}")
    if heading == "## Privacy curve":
        name = "window_logistic"
        if label == "`pr_auc` without privacy":
            return (
                _triple("metrics.reference.risk.pr_auc"),
                name,
                _triple(f"baselines.{name}.reference.pr_auc"),
            )
        if label == "`prevalence`":
            return _triple("metrics.reference.risk.prevalence"), None, []
        point = re.search(r"metrics\.points\.(\d+)\.epsilon", cells[0])
        if label == "`pr_auc` at ε" and point:
            n = point[1]
            return (
                _triple(f"metrics.points.{n}.risk.pr_auc"),
                name,
                _triple(f"baselines.{name}.points.{n}.pr_auc"),
            )
    raise AssertionError(f"{heading}: no rule for the row label {cells[0]!r}")


def row_path_problems(text: str) -> list[str]:
    """Each risk, forecast, policy, privacy and per-type row quotes the metric its
    label names: the label, the baseline's name and every marker path agree."""
    problems = []
    for heading in ("## Risk", "## Forecast", "## Policy", "## Privacy curve"):
        zone = results_zones(section_of(text, heading))[0]
        ((_, rows),) = zone_tables(zone)[0]
        for row in rows:
            cells = _cells(row)
            model, name, baseline = _expected_cells(heading, cells)
            got_model = [m[2] for m in NUM_RE.findall(cells[1])]
            got_base = [m[2] for m in NUM_RE.findall(cells[2])]
            if got_model != model:
                problems.append(f"{cells[0]}: Model cell reads {got_model}")
            if got_base != baseline:
                problems.append(f"{cells[0]}: baseline cell reads {got_base}")
            if name and not cells[2].startswith(f"`{name}`"):
                problems.append(f"{cells[0]}: baseline cell is not {name}")
            if name is None and cells[2] != "n/a":
                problems.append(f"{cells[0]}: baseline cell should be n/a")
    zone = results_zones(section_of(text, "## Per-type results"))[0]
    ((_, rows),) = zone_tables(zone)[0]
    for row in rows:
        cells = _cells(row)
        base = f"metrics.per_type.{cells[0].strip('`')}"
        if [m[2] for m in NUM_RE.findall(cells[1])] != _triple(base):
            problems.append(f"{cells[0]}: Recall cell does not read {base}")
        if [m[2] for m in NUM_RE.findall(cells[2])] != [f"{base}.n_windows"]:
            problems.append(f"{cells[0]}: n_windows cell does not read {base}")
    return problems


def opening_paragraph(text: str) -> str:
    return next(par for par in text.split("\n\n") if par.startswith("**Outcome.**"))


def _swap_cells(text: str, label_a: str, label_b: str, column: int) -> str:
    """Swap one cell between the rows labelled a and b (the last such row)."""
    lines = text.split("\n")

    def row(label):
        return [i for i, line in enumerate(lines) if line.startswith(f"| {label} |")][
            -1
        ]

    a, b = row(label_a), row(label_b)
    parts_a, parts_b = lines[a].split("|"), lines[b].split("|")
    parts_a[column + 1], parts_b[column + 1] = parts_b[column + 1], parts_a[column + 1]
    lines[a], lines[b] = "|".join(parts_a), "|".join(parts_b)
    return "\n".join(lines)


def lead_basis_problems(results_dir: Path = RESULTS_DIR) -> list[str]:
    """A median lead time rests on the scored onsets less the missed ones."""
    forecast = _results_json("forecast", results_dir)
    problems = []
    for who, lead in (
        ("model", forecast["metrics"]["lead_time"]),
        ("baseline", forecast["baselines"]["risk_score_forecast"]["lead_time"]),
    ):
        if len(lead["values"]) != lead["n_onsets"] - lead["n_missed"]:
            problems.append(f"{who}: lead_time values are not onsets less missed")
    return problems


def spread_std_problems(text: str, source: str) -> list[str]:
    """The doc says the standard deviation is the sample one; eval.run computes it
    that way."""
    problems = []
    if "ddof=1" not in source:
        problems.append("eval.run does not compute the sample standard deviation")
    if "sample standard deviation, not the population one" not in _flat(text):
        problems.append("the Protocol does not say the sample standard deviation")
    return problems


def forecast_cell_problems(text: str, results_dir: Path = RESULTS_DIR) -> list[str]:
    """No horizon's row shows a tie (the same printed number for the model and the
    baseline) where the full-precision values in forecast.json differ, whichever of
    the two is larger: a tie in print would hide the comparison the section makes."""
    forecast = _results_json("forecast", results_dir)
    pairs = zip(
        forecast["metrics"]["pr_auc_by_horizon"],
        forecast["baselines"]["risk_score_forecast"]["pr_auc_by_horizon"],
        strict=True,
    )
    exact = {n: (m["value"], b["value"]) for n, (m, b) in enumerate(pairs)}
    zone = results_zones(section_of(text, "## Forecast"))[0]
    ((_, rows),) = zone_tables(zone)[0]
    problems = []
    for row in rows:
        cells = _cells(row)
        found = re.fullmatch(r"`pr_auc_by_horizon\.(\d+)`", cells[0])
        if not found:
            continue
        model, baseline = (
            float(split_token(NUM_RE.findall(cell)[0][0])[1]) for cell in cells[1:3]
        )
        full_model, full_baseline = exact[int(found[1])]
        if full_model != full_baseline and model == baseline:
            problems.append(
                f"{cells[0]}: both cells print {model} but forecast.json has "
                f"{full_model} and {full_baseline}; raise the digits of the .Nf "
                "format in the markers of that row, then refresh"
            )
    return problems


def runbook_problems(runbook: str) -> list[str]:
    """The runbook moves the archive's files into the tree, then refreshes and
    tests, in that order, after it has unpacked the archive."""
    steps = (
        "tar -xzf runs/colab/dignity_results_$SEED.tar.gz -C runs/colab",
        "cp -r runs/colab/results/v1.0/. results/v1.0/",
        "cp -r runs/colab/docs/img/. docs/img/",
        "python scripts/refresh_doc_numbers.py README.md docs/RESULTS.md",
        "python -m pytest tests/test_docs.py",
    )
    lines = runbook.splitlines()
    missing = [f"missing step: {s}" for s in steps if s not in lines]
    if missing:
        return missing
    at = [lines.index(s) for s in steps]
    return [] if at == sorted(at) else ["the steps are out of order"]


def agent_guide_problems(results_text: str, guide: str) -> list[str]:
    """.claude/CLAUDE.md names the full-run command the Source cells name, and says
    where the marker grammar is defined."""
    commands = {c for c, _ in SOURCE_CELL_RE.findall(results_text)}
    problems = []
    if len(commands) != 1:
        problems.append(f"the Source cells name {len(commands)} commands")
    else:
        (command,) = commands
        if f"{command}  #" not in guide:
            problems.append(f"the full-run command line is not: {command}")
    if "the one helper the M6.6 refresh imports" in guide:
        problems.append("the import direction of split_token is reversed")
    if "defined once in `scripts/refresh_doc_numbers.py`" not in guide:
        problems.append("the guide does not say where the marker grammar is defined")
    return problems


def seed_dir(tmp_path: Path, seed: int) -> Path:
    """A results directory holding only the seed that risk.json records."""
    directory = tmp_path / f"seed_{seed}"
    directory.mkdir()
    (directory / "risk.json").write_text(json.dumps({"seed": seed}))
    return directory


def _interval(value, low, high):
    return {"value": value, "ci_low": low, "ci_high": high}


def outcome_payloads() -> dict[str, dict]:
    """The four results files in one fixed state, the base state of the
    outcome-sentence tests: risk gates pass, the forecast model is below its
    baseline at every horizon with its interval around the chance line, the
    baseline's lower bound is above the model's estimate at 9 of 10 horizons, the
    model warns ahead of no onset, the triage head costs more on every seed, and
    the privacy curve is at chance except at the two largest epsilons. The state
    is written here, so no test depends on what the committed files say."""
    horizons = 10
    model = [_interval(0.04, 0.02, 0.07) for _ in range(horizons)]
    base = [_interval(0.06, 0.055, 0.09) for _ in range(horizons)]
    model[0] = _interval(0.058, 0.02, 0.07)  # the baseline is not ahead at h0
    best = {"majority": 0.3, "window_logistic": 0.5, "block_tree": 0.4}
    return {
        "risk": {
            "seed": EvalConfig().tuning_seed,
            "metrics": {
                "pr_auc": _interval(0.8, 0.7, 0.9),
                "block_level": {"pr_auc": _interval(0.85, 0.75, 0.95)},
            },
            "baselines": {
                name: {"pr_auc": _interval(v, v - 0.1, v + 0.1)}
                for name, v in best.items()
            },
        },
        "forecast": {
            "metrics": {
                "pr_auc_by_horizon": model,
                "prevalence_by_horizon": [0.05] * horizons,
                "lead_time": {"n_missed": 3, "n_onsets": 3, "median": None},
                "seed_spread": {
                    "lead_time_median": {"values": [None, 10.0, 1.0]},
                },
            },
            "baselines": {
                "risk_score_forecast": {
                    "pr_auc_by_horizon": base,
                    "lead_time": {"n_missed": 2, "n_onsets": 3, "median": 10.0},
                }
            },
        },
        "policy": {
            "metrics": {
                "expected_cost": _interval(1.2, 0.8, 1.6),
                "seed_spread": {
                    "expected_cost": {"values": [1.2, 1.2, 1.2]},
                    "baseline_expected_cost": {"values": [1.1, 1.1, 1.1]},
                },
            },
            "baselines": {
                "threshold_policy": {"expected_cost": _interval(1.1, 0.7, 1.5)}
            },
        },
        "privacy": {
            "metrics": {
                "reference": {"risk": {"prevalence": _interval(0.27, 0.2, 0.34)}},
                "points": [
                    {
                        "epsilon": eps,
                        "risk": {
                            "pr_auc": _interval(0.3, 0.2, 0.4)
                            if i < 5
                            else _interval(0.6, 0.5, 0.7)
                        },
                    }
                    for i, eps in enumerate((0.1, 0.3, 1, 3, 10, 30, 100))
                ],
            }
        },
    }


def outcome_results(tmp_path: Path, name: str = "base", **edits) -> Path:
    """The base state written to a directory; `edits` maps a file name to a
    function that changes that payload first."""
    payloads = outcome_payloads()
    for file_name, edit in edits.items():
        edit(payloads[file_name])
    directory = tmp_path / name
    directory.mkdir()
    for file_name, payload in payloads.items():
        (directory / f"{file_name}.json").write_text(json.dumps(payload))
    return directory


def forecast_doc(model: float, baseline: float, fmt: str) -> str:
    """A one-row Forecast section as the refresh would print it."""
    row = (
        f"| `pr_auc_by_horizon.0` | {model} "
        f"<!--num forecast.json metrics.pr_auc_by_horizon.0.value {fmt}--> | "
        f"`risk_score_forecast` {baseline} <!--num forecast.json baselines."
        f"risk_score_forecast.pr_auc_by_horizon.0.value {fmt}--> | s |"
    )
    return (
        "## Forecast\n\n<!--results-start-->\n"
        "| Metric | Model | Best baseline | Source |\n|---|---|---|---|\n"
        f"{row}\n<!--results-end-->\n"
    )


def typed_fact_problems(text: str, results_dir: Path = RESULTS_DIR) -> list[str]:
    """The facts RESULTS.md types by hand that have a source, tied to it: the
    false-positive rate of the recall metric and of the alarm threshold (to the
    metric key and to eval.metrics.FPR_BUDGET), the preset of the headline run (to
    the Source cells' config path), 'lower is better' for cost (to how the policy
    is scored), and the privacy reference's feature set (to privacy.json)."""
    import inspect

    from eval import metrics as eval_metrics
    from eval import run as eval_run
    from train import loop as train_loop

    flat = _flat(text)
    problems = []

    def need(sentence: str, why: str) -> None:
        if sentence not in flat:
            problems.append(f"rewrite the sentence so it says: {sentence!r} ({why})")

    # the false-positive rate: metric key, code constant, description, threshold
    keys = [
        k
        for k in _results_json("risk", results_dir)["metrics"]
        if re.fullmatch(r"recall_at_\d+pct_fpr", k)
    ]
    if len(keys) != 1:
        return [f"risk.json has the recall keys {keys}, expected exactly one"]
    pct = int(re.findall(r"\d+", keys[0])[0])
    if pct != round(eval_metrics.FPR_BUDGET * 100):
        problems.append(
            f"{keys[0]} says {pct}% but eval.metrics.FPR_BUDGET is "
            f"{eval_metrics.FPR_BUDGET}"
        )
    need(
        f"`{keys[0]}` is the recall at a {pct}% false-positive rate",
        "the percent is read from the metric key",
    )
    need(
        f"{pct}%-false-positive threshold of the whole test split",
        "the per-type recall uses eval.metrics.FPR_BUDGET",
    )
    need(
        f"is the {pct}% false-positive point of the last-horizon column on "
        "validation, and an alarm fires at or above it",
        "eval.run._early_warning and eval.metrics.lead_times",
    )
    early = inspect.getsource(eval_run._early_warning)
    if "threshold_at_fpr(y_val, s_val, FPR_BUDGET)" not in early or (
        "val_score[m, last]" not in early
    ):
        problems.append(
            "eval.run._early_warning no longer sets the alarm threshold at the "
            "FPR_BUDGET point of the last-horizon validation column: rewrite the "
            "Early warning bullet of the Protocol"
        )
    if "alarm >= threshold" not in inspect.getsource(eval_metrics.lead_times):
        problems.append(
            "eval.metrics.lead_times no longer fires at or above the threshold: "
            "rewrite the Early warning bullet of the Protocol"
        )

    # the preset the headline run uses, from the config path of the Source cells
    configs = set()
    for command, _ in SOURCE_CELL_RE.findall(text):
        argv = shlex.split(command)
        configs.add(argv[argv.index("--config") + 1] if "--config" in argv else None)
    found = re.fullmatch(r"config/synthetic_(\w+)\.yaml", sorted(map(str, configs))[0])
    if len(configs) != 1 or not found:
        problems.append(
            f"the Source cells name the configs {sorted(map(str, configs))}"
        )
    else:
        (config,) = configs
        preset = found[1]
        need(f"on the `{preset}` synthetic preset", f"the Source cells run {config}")
        need(
            f"on the `{preset}` preset (`{config}`)",
            f"the Source cells run {config}",
        )

    # cost: lower is better, as best.pt for the policy is kept on the negated cost
    need("so lower is better", "the policy is selected on neg_cost, the negated cost")
    if "higher is better" in flat:
        problems.append("expected cost is not higher-is-better: rewrite that sentence")
    if not re.search(r"neg_cost\s*=\s*-", inspect.getsource(train_loop)):
        problems.append(
            "train.loop no longer scores the policy on the negated cost: check the "
            "sentence 'lower is better' of the Policy section"
        )

    # the privacy reference's feature set, and how each feature is released
    privacy = _results_json("privacy", results_dir)["metrics"]
    features = privacy["features"]
    from core.config import DignityConfig

    full = DignityConfig.from_yaml(str(ROOT / "config" / "synthetic_hard.yaml"))
    if not set(features) < set(full.data.features):
        problems.append(
            f"privacy.json features {features} are not a proper subset of the "
            f"configured full set {full.data.features}"
        )
    need(
        "non-private model trained on the privacy feature set",
        f"privacy.json records the features {features}",
    )
    need("which uses the full feature set", "the headline risk model is not restricted")
    released = {n: f["mechanism"] for n, f in privacy["template"]["features"].items()}
    if released != {
        "volume": "laplace",
        "fee_rate": "generalize",
        "tx_count": "generalize",
    } or ("price" in features):
        problems.append(
            f"privacy.json releases {released} over the features {features}: the "
            "Privacy curve opener (Laplace on volume, generalization of fee_rate "
            "and tx_count, price not consumed) is no longer true"
        )
    return problems


class TestResultsDoc:
    REQUIRED = (
        "## Protocol",
        "## Baselines",
        "## Risk",
        "## Forecast",
        "## Policy",
        "## Privacy curve",
        "## Per-type results",
        "## Caveats",
        "## Reproduce",
        "## Colab runbook",
    )

    @pytest.fixture(scope="class")
    def text(self):
        return RESULTS_MD.read_text()

    def test_results_doc_has_every_required_section(self, text):
        for heading in self.REQUIRED:
            assert heading in text

    def test_every_results_section_holds_one_zone_and_the_zones_are_clean(self, text):
        for heading in ZONE_SECTIONS:
            assert len(results_zones(section_of(text, heading))) == 1, heading
        assert len(results_zones(text)) == len(ZONE_SECTIONS)

    def test_every_results_table_row_names_command_and_results_file(self, text):
        problems = [p for zone in results_zones(text) for p in zone_problems(zone)]
        assert problems == []

    def test_a_row_that_reads_another_results_file_than_its_source_cell_is_reported(
        self,
    ):
        source = (
            "`python -m eval.run --config config/synthetic_hard.yaml --seed tuning "
            "--tasks risk,forecast,policy --privacy-sweep` -> `results/v1.0/risk.json`"
        )
        header = "| Metric | Model | Best baseline | Source |\n|---|---|---|---|\n"
        good = f"| `pr_auc` | 0 <!--num risk.json metrics.pr_auc.value .2f--> | x | {source} |"
        wrong = good.replace("<!--num risk.json", "<!--num forecast.json")
        assert zone_problems(header + good) == []
        assert any("forecast.json" in p for p in zone_problems(header + wrong))

    def test_a_marker_the_shared_grammar_cannot_read_is_reported_not_skipped(self):
        source = (
            "`python -m eval.run --config config/synthetic_hard.yaml --seed tuning "
            "--tasks risk,forecast,policy --privacy-sweep` -> `results/v1.0/risk.json`"
        )
        header = "| Metric | Model | Best baseline | Source |\n|---|---|---|---|\n"
        row = (
            f"| `pr_auc` | 0 <!--num risk.json metrics.pr_auc.value--> | x | {source} |"
        )
        assert any("grammar" in p for p in zone_problems(header + row))

    def test_a_privacy_row_that_mixes_two_points_is_reported(self):
        source = (
            "`python -m eval.run --config config/synthetic_hard.yaml --seed tuning "
            "--tasks risk,forecast,policy --privacy-sweep` -> "
            "`results/v1.0/privacy.json`"
        )
        header = "| Metric | Model | Best baseline | Source |\n|---|---|---|---|\n"
        same = (
            "| `pr_auc` | 0 <!--num privacy.json metrics.points.2.risk.pr_auc.value .2f-->"
            " | 0 <!--num privacy.json baselines.window_logistic.points.2.pr_auc.value "
            f".2f--> | {source} |"
        )
        mixed = same.replace("points.2.pr_auc", "points.3.pr_auc")
        assert zone_problems(header + same) == []
        assert any("mixes points" in p for p in zone_problems(header + mixed))

    def test_a_source_command_eval_run_would_reject_is_reported(self):
        good = (
            "python -m eval.run --config config/synthetic_hard.yaml --seed tuning "
            "--tasks risk,forecast,policy --privacy-sweep"
        )
        assert command_problems(good, "privacy") == []
        for bad, why in (
            (good.replace("--privacy-sweep", "--privacy-sweeps"), "rejects"),
            (good.replace("synthetic_hard", "synthetic_gone"), "not a file"),
            (good.replace("python -m eval.run", "python -m eval.gates"), "start"),
            (good.replace("--seed tuning", "--seed 7"), "rejects"),
            (good.replace("--privacy-sweep", ""), "needs --privacy-sweep"),
            (good.replace("risk,forecast,policy", "forecast"), "include risk"),
            (good.replace("risk,forecast,policy", "risk,forcast,policy"), "unknown"),
        ):
            name = "privacy" if "privacy-sweep" not in bad else "risk"
            if "--tasks forecast" in bad:
                name = "risk"
            assert any(why in p for p in command_problems(bad, name)), bad

    def test_a_source_cell_seed_follows_the_seed_of_the_committed_results(self, text):
        problems = source_seed_problems(text)
        assert problems == [], (
            "rewrite the --seed word of these Source cells: it is `tuning` while "
            "risk.json carries the tuning seed and `eval` once it carries the "
            f"evaluation seed (the test decides from risk.json's seed): {problems}"
        )

    def test_a_source_cell_naming_the_wrong_seed_word_is_reported(self, tmp_path):
        cell = (
            "`python -m eval.run --config config/synthetic_hard.yaml --seed tuning` "
            "-> `results/v1.0/risk.json`"
        )
        tuning = seed_dir(tmp_path, EvalConfig().tuning_seed)
        evaluation = seed_dir(tmp_path, EvalConfig().eval_seed)
        assert source_seed_problems(cell, tuning) == []
        assert source_seed_problems(cell, evaluation) != []
        assert source_seed_problems(cell.replace("tuning", "eval"), evaluation) == []
        assert source_seed_problems(cell.replace("tuning", "eval"), tuning) != []

    def test_every_sentence_true_only_on_the_tuning_seed_follows_the_results_seed(
        self, text
    ):
        problems = seed_wording_problems(text)
        assert problems == [], (
            "rewrite these sentences of docs/RESULTS.md: the test requires the "
            "tuning-seed sentences while risk.json carries the tuning seed and "
            "rejects them, and any wording that names the tuning seed as the source "
            "of the numbers, once it carries the evaluation seed (then say the "
            f"numbers are evaluation-seed scores): {problems}"
        )

    def test_a_tuning_seed_sentence_is_required_while_the_results_use_that_seed(
        self, tmp_path
    ):
        tuning = seed_dir(tmp_path, EvalConfig().tuning_seed)
        assert results_on_tuning_seed(tuning)
        doc = " ".join(TUNING_ONLY_SENTENCES)
        assert seed_wording_problems(doc, tuning) == []
        for sentence in TUNING_ONLY_SENTENCES:
            gone = doc.replace(sentence, "A sentence about something else.")
            assert gone != doc
            problems = seed_wording_problems(gone, tuning)
            assert any(sentence in p for p in problems), sentence

    def test_a_tuning_seed_sentence_is_rejected_once_the_results_carry_the_eval_seed(
        self, tmp_path
    ):
        evaluation = seed_dir(tmp_path, EvalConfig().eval_seed)
        assert not results_on_tuning_seed(evaluation)
        doc = " ".join(TUNING_ONLY_SENTENCES)
        problems = seed_wording_problems(doc, evaluation)
        for sentence in TUNING_ONLY_SENTENCES:
            assert any(sentence in p for p in problems), sentence
        for wording in TUNING_ONLY_WORDING:
            assert any(wording in p for p in problems), wording
        rewritten = doc
        for sentence in TUNING_ONLY_SENTENCES:
            rewritten = rewritten.replace(
                sentence, "The numbers are evaluation-seed scores."
            )
        assert seed_wording_problems(rewritten, evaluation) == []

    def test_a_refreshed_evaluation_seed_in_the_marked_sentence_is_not_a_stray_literal(
        self,
    ):
        seed = EvalConfig().eval_seed
        zone = (
            "<!--results-start-->\n| a | 1 <!--num risk.json metrics.pr_auc.value "
            ".2f--> |\n<!--results-end-->\n"
        )
        text = (
            "The numbers below come from seed 1000 <!--num risk.json seed d-->; "
            f"`{seed}` is the evaluation seed.\n{zone}"
        )
        assert eval_seed_literal_problems(text) == []
        refreshed = text.replace("seed 1000 <!--", f"seed {seed} <!--")
        assert refreshed != text
        assert eval_seed_literal_problems(refreshed) == []
        typed = refreshed.replace(" <!--num risk.json seed d-->", "")
        assert eval_seed_literal_problems(typed) != []
        in_zone = text.replace("<!--results-end-->", f"{seed} <!--results-end-->", 1)
        assert eval_seed_literal_problems(in_zone) != []
        # a marked value that merely holds the digits of the seed is not a hit
        for value, fmt in ((f"0.{seed}", ".4f"), (f"{seed}", "d"), (f"1{seed}", "d")):
            marked = text.replace(
                "1 <!--num risk.json metrics.pr_auc.value .2f-->",
                f"{value} <!--num risk.json metrics.n_windows {fmt}-->",
            )
            assert marked != text
            assert eval_seed_literal_problems(marked) == [], value
        for value in (f"0.{seed}", f"1{seed}"):
            assert eval_seed_literal_problems(text + f"\nand {value} typed\n") == []

    def test_every_results_row_quotes_the_metric_its_label_names(self, text):
        assert row_path_problems(text) == []

    ROW_CORRUPTIONS = {
        "swapped Model cells of auroc and pr_auc": lambda t: _swap_cells(
            t, "`auroc`", "`pr_auc`", 1
        ),
        "swapped baseline cells of auroc and pr_auc": lambda t: _swap_cells(
            t, "`auroc`", "`pr_auc`", 2
        ),
        "auroc ci_low repointed to pr_auc": lambda t: t.replace(
            "metrics.auroc.ci_low", "metrics.pr_auc.ci_low", 1
        ),
        "per block pr_auc repointed to per block auroc": lambda t: t.replace(
            "metrics.block_level.pr_auc.value", "metrics.block_level.auroc.value", 1
        ),
        "swapped lead_time n_missed and n_onsets": lambda t: _swap_cells(
            t, "`lead_time.n_missed`", "`lead_time.n_onsets`", 1
        ),
        "swapped policy review_rate and block_rate": lambda t: _swap_cells(
            t, "`review_rate`", "`block_rate`", 1
        ),
        "swapped privacy no-privacy and prevalence": lambda t: _swap_cells(
            t, "`pr_auc` without privacy", "`prevalence`", 1
        ),
        "policy expected_cost ci_high repointed": lambda t: t.replace(
            "policy.json metrics.expected_cost.ci_high",
            "policy.json metrics.review_rate.ci_high",
            1,
        ),
        "review_rate label renamed block_rate": lambda t: t.replace(
            "| `review_rate` |", "| `block_rate` |", 1
        ),
        "window_logistic named majority": lambda t: re.sub(
            r"`window_logistic`( \S+ <!--num risk\.json)", r"`majority`\1", t, count=1
        ),
        "block_tree named window_logistic": lambda t: re.sub(
            r"`block_tree`( \S+ <!--num risk\.json)", r"`window_logistic`\1", t, count=1
        ),
        "privacy baseline repointed to another point": lambda t: t.replace(
            "baselines.window_logistic.points.2.pr_auc.value",
            "baselines.window_logistic.points.3.pr_auc.value",
            1,
        ),
        "no-privacy baseline repointed to a grid point": lambda t: t.replace(
            "baselines.window_logistic.reference.pr_auc.ci_low",
            "baselines.window_logistic.points.0.pr_auc.ci_low",
            1,
        ),
        "per-type recall repointed to another type": lambda t: t.replace(
            "metrics.per_type.fee_spike.value", "metrics.per_type.fee_evasion.value", 1
        ),
    }

    @pytest.mark.parametrize("name", sorted(ROW_CORRUPTIONS))
    def test_a_swapped_or_repointed_cell_is_reported(self, text, name):
        corrupted = self.ROW_CORRUPTIONS[name](text)
        assert corrupted != text, (
            f"the corruption {name!r} found nothing to change in docs/RESULTS.md: "
            "edit ROW_CORRUPTIONS in tests/test_docs.py so that it takes its token "
            "from the document, not from a number the document no longer has"
        )
        assert row_path_problems(corrupted) != [], (
            f"row_path_problems missed the corruption {name!r}: a cell that names "
            "one metric but reads another must be reported"
        )

    def test_the_header_exemption_covers_every_table_header_and_no_data_row(self):
        zone = (
            "| Type | Recall [CI] | n_windows | Source |\n|---|---|---|---|\n"
            "| `layering` | 1 | 2 | nothing |\n\n"
            "| Metric | Model | Best baseline | Source |\n|---|---|---|---|\n"
        )
        problems = zone_problems(zone)
        assert any("layering" in p for p in problems)
        tables, loose = zone_tables(zone)
        assert [h for h, _ in tables] == [
            "| Type | Recall [CI] | n_windows | Source |",
            "| Metric | Model | Best baseline | Source |",
        ]
        assert loose == []
        for data_row_named_like_a_header in (
            "| Metric | Model | Best baseline | Source |\n|---|---|---|---|\n"
            "| Metric | 1 | 2 | nothing |\n",
        ):
            assert zone_problems(data_row_named_like_a_header) != []

    def test_the_results_tables_use_exactly_the_two_agreed_headers(self, text):
        headers = {
            " | ".join(_cells(h))
            for zone in results_zones(text)
            for h, _ in zone_tables(zone)[0]
        }
        assert headers == set(RESULTS_HEADERS)

    SECTION = {
        "risk": "## Risk",
        "forecast": "## Forecast",
        "policy": "## Policy",
    }

    @pytest.mark.parametrize("task", ["risk", "forecast", "policy"])
    def test_the_outcome_sentence_of_each_task_matches_the_committed_results(
        self, text, task
    ):
        phrases = outcome_phrases()[task]
        section = section_of(text, self.SECTION[task])
        problems = phrase_problems(section, phrases.required, phrases.forbidden)
        assert problems == [], (
            f"rewrite the outcome sentences of the {self.SECTION[task]} section of "
            "docs/RESULTS.md: each one is recomputed from the committed results "
            "file (missing = write it, contradicts = it says the opposite of the "
            f"numbers): {problems}"
        )

    @pytest.mark.parametrize("task", ["risk", "forecast", "policy", "privacy"])
    def test_the_opening_paragraph_gives_the_verdict_of_each_task_and_the_curve(
        self, text, task
    ):
        phrases = outcome_phrases()[task]
        paragraph = opening_paragraph(text)
        problems = phrase_problems(paragraph, phrases.headline, phrases.forbidden)
        assert problems == [], (
            f"rewrite the {task} verdict in the Outcome paragraph at the top of "
            "docs/RESULTS.md: it must repeat the headline phrases of that task's "
            "section, recomputed from the committed results file (a task's phrases "
            f"are listed in outcome_phrases in tests/test_docs.py): {problems}"
        )

    def test_the_opening_paragraph_comes_before_the_mechanics_and_holds_no_number(
        self, text
    ):
        paragraph = opening_paragraph(text)
        assert text.index(paragraph) < text.index("This file reports"), (
            "the Outcome paragraph must come before the paragraph that starts "
            "'This file reports'"
        )
        assert not re.search(r"\d", paragraph), (
            "the Outcome paragraph states outcomes in words only: take every "
            "digit out of it (the numbers are in the tables)"
        )

    def test_an_outcome_sentence_fails_when_the_results_change_and_it_does_not(
        self, tmp_path
    ):
        base = outcome_phrases(outcome_results(tmp_path))
        sections = {t: " ".join(base[t].required) for t in base}
        openings = {t: " ".join(base[t].headline) for t in base}
        for task in base:
            assert (
                phrase_problems(
                    sections[task], base[task].required, base[task].forbidden
                )
                == []
            )
            assert (
                phrase_problems(
                    openings[task], base[task].headline, base[task].forbidden
                )
                == []
            )

        def risk_gate_fails(r):
            r["metrics"]["pr_auc"]["ci_low"] = 0.1

        def forecast_has_lead_times(r):
            r["metrics"]["lead_time"].update(n_missed=0, median=3.0)

        def privacy_leaves_chance(r):
            r["metrics"]["points"][0]["risk"]["pr_auc"]["ci_low"] = 0.99

        def policy_clears(r):
            r["metrics"]["expected_cost"].update(value=0.1, ci_low=0.05, ci_high=0.2)

        drifts = (
            ("risk", risk_gate_fails),
            ("forecast", forecast_has_lead_times),
            ("privacy", privacy_leaves_chance),
            ("policy", policy_clears),
        )
        for task, edit in drifts:
            drifted = outcome_phrases(
                outcome_results(tmp_path, f"drift_{task}", **{task: edit})
            )[task]
            if task != "privacy":
                assert phrase_problems(
                    sections[task], drifted.required, drifted.forbidden
                ), task
            assert phrase_problems(
                openings[task], drifted.headline, drifted.forbidden
            ), task

    FORECAST_DRIFTS = (
        (
            "the model's interval stops containing the prevalence",
            lambda r: r["metrics"]["pr_auc_by_horizon"][0].update(
                ci_low=0.9, ci_high=1.0
            ),
        ),
        (
            "the baseline's interval stops lying above the prevalence",
            lambda r: r["baselines"]["risk_score_forecast"]["pr_auc_by_horizon"][
                0
            ].update(ci_low=0.0),
        ),
        (
            "the baseline's lower bound is above the model's estimate everywhere",
            lambda r: [h.update(value=-1.0) for h in r["metrics"]["pr_auc_by_horizon"]],
        ),
        (
            "the model's estimate stops being below the baseline's",
            lambda r: r["metrics"]["pr_auc_by_horizon"][3].update(value=0.5),
        ),
        (
            "every training seed warns ahead of an onset",
            lambda r: r["metrics"]["seed_spread"]["lead_time_median"].update(
                values=[3.0, 3.0, 3.0]
            ),
        ),
        (
            "no training seed warns ahead of an onset",
            lambda r: r["metrics"]["seed_spread"]["lead_time_median"].update(
                values=[None, None, None]
            ),
        ),
        (
            "the file records onset counts per seed",
            lambda r: r["metrics"]["seed_spread"]["lead_time_median"].update(
                n_missed=[1, 2, 3]
            ),
        ),
    )

    @pytest.mark.parametrize(
        ("name", "edit"), FORECAST_DRIFTS, ids=[n for n, _ in FORECAST_DRIFTS]
    )
    def test_each_forecast_comparison_sentence_fails_when_its_own_numbers_change(
        self, tmp_path, name, edit
    ):
        base = outcome_phrases(outcome_results(tmp_path))["forecast"]
        doc = " ".join(base.required)
        assert phrase_problems(doc, base.required, base.forbidden) == []
        drifted = outcome_phrases(outcome_results(tmp_path, "drift", forecast=edit))
        problems = phrase_problems(
            doc, drifted["forecast"].required, drifted["forecast"].forbidden
        )
        assert problems, name

    def test_the_n_a_explanation_is_required_only_while_a_seed_has_no_lead_time(
        self, tmp_path
    ):
        explanation = "(n/a: no onset warned ahead on that seed)"
        some_null = outcome_phrases(outcome_results(tmp_path))["forecast"]
        assert explanation in some_null.required
        assert explanation not in some_null.forbidden

        def all_warned(r):
            r["metrics"]["seed_spread"]["lead_time_median"]["values"] = [1.0, 2.0, 3.0]

        none_null = outcome_phrases(
            outcome_results(tmp_path, "warned", forecast=all_warned)
        )
        assert explanation in none_null["forecast"].forbidden
        assert explanation not in none_null["forecast"].required

    def test_at_how_many_horizons_every_count_has_exactly_one_word(self, tmp_path):
        words = {
            (n, total): horizon_word(n, total)
            for total in (2, 3, 10)
            for n in range(total + 1)
        }
        assert words[(0, 10)] == "none" and words[(10, 10)] == "every"
        assert words[(3, 10)] == "some" and words[(4, 10)] == "some"
        assert words[(5, 10)] == "half"
        assert words[(6, 10)] == "most" and words[(9, 10)] == "most"
        assert words[(1, 2)] == "half" and words[(1, 3)] == "some"
        assert words[(2, 3)] == "most"
        assert set(words.values()) == set(HORIZON_PHRASES)

        # with exactly half of the horizons, the doc must say half, not most
        def half_ahead(r):
            for m in r["metrics"]["pr_auc_by_horizon"][:5]:
                m["value"] = 0.058  # the baseline's lower bound is not above it

        phrases = outcome_phrases(
            outcome_results(tmp_path, "half", forecast=half_ahead)
        )
        template = "the baseline's lower bound is above the model's point estimate {}"
        assert template.format(HORIZON_PHRASES["half"]) in phrases["forecast"].required
        assert template.format(HORIZON_PHRASES["most"]) in phrases["forecast"].forbidden
        assert HORIZON_WORDS_RULE in phrases["forecast"].required

    def test_the_policy_seed_sentence_fails_when_the_seeds_stop_agreeing(
        self, tmp_path
    ):
        base = outcome_phrases(outcome_results(tmp_path))["policy"]
        doc = " ".join(base.required)
        assert phrase_problems(doc, base.required, base.forbidden) == []

        def seeds_disagree(r):
            r["metrics"]["seed_spread"]["baseline_expected_cost"]["values"] = [
                9.0,
                1.0,
                1.0,
            ]

        drifted = outcome_phrases(
            outcome_results(tmp_path, "drift", policy=seeds_disagree)
        )
        assert phrase_problems(
            doc, drifted["policy"].required, drifted["policy"].forbidden
        )

    def test_the_forecast_section_states_the_chance_line_and_the_lead_time_spread(
        self, text
    ):
        section = _flat(section_of(text, "## Forecast"))
        for needle in (
            "`prevalence_by_horizon`",
            "the scored onsets less the missed ones",
            "metrics.seed_spread.lead_time_median.values.0",
            "metrics.seed_spread.lead_time_median.values.1",
            "metrics.seed_spread.lead_time_median.values.2",
        ):
            assert needle in section, needle

    def test_a_median_lead_time_rests_on_the_scored_onsets_less_the_missed_ones(
        self, tmp_path
    ):
        problems = lead_basis_problems()
        assert problems == [], (
            "forecast.json's lead_time values are not the scored onsets less the "
            "missed ones, so the sentence that says a median rests on them is "
            f"false: rewrite it in the Forecast section ({problems})"
        )
        (tmp_path / "forecast.json").write_text(
            (RESULTS_DIR / "forecast.json").read_text()
        )
        forecast = _results_json("forecast", tmp_path)
        lead = forecast["baselines"]["risk_score_forecast"]["lead_time"]
        lead["n_missed"] = lead["n_onsets"] - len(lead["values"]) + 1
        (tmp_path / "forecast.json").write_text(json.dumps(forecast))
        assert lead_basis_problems(tmp_path) != []

    def test_the_forecast_cells_never_show_a_tie_where_the_values_differ(self, text):
        problems = forecast_cell_problems(text)
        assert problems == [], (
            "a Forecast row prints the same number for the model and the baseline "
            "where forecast.json has different ones: raise the digits of the .Nf "
            f"format in the markers of that row, then refresh ({problems})"
        )

    @pytest.mark.parametrize(
        ("model", "baseline"), [(0.0100, 0.0104), (0.0104, 0.0100)]
    )
    def test_a_tie_in_print_where_the_values_differ_is_reported_whichever_is_larger(
        self, tmp_path, model, baseline
    ):
        def horizon_zero(r):
            r["metrics"]["pr_auc_by_horizon"][0].update(value=model)
            r["baselines"]["risk_score_forecast"]["pr_auc_by_horizon"][0].update(
                value=baseline
            )

        results = outcome_results(tmp_path, forecast=horizon_zero)
        coarse = forecast_doc("0.01", "0.01", ".2f")
        fine = forecast_doc(f"{model:.4f}", f"{baseline:.4f}", ".4f")
        assert forecast_cell_problems(coarse, results) != []
        assert forecast_cell_problems(fine, results) == []

    def test_equal_values_may_print_equal(self, tmp_path):
        def tied(r):
            r["metrics"]["pr_auc_by_horizon"][0].update(value=0.0100)
            r["baselines"]["risk_score_forecast"]["pr_auc_by_horizon"][0].update(
                value=0.0100
            )

        results = outcome_results(tmp_path, forecast=tied)
        assert (
            forecast_cell_problems(forecast_doc("0.01", "0.01", ".2f"), results) == []
        )

    def test_the_hand_typed_facts_that_have_a_source_follow_it(self, text):
        problems = typed_fact_problems(text)
        assert problems == [], (
            "docs/RESULTS.md types these facts by hand and the code or results "
            "file that defines them says otherwise (the test reads the false-"
            f"positive rate from the metric key and eval.metrics.FPR_BUDGET, the "
            "preset from the Source cells' config path, the cost direction from "
            "how the policy is selected and the feature set from privacy.json): "
            f"{problems}"
        )

    @pytest.mark.parametrize(
        ("name", "pattern", "replacement"),
        [
            (
                "recall metric described at another rate",
                r"(is the recall at a )\d+(% false-positive rate)",
                r"\g<1>7\2",
            ),
            (
                "per-type recall described at another rate",
                r"\d+(%-false-positive threshold of the whole test split)",
                r"7\1",
            ),
            (
                "alarm threshold at another rate",
                r"(The alarm threshold of the lead time is the )\d+%",
                r"\g<1>7%",
            ),
            (
                "alarm threshold on another column",
                r"last-horizon column on validation",
                "first-horizon column on validation",
            ),
            (
                "alarm fires only above the threshold",
                r"an alarm fires at or above it",
                "an alarm fires above it",
            ),
            (
                "headline preset named as another preset",
                r"on the `\w+` synthetic preset",
                "on the `easy` synthetic preset",
            ),
            (
                "Presets bullet names another preset",
                r"on the `\w+` preset \(`config/synthetic_\w+\.yaml`\)",
                "on the `easy` preset (`config/synthetic_easy.yaml`)",
            ),
            ("cost direction reversed", r"so lower is better", "so higher is better"),
            (
                "reference feature set called the full one",
                r"trained on the privacy feature set",
                "trained on the full feature set",
            ),
        ],
    )
    def test_a_hand_typed_fact_that_contradicts_its_source_is_reported(
        self, text, name, pattern, replacement
    ):
        corrupted = re.sub(pattern, replacement, text, count=1)
        assert corrupted != text, (
            f"{name!r}: the sentence it corrupts is not in docs/RESULTS.md; "
            "typed_fact_problems requires it, so restore the sentence first"
        )
        assert typed_fact_problems(corrupted) != [], name

    def test_a_fact_whose_source_changed_is_reported(self, tmp_path, text):
        directory = tmp_path / "privacy"
        directory.mkdir()
        for name in ("risk", "privacy"):
            (directory / f"{name}.json").write_text(
                (RESULTS_DIR / f"{name}.json").read_text()
            )
        privacy = _results_json("privacy", directory)
        privacy["metrics"]["features"] = privacy["metrics"]["features"] + ["price"]
        (directory / "privacy.json").write_text(json.dumps(privacy))
        assert typed_fact_problems(text, directory) != []
        risk = _results_json("risk", directory)
        risk["metrics"]["recall_at_5pct_fpr"] = risk["metrics"].pop(
            "recall_at_1pct_fpr"
        )
        (directory / "risk.json").write_text(json.dumps(risk))
        assert any("FPR_BUDGET" in p for p in typed_fact_problems(text, directory))

    def test_the_spread_is_the_sample_standard_deviation(self, text):
        import inspect

        from eval import run as eval_run

        source = inspect.getsource(eval_run._spread)
        assert spread_std_problems(text, source) == []
        assert spread_std_problems(text, source.replace("ddof=1", "ddof=0")) != []
        assert spread_std_problems(text.replace("not the population one", ""), source)

    def test_every_gate_and_outcome_sentence_is_in_the_section_it_describes(self, text):
        # one outcome sentence per task, written from the JSON, so it names the
        # gate or the baseline it compares with
        assert "model gate" in section_of(text, "## Risk")
        assert "`risk_score_forecast`" in section_of(text, "## Forecast")
        assert "`threshold_policy`" in section_of(text, "## Policy")

    def test_the_protocol_names_the_tuning_seed_by_marker_and_the_evaluation_seed_in_code(
        self, text
    ):
        protocol = section_of(text, "## Protocol")
        assert "1000 <!--num data_gate_tuning.json seed d-->" in protocol, (
            "the Protocol names the data-gate seed through a marker on "
            "data_gate_tuning.json, which stays on the tuning seed: restore it"
        )
        assert "`core.config.EvalConfig.eval_seed`" in protocol, (
            "the Protocol names the evaluation seed by the config field that "
            "defines it, in code"
        )
        assert "`2026`" in protocol, (
            "the Protocol writes the evaluation seed in backticks (code), outside "
            "every results zone, as the one place it is typed"
        )
        problems = eval_seed_literal_problems(text)
        assert problems == [], (
            "write the evaluation seed only in code outside a results zone, or "
            f"through a marker: {problems}"
        )
        # risk.json's seed is quoted only in a sentence with no role for it
        for line in text.splitlines():
            if "<!--num risk.json seed d-->" in line:
                assert "numbers below come from seed" in line, (
                    "risk.json's seed is quoted only in the sentence that starts "
                    f"'The numbers below come from seed': {line[:60]!r}"
                )

    def test_the_protocol_states_the_freeze_as_a_standing_commitment(self, text):
        flat = _flat(text)
        assert (
            "the final run is made from a commit tagged `v1.0-freeze`, the "
            "generator presets and the evaluation seed do not change after that "
            "tag is created, and the model and the presets are not re-tuned "
            "against the evaluation seed's test split" in flat
        )
        # no sentence may say the tag exists: it is created at the release
        assert "frozen at the `v1.0-freeze`" not in flat
        assert "were frozen" not in flat

    def test_the_protocol_states_selection_and_warm_start_per_task(self, text):
        flat = _flat(section_of(text, "## Protocol"))
        for needle in (
            "block-level 70/15/15 split",
            "`data.split`",
            "validation PR-AUC",
            "`train.select_metric`",
            "`neg_cost` under the review cap",
            "`train.init_backbone_from`",
            "`eval.n_bootstrap`",
            "`eval.n_train_seeds`",
            "triage head has no tuned thresholds",
            "quoted under the risk, forecast and policy tables",
            "95% bootstrap intervals over blocks",
            "first training seed",
            "`metrics.seed_spread`",
            "validation split only",
        ):
            assert needle in flat, needle
        for stale in ("The policy settings are tuned", "under each table"):
            assert stale not in flat, stale

    def test_the_provenance_sentences_cover_the_four_eval_run_files_only(self, text):
        flat = _flat(text)
        for needle in (
            "The data-gate numbers in the Protocol are the exception",
            "Its output is `results/v1.0/data_gate_tuning.json`: validation scores "
            "that `eval.gates` writes",
            "The data-gate numbers are validation scores that `eval.gates` writes "
            "on the tuning seed",
            "Each of the four `eval.run` results records `git_sha` and `config_hash`",
            "`results/v1.0/data_gate_tuning.json` records neither",
            "The four `eval.run` results were produced on a Colab L4",
            "The numbers of the four `eval.run` results above come from the full run",
        ):
            assert needle in flat, needle
        for stale in (
            "The committed results were produced",
            "The numbers above come from the full run",
            "`results/v1.0/risk.json` records `git_sha`",
        ):
            assert stale not in flat, stale
        for name in ("risk", "forecast", "policy", "privacy"):
            assert {"git_sha", "config_hash"} <= set(_results_json(name)), name
        assert not {"git_sha", "config_hash"} & set(_results_json("data_gate_tuning"))

    def test_the_reproduce_section_gives_the_command_that_writes_the_gate_file(
        self, text
    ):
        assert eval_gates_command_problems(text) == []

    def test_a_gate_command_eval_gates_rejects_or_that_writes_elsewhere_is_reported(
        self,
    ):
        command = (
            "python -m eval.gates --seed 1000 --out results/v1.0/data_gate_tuning.json"
        )
        assert eval_gates_command_problems(f"## Reproduce\n\n{command}\n") == []
        for bad in (
            command.replace("--out", "--output"),
            command.replace("1000", "7"),
            command.replace("data_gate_tuning", "other"),
            command.replace(" --out results/v1.0/data_gate_tuning.json", ""),
            "",
        ):
            assert eval_gates_command_problems(f"## Reproduce\n\n{bad}\n"), bad

    def test_the_block_level_target_is_not_named_after_the_training_label(self, text):
        assert "block label" not in _flat(text).lower()
        assert "block_label" not in _flat(section_of(text, "## Caveats"))
        assert text.count("`block_label`") == 1

    def test_the_agent_guide_names_the_command_the_source_cells_name(self, text):
        guide = (ROOT / ".claude" / "CLAUDE.md").read_text()
        problems = agent_guide_problems(text, guide)
        assert problems == [], (
            "set the full-run line of .claude/CLAUDE.md to the command of the "
            "Source cells of docs/RESULTS.md (the final release run uses "
            f"--seed eval), and keep the marker-grammar sentence: {problems}"
        )
        command = " --tasks risk,forecast,policy --privacy-sweep  #"
        assert command in guide
        assert agent_guide_problems(text, guide.replace(command, "  #"))
        reversed_ = guide.replace(
            "defined once in `scripts/refresh_doc_numbers.py`",
            "the one helper the M6.6 refresh imports",
        )
        assert agent_guide_problems(text, reversed_)

    def test_the_baselines_section_names_every_baseline_and_both_logistic_provenances(
        self, text
    ):
        flat = _flat(section_of(text, "## Baselines"))
        for needle in (
            "`majority`",
            "`window_logistic`",
            "`block_tree`",
            "`risk_score_forecast`",
            "`threshold_policy`",
            "`window_logistic` on the raw columns",
            "`window_logistic` on the released windows",
        ):
            assert needle in flat, needle

    def test_risk_rows_pair_window_with_window_logistic_and_block_with_block_tree(
        self, text
    ):
        rows = [
            r
            for h, rs in zone_tables(results_zones(section_of(text, "## Risk"))[0])[0]
            for r in rs
        ]
        assert len(rows) == 10, (
            f"the Risk table has {len(rows)} rows, expected five window rows and "
            "five block rows"
        )
        for row in rows[:5]:
            assert "baselines.window_logistic." in row and "block_level" not in row, (
                f"a window row of the Risk table does not read window_logistic: {row[:40]!r}"
            )
        for row in rows[5:]:
            assert "baselines.block_tree." in row and "metrics.block_level." in row, (
                f"a block row of the Risk table does not read block_tree: {row[:40]!r}"
            )

    def test_forecast_rows_pair_every_horizon_with_risk_score_forecast(self, text):
        rows = [
            r
            for h, rs in zone_tables(results_zones(section_of(text, "## Forecast"))[0])[
                0
            ]
            for r in rs
        ]
        horizons = len(_results_json("forecast")["metrics"]["pr_auc_by_horizon"])
        for n in range(horizons):
            row = next(
                (r for r in rows if f"`pr_auc_by_horizon.{n}`" in r), "(no such row)"
            )
            assert f"metrics.pr_auc_by_horizon.{n}.value" in row, (
                f"forecast.json has {horizons} horizons: the Forecast table needs "
                f"a pr_auc_by_horizon.{n} row that reads the model's value"
            )
            assert (
                f"baselines.risk_score_forecast.pr_auc_by_horizon.{n}.value" in row
            ), f"the pr_auc_by_horizon.{n} row must read risk_score_forecast"
        joined = "".join(rows)
        for path in (
            "metrics.lead_time.median",
            "metrics.lead_time.n_missed",
            "metrics.lead_time.n_onsets",
        ):
            assert path in joined

    def test_the_lead_time_chart_is_embedded_only_when_the_model_has_lead_times(
        self, text
    ):
        has_values = bool(_results_json("forecast")["metrics"]["lead_time"]["values"])
        assert ("forecast_lead_time.png" in text) == has_values, (
            "forecast.json has "
            + (
                "lead-time values for the model, so embed"
                if has_values
                else "no lead-time values for the model, so remove"
            )
            + " forecast_lead_time.png in the Forecast section of docs/RESULTS.md"
        )

    def test_the_policy_cost_table_and_review_budget_are_the_configured_ones(
        self, text
    ):
        from core.config import DignityConfig

        policy = DignityConfig.from_yaml(
            str(ROOT / "config" / "synthetic_hard.yaml")
        ).policy
        section = section_of(text, "## Policy")
        assert "`policy.costs`" in section
        rows = [
            _cells(line)
            for line in section.splitlines()
            if re.match(r"\|\s*`(allow|review|block)`", line)
        ]
        assert {r[0].strip("`") for r in rows} == set(policy.costs)
        for action, anomaly, normal in rows:
            expected = policy.costs[action.strip("`")]
            assert float(anomaly) == expected["anomaly"], action
            assert float(normal) == expected["normal"], action
        budget = re.search(r"`policy\.review_budget` is ([\d.]+)", section)
        assert budget and float(budget[1]) == policy.review_budget

    def test_the_per_type_table_lists_every_type_the_results_file_scores(self, text):
        per_type = _results_json("risk")["metrics"]["per_type"]
        zone = results_zones(section_of(text, "## Per-type results"))[0]
        ((header, rows),) = zone_tables(zone)[0]
        assert _cells(header) == ["Type", "Recall [CI]", "n_windows", "Source"]
        assert [_cells(r)[0].strip("`") for r in rows] == sorted(per_type), (
            "risk.json scores these anomaly types, sorted: "
            f"{sorted(per_type)}; make the Per-type table list exactly them"
        )
        for row in rows:
            name = _cells(row)[0].strip("`")
            assert f"metrics.per_type.{name}.value" in row
            assert f"metrics.per_type.{name}.n_windows d-->" in row
        flat = _flat(section_of(text, "## Per-type results"))
        assert "baselines are not scored per type" in flat
        assert "1%-false-positive threshold of the whole test split" in flat
        assert all(
            not _results_json("risk")["baselines"][b]["per_type"]
            for b in ("majority", "window_logistic", "block_tree")
        )

    def test_the_privacy_section_reuses_the_privacy_doc_sentences_verbatim(self, text):
        privacy = (ROOT / "docs/PRIVACY.md").read_text()
        problems = privacy_verbatim_problems(text, privacy)
        assert problems == [], (
            "the Privacy curve section of docs/RESULTS.md reuses these sentences "
            "of the curve section of docs/PRIVACY.md word for word, and they have "
            "changed: copy them again, as the statement is rewritten at the final "
            f"evaluation: {problems}"
        )
        section = _flat(section_of(text, "## Privacy curve"))
        assert (
            "(THREAT-MODEL.md)" in text
            and "PRIVACY.md#privacy-versus-utility-curve" in text
        )
        assert "`window_logistic` on the released windows" in section

    def test_a_sentence_reused_from_the_privacy_doc_is_reused_whole_or_reported(
        self, text
    ):
        privacy = (ROOT / "docs/PRIVACY.md").read_text()
        # the extraction keeps a sentence to its full stop
        shipped = sentences_of("A first. The shipped x, with y and z. Third one.")
        assert shipped[1] == "The shipped x, with y and z."
        for tail in (
            ", with unrounded public bounds and in seeded mode",
            ", and the floor test above runs on the easy preset",
            "The statement is about PR-AUC. ",
            " The file also carries AUROC, and up to ε = 10 its intervals do not "
            "all contain 0.5.",
        ):
            assert tail in text, (
                f"the reused privacy sentence is not in docs/RESULTS.md as {tail!r}: "
                "this case cuts it to show the verbatim check reports a cut, so "
                "update its text to the rewritten sentence"
            )
            cut = text.replace(tail, "")
            problems = privacy_verbatim_problems(cut, privacy)
            assert problems, tail

    def test_the_privacy_table_marks_the_reference_chance_line_and_every_grid_point(
        self, text
    ):
        zone = results_zones(section_of(text, "## Privacy curve"))[0]
        paths = [m[2] for m in NUM_RE.findall(zone)]
        grid = len(_results_json("privacy")["metrics"]["points"])
        indexes = {int(i) for d in paths for i in re.findall(r"\bpoints\.(\d+)\.", d)}
        assert indexes == set(range(grid)), (
            f"privacy.json has {grid} grid points: the Privacy curve table needs "
            f"one row for each of points 0 to {grid - 1} (it has {sorted(indexes)})"
        )
        for path in (
            "metrics.reference.risk.pr_auc.value",
            "metrics.reference.risk.pr_auc.ci_low",
            "metrics.reference.risk.prevalence.value",
        ):
            assert path in paths, f"the Privacy curve table must read {path}"
        for n in range(grid):
            assert f"metrics.points.{n}.epsilon" in paths, (
                f"the label of grid point {n} must carry the marker "
                f"metrics.points.{n}.epsilon"
            )
        assert not any("per_type" in d for d in paths)

    def test_no_privacy_gate_derived_privacy_number_or_attribution_is_written(
        self, text
    ):
        section = section_of(text, "## Privacy curve")
        assert not re.search(r"\bgate\b", section, re.I)
        flat = _flat(text).lower()
        for phrase in (
            "most retained signal",
            "attribution",
            "candidate",
            "comes from the generalized",
            "costs ",
        ):
            assert phrase not in _flat(section).lower(), phrase
        for phrase in ("most retained signal", "attribution", "candidate"):
            assert phrase not in flat, phrase

    def test_the_caveats_state_each_limit_of_the_results(self, text):
        flat = _flat(section_of(text, "## Caveats"))
        for needle in (
            "fee_rate",
            "tx_count",
            "not calibrated",
            "evaluation seed",
            "not re-tuned",
            "synthetic data only",
            "DGNTY-34",
            "DGNTY-27",
            "DGNTY-35",
            "DGNTY-44",
            "no window covers",
            "counts as missed",
            "small sample",
            "<!--num forecast.json metrics.lead_time.n_onsets d-->",
            "one hourly bar per row",
            "trained and evaluated on ε-private inputs",
        ):
            assert needle in flat or needle in flat.replace("-", " "), needle

    def test_the_caveats_name_the_calibration_exchange_pairs_and_windows_as_recorded(
        self, text
    ):
        sources = [
            s
            for stat in json.loads(
                (ROOT / "data/calibration/reference_stats.json").read_text()
            ).values()
            for s in stat["sources"]
        ]
        flat = _flat(section_of(text, "## Caveats"))
        for source in sources:
            exchange, pair, _, window = source.split()[:4]
            assert exchange.capitalize() in flat, exchange
            assert pair in flat, pair
            assert window in flat, window

    def test_the_runbook_requests_the_gpu_the_committed_results_ran_on(self, text):
        runbook = section_of(text, "## Colab runbook")
        assert "colab new -s dignity-eval --gpu L4" in runbook
        assert "T4" not in runbook
        flat = _flat(runbook)
        assert "produced on a Colab L4" in flat
        assert "`SEED=eval`" in flat
        assert "the commit tagged `v1.0-freeze`" in flat

    def test_the_reproduce_section_gives_the_smoke_run_the_doc_check_and_the_refresh(
        self, text
    ):
        flat = section_of(text, "## Reproduce")
        for needle in (
            "python -m eval.run --smoke --seed tuning",
            "pytest tests/test_docs.py",
            "python scripts/refresh_doc_numbers.py README.md docs/RESULTS.md",
        ):
            assert needle in flat, needle

    def test_every_image_the_docs_embed_exists(self):
        # resolved against the embedding file's folder, like every other link, and
        # a badge's URL is not a path
        for path in (ROOT / "README.md", RESULTS_MD):
            for ref in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", path.read_text()):
                if re.match(r"\w+:", ref):
                    continue
                assert (path.parent / ref).exists(), ref

    def test_every_chart_is_embedded_with_alt_text_that_says_what_it_shows(self, text):
        charts = re.findall(r"!\[([^\]]*)\]\(img/(\w+\.png)\)", text)
        names = {name for _, name in charts}
        assert {
            "risk_pr_curve.png",
            "risk_per_type_recall.png",
            "forecast_pr_auc_by_horizon.png",
            "policy_cost.png",
            "privacy_curve.png",
        } <= names
        for alt, name in charts:
            assert len(alt.split()) >= 6, name


HOW_BUILT = ROOT / "docs" / "HOW-IT-WAS-BUILT.md"
STATUSES = (
    "fixed",
    "open (maintainer decision)",
    "closes with the final evaluation (M7)",
    "deferred (v1.1)",
)
PR_LINK = re.compile(r"\[#\d+\]\(https://github\.com/\S+/pull/\d+\)")
BUILT_REPO = "https://github.com/crichalchemist/Dignity"
THIS_PR = 41  # the pull request that carries this page; a page cannot cite its own run
# Each id's PRs, in the order the row lists them, as `gh pr list --state merged` maps
# them (titles and bodies that name the id, and the PRs that removed or rewrote the
# four guides); an empty tuple means no PR addressed the item.
PR_MAP = {
    1: (31, 33),
    2: (32, 33),
    3: (22, 23),
    4: (23,),
    5: (21,),
    6: (15, 17),
    7: (),
    8: (36, 37),
    9: (29, 30),
    10: (27, 26, 25),
    11: (26, 25, 24),
    12: (16, 40, 41),
    13: (),
    14: (),
    15: (),
    16: (26, 25),
    17: (26,),
    18: (26, 25),
    19: (26,),
    20: (35,),
    21: (28,),
    22: (22,),
    23: (21, 20),
    24: (23,),
    25: (23,),
    26: (26,),
    27: (40, 34),
    28: (36,),
    29: (16,),
    30: (39, 30),
    31: (21,),
    32: (19,),
    33: (22, 21),
    34: (34,),
    35: (23,),
    36: (40, 18, 41),
    37: (34,),
    38: (27,),
    39: (27, 25),
    40: (38, 37, 36),
    41: (20,),
    42: (),
    43: (),
    44: (),
    45: (),
}
# The push run of each merged PR's merge commit on dev (workflow ci, success).
RUN_FOR = {
    15: 42538310125,
    16: 42539967316,
    17: 42540734883,
    18: 42541872548,
    19: 42543678002,
    20: 42544661411,
    21: 42546616622,
    22: 42547586587,
    23: 42548635331,
    24: 42552101263,
    25: 42553746645,
    26: 42565042769,
    27: 42570160464,
    28: 42574940294,
    29: 42578189720,
    30: 42597188123,
    31: 42612468724,
    32: 42707762444,
    33: 42714693737,
    34: 42741432391,
    35: 42753203935,
    36: 42765940079,
    37: 42815370353,
    38: 42920455265,
    39: 42949003267,
    40: 42971898695,
}
# The review page's counts and the ids it filed.
REVIEWERS = ("code", "training", "docs", "synthetic data", "datasets")
RAW_FINDINGS, DEFECTS, RECOMMENDATIONS = 42, 31, 8
FIRST_DEFECT, LAST_DEFECT, LAST_RECOMMENDATION = 7, 37, 45
NUMBER_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 8: "Eight"}
# The checks both protected branches require, from
# `gh api repos/crichalchemist/Dignity/branches/{dev,main}/protection`
# (required_status_checks); the doc's list is also derived from ci.yml.
REQUIRED_CHECKS = ("lint", "test (3.10)", "test (3.12)", "smoke")
UNTITLED_PRS = (12, 14)  # PRs merged into dev whose titles carry no DGNTY-n
# `gh pr list --state merged --base dev --limit 200` listed 28 PRs when the page was
# written; 11 more were merged into master and main (39 in all)
MERGED_INTO_DEV, MERGED_ELSEWHERE = 28, 11
# Finding and Area cells of every row except DGNTY-8 (which privacy_row_problems
# ties to docs/PRIVACY.md), as the tracker titles and the review page give them.
# If a paraphrase is edited on purpose, check it against the tracker, then update.
FINDING_DIGEST = "f2b0754637e1e6d10dcd5ac6751933083977f85d0ea876f120d1d45be8b3a8e6"
PRIVATE_STRINGS = (
    "/Users/",
    "/home/",
    "COGNEE",
    "FRED",
    "TWELVEDATA",
    "Courtney",
    "Richardson",
    "controlroom",
    ".internal",
    "app.plane.so",
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
HOME_PATH_RE = re.compile(r"(?:^|[\s(`])(?:~|\$HOME)/")
# The sentences true only while the committed results are tuning-seed results;
# required while results_on_tuning_seed() is true and forbidden after.
BUILT_TUNING_ONLY = (
    "While the committed results are tuning-seed results, DGNTY-10 carries a "
    "fourth status, closes with the final evaluation (M7).",
    "The finding closes only once the final evaluation has run.",
    "On the tuning seed, the shipped allocation leaves the risk model's PR-AUC",
)
BUILT_TUNING_WORDING = (
    "tuning-seed",
    "On the tuning seed",
    "closes with the final evaluation",
    "closes only once the final evaluation has run",
    "has not run",
)
# What v1.0 does with a checkpoint written before it: the one statement, repeated
# word for word in RESULTS.md (Caveats) and HOW-IT-WAS-BUILT.md (the note on DGNTY-35).
CHECKPOINT_CLAIM = (
    "v1.0 refuses a pre-v1.0 checkpoint with a clear error, converting old "
    "checkpoints is deferred to v1.1, and a new model is trained with "
    "`dignity-train`"
)
PRIVACY_ROW_LEAD = (
    "The shipped privacy block drove the risk model to chance and the training "
    "command still saved it as best."
)


def built_rows(text: str) -> dict[int, list[str]]:
    rows = {}
    for line in text.splitlines():
        m = re.search(r"\| DGNTY-(\d+) \|", line)
        if m:
            rows[int(m.group(1))] = [c.strip() for c in line.strip("|").split("|")]
    return rows


def trail_row_problems(text: str) -> list[str]:
    """Every row's PR cell and CI cell equal the pinned id-to-PR map: same PRs in
    the same order, links whose text and address agree, one run per merged PR."""
    problems = []
    rows = built_rows(text)
    if sorted(rows) != list(range(1, 46)):
        return [f"the table has rows {sorted(rows)}, not DGNTY-1 to DGNTY-45"]
    for n, cells in rows.items():
        verdict = "confirmed" if FIRST_DEFECT <= n <= LAST_DEFECT else "not recorded"
        if cells[2] != verdict:
            problems.append(f"DGNTY-{n}: verdict {cells[2]!r}, expected {verdict!r}")
        prs = re.findall(r"\[#(\d+)\]\((\S+?)\)", cells[5])
        if tuple(int(p) for p, _ in prs) != PR_MAP[n]:
            problems.append(f"DGNTY-{n}: PR cell {cells[5]!r} is not {PR_MAP[n]}")
        problems += [
            f"DGNTY-{n}: link #{p} goes to {url}"
            for p, url in prs
            if url != f"{BUILT_REPO}/pull/{p}"
        ]
        runs = re.findall(r"\[run for #(\d+)\]\((\S+?)\)", cells[6])
        merged = tuple(p for p in PR_MAP[n] if p != THIS_PR)
        if not PR_MAP[n]:
            if (cells[5], cells[6]) != ("none", "not recorded"):
                problems.append(f"DGNTY-{n}: no PR, so 'none' and 'not recorded'")
            continue
        if tuple(int(p) for p, _ in runs) != merged:
            problems.append(f"DGNTY-{n}: CI cell {cells[6]!r} has not one run per PR")
        problems += [
            f"DGNTY-{n}: run for #{p} is {url}"
            for p, url in runs
            if int(p) in RUN_FOR
            and url != f"{BUILT_REPO}/actions/runs/{RUN_FOR[int(p)]}"
        ]
        note = f"#{THIS_PR}: not recorded"
        if (THIS_PR in PR_MAP[n]) != (note in cells[6]):
            problems.append(f"DGNTY-{n}: the note {note!r} goes with #{THIS_PR} only")
        rest = re.sub(r"\[run for #\d+\]\(\S+?\)", "", cells[6]).replace(note, "")
        if rest.strip():
            problems.append(f"DGNTY-{n}: CI cell holds more than runs: {cells[6]!r}")
    digest = hashlib.sha256(
        "\n".join(
            f"{n}|{c[0]}|{c[1]}" for n, c in sorted(rows.items()) if n != 8
        ).encode()
    ).hexdigest()
    if digest != FINDING_DIGEST:
        problems.append(
            "the Finding or Area cells changed (digest "
            f"{digest}): if a paraphrase was edited on purpose, check it against "
            "the tracker titles and the review page, then set FINDING_DIGEST to it"
        )
    return problems


def privacy_row_expected(privacy_text: str) -> str | None:
    """The DGNTY-8 finding cell: its paraphrase, then the headline paragraph of the
    curve section of docs/PRIVACY.md word for word (up to its Test line, so the
    replicate-0 qualifiers stay with it), then a link to the section."""
    sentences = sentences_of(
        section_of(privacy_text, "## Privacy-versus-utility curve").split("\n", 1)[1]
    )
    try:
        at = next(i for i, s in enumerate(sentences) if "the shipped allocation" in s)
        end = next(i for i, s in enumerate(sentences) if s.startswith("Test:"))
    except StopIteration:
        return None
    return (
        f"{PRIVACY_ROW_LEAD} {' '.join(sentences[at:end])} See "
        "[Privacy-versus-utility curve](PRIVACY.md#privacy-versus-utility-curve)."
    )


def privacy_row_problems(text: str, privacy_text: str) -> list[str]:
    """The DGNTY-8 finding cell is its paraphrase, then the headline paragraph of
    the curve section of docs/PRIVACY.md word for word, then a link to it."""
    expected = privacy_row_expected(privacy_text)
    if expected is None:
        return [
            "docs/PRIVACY.md no longer has the curve headline paragraph (a sentence "
            "with 'the shipped allocation', then a 'Test:' sentence): the DGNTY-8 "
            "cell of HOW-IT-WAS-BUILT.md copies the paragraph between them"
        ]
    cell = _flat(built_rows(text)[8][0])
    if cell == expected:
        return []
    return [
        "the DGNTY-8 finding cell is not the PRIVACY.md curve headline paragraph "
        f"word for word: the row says {cell!r}, PRIVACY.md gives {expected!r}; "
        "copy the paragraph into the row again"
    ]


def claude_md_bullets(section: str) -> list[str]:
    """The whole bullets of a section: a '- ' line and its indented continuation."""
    bullets, current = [], []
    for line in section.splitlines()[1:]:
        if line.startswith("- "):
            bullets.append("\n".join(current)) if current else None
            current = [line]
        elif current and line.startswith("  "):
            current.append(line)
        else:
            bullets.append("\n".join(current)) if current else None
            current = []
    if current:
        bullets.append("\n".join(current))
    return bullets


def excerpt_problems(text: str, source: str) -> list[str]:
    """The excerpt is whole bullets of .claude/CLAUDE.md, in source order, and the
    prose around it names only what the bullets show."""
    block = text.split("<!--excerpt .claude/CLAUDE.md-->")[1].split("<!--/excerpt-->")[
        0
    ]
    body = "\n".join(x for x in block.splitlines() if not x.startswith("```"))
    taken = claude_md_bullets("\n" + body.strip("\n"))
    if not taken:
        return ["the excerpt holds no bullet"]
    available = claude_md_bullets(
        section_of(source, "## Behavior that will surprise you")
    ) + claude_md_bullets(section_of(source, "## Conventions"))
    problems = [
        f"not a whole bullet of the source: {b[:50]!r}"
        for b in taken
        if b not in available
    ]
    positions = [available.index(b) for b in taken if b in available]
    if positions != sorted(set(positions)):
        problems.append("the bullets are not in source order")
    problems += [f"private string {s!r}" for s in PRIVATE_STRINGS if s in block]
    prose = _flat(section_of(text, "## Agent conventions"))
    count = f"{NUMBER_WORDS.get(len(taken), len(taken)).lower()} whole bullets"
    if count not in prose:
        problems.append(f"the prose does not say {count!r}")
    flat_block = _flat(body)
    for claim, shown in (
        ("is dropped without a warning", "silently dropped"),
        ("Configuration is dataclass-backed", "Config is dataclass-backed"),
        ("its bounds are public parameters", "`bounds` are public parameters"),
        ("the shipped configs stay off it", "shipped configs stay on SystemRandom"),
    ):
        if claim not in prose or shown not in flat_block:
            problems.append(f"prose claim {claim!r} and excerpt line {shown!r}")
    return problems


def ci_check_names(ci_text: str) -> list[str]:
    names = []
    for job, spec in yaml.safe_load(ci_text)["jobs"].items():
        versions = spec.get("strategy", {}).get("matrix", {}).get("python-version")
        names += [f"{job} ({v})" for v in versions] if versions else [job]
    return names


def gates_problems(
    text: str, ci_text: str, precommit_text: str, pyproject_text: str
) -> list[str]:
    """The Gates section names the checks, commands and hooks the workflow and the
    pre-commit file hold, the coverage gate and what it measures (ci.yml and
    pyproject.toml), and makes no claim of enforcement they do not show."""
    gates = section_of(text, "## Gates")
    spans = re.findall(r"`([^`]+)`", gates)
    runs = [
        step["run"]
        for spec in yaml.safe_load(ci_text)["jobs"].values()
        for step in spec["steps"]
        if "run" in step
    ]
    problems = []
    if tuple(ci_check_names(ci_text)) != REQUIRED_CHECKS:
        problems.append(f"ci.yml jobs are {ci_check_names(ci_text)}")
    problems += [f"check not named: {c}" for c in REQUIRED_CHECKS if c not in spans]
    prefixes = ("ruff ", "pytest ", "python ", "dignity-train ", "git status")
    problems += [
        f"command not in ci.yml: {s}"
        for s in spans
        if s.startswith(prefixes) and not any(s in r for r in runs)
    ]
    hooks = [
        h["id"]
        for repo in yaml.safe_load(precommit_text)["repos"]
        for h in repo["hooks"]
    ]
    problems += [f"hook not named: {h}" for h in hooks if h not in spans]
    flat = _flat(gates)
    fail_under = re.findall(r"--cov-fail-under=(\d+)", " ".join(runs))
    if fail_under != ["100"] * len(fail_under) or not fail_under:
        problems.append(f"ci.yml gates coverage at {fail_under}, not 100%")
    run_block = pyproject_text.split("[tool.coverage.run]")[1].split("\n[")[0]
    include = re.findall(r"^include\s*=\s*\[(.*?)\]", run_block, re.M | re.S)
    if include != ['"core/privacy.py"']:
        problems.append(f"pyproject.toml measures {include}, not core/privacy.py")
    for needed in (
        f"{NUMBER_WORDS[len(REQUIRED_CHECKS)].lower()} status checks",
        "The coverage gate is 100% and measures only `core/privacy.py`",
        "`.github/workflows/ci.yml`, `.pre-commit-config.yaml`, the branch protection "
        "settings and the list of merged pull requests on GitHub, the coverage "
        "configuration in `pyproject.toml` and the "
        "doc checkers in `tests/test_docs.py` show",
        f"{MERGED_INTO_DEV} PRs had been merged into `dev`",
        f"the {MERGED_ELSEWHERE} PRs merged into other branches are not counted",
        "forbids deleting the branch and force-pushing",
        "does not require a pull-request review",
        "does not apply to repository administrators",
        "by convention",
        "Nothing in the workflow or in the protection checks it",
        *(f"#{n}" for n in UNTITLED_PRS),
    ):
        if needed not in flat:
            problems.append(f"the Gates section must say: {needed!r}")
    for claimed in ("Every change reaches", "each PR title carries"):
        if claimed in flat:
            problems.append(f"unsupported claim: {claimed}")
    return problems


def private_string_problems(text: str) -> list[str]:
    problems = [f"private string {s!r}" for s in PRIVATE_STRINGS if s in text]
    problems += [f"email-like {m.group(0)!r}" for m in EMAIL_RE.finditer(text)]
    problems += [
        f"home path {m.group(0).strip()!r}" for m in HOME_PATH_RE.finditer(text)
    ]
    return problems


def review_count_problems(text: str) -> list[str]:
    """The review's counts, said once in the prose, equal the pinned constants and
    the verdict column."""
    problems = []
    if len(REVIEWERS) != 5 or DEFECTS != LAST_DEFECT - FIRST_DEFECT + 1:
        problems.append("the constants disagree with each other")
    if RECOMMENDATIONS != LAST_RECOMMENDATION - LAST_DEFECT:
        problems.append("the recommendation count disagrees with the id range")
    review = _flat(section_of(text, "## The adversarial review"))
    for needed in (
        f"Five reviewers ({', '.join(REVIEWERS)})",
        f"{RAW_FINDINGS} raw findings merged into {DEFECTS} defects",
        f"all {DEFECTS} were confirmed",
        f"DGNTY-{FIRST_DEFECT} to DGNTY-{LAST_DEFECT}",
        f"Eight recommendation items (DGNTY-{LAST_DEFECT + 1} to "
        f"DGNTY-{LAST_RECOMMENDATION})",
        f"DGNTY-1 to DGNTY-{FIRST_DEFECT - 1} were known defects",
    ):
        if needed not in review:
            problems.append(f"the adversarial-review section must say: {needed!r}")
    confirmed = [n for n, c in built_rows(text).items() if c[2] == "confirmed"]
    if len(confirmed) != DEFECTS:
        problems.append(f"{len(confirmed)} rows are confirmed, not {DEFECTS}")
    return problems


def how_built_seed_problems(text: str, results_dir: Path = RESULTS_DIR) -> list[str]:
    """Every sentence and status that holds only while the committed results are
    tuning-seed results is required then and forbidden after."""
    flat = _flat(text)
    status = built_rows(text)[10][4]
    if results_on_tuning_seed(results_dir):
        problems = [
            f"missing while the results are on the tuning seed: {s}"
            for s in BUILT_TUNING_ONLY
            if s not in flat
        ]
        if status != STATUSES[2]:
            problems.append(f"DGNTY-10 reads {status!r} on the tuning seed")
        return problems
    return [
        "holds only on the tuning seed, but the results carry the evaluation seed; "
        f"rewrite it: {w}"
        for w in BUILT_TUNING_WORDING
        if w in flat
    ]


def _corrupt(text: str, old: str, new: str) -> str:
    assert text.count(old) >= 1, old
    return text.replace(old, new, 1)


class TestHowItWasBuilt:
    """The review-to-PR trail is complete, honest about open items and sanitised."""

    DEFERRED = {7, 13, 14, 15, 35, 41, 42, 43, 44, 45}  # deferred to v1.1
    OPEN = {8, 27, 40}  # the maintainer's call, partly addressed
    CLOSES_WITH_EVALUATION = {10}  # DGNTY-12 is fixed: its guides are gone or rewritten

    @pytest.fixture(scope="class")
    def text(self):
        return HOW_BUILT.read_text()

    @pytest.fixture(scope="class")
    def privacy_text(self):
        return (ROOT / "docs" / "PRIVACY.md").read_text()

    @pytest.fixture(scope="class")
    def claude_md(self):
        return (ROOT / ".claude/CLAUDE.md").read_text()

    @pytest.fixture(scope="class")
    def ci_text(self):
        return (ROOT / ".github/workflows/ci.yml").read_text()

    @pytest.fixture(scope="class")
    def precommit_text(self):
        return (ROOT / ".pre-commit-config.yaml").read_text()

    @pytest.fixture(scope="class")
    def pyproject_text(self):
        return (ROOT / "pyproject.toml").read_text()

    def _rows(self, text):
        return built_rows(text)

    @staticmethod
    def built_fixture(status):
        """The sentences that hold only on the tuning seed and the DGNTY-10 row, as
        a small document of the test's own."""
        row = (
            "| The synthetic risk task was trivially separable | data | confirmed "
            f"| DGNTY-10 | {status} | none | not recorded |"
        )
        return " ".join(BUILT_TUNING_ONLY) + f"\n\n{row}\n"

    def test_finding_table_covers_every_review_finding_exactly_once(self, text):
        ids = [int(m) for m in re.findall(r"\| DGNTY-(\d+) \|", text)]
        assert sorted(ids) == list(range(1, 46))

    def test_every_row_carries_one_status_from_the_closed_vocabulary(self, text):
        for n, cells in self._rows(text).items():
            assert cells[4] in STATUSES, (
                f"DGNTY-{n} has the status {cells[4]!r}: use one of {STATUSES}"
            )

    def test_a_fixed_finding_links_a_pr_and_never_reads_as_fixed_otherwise(self, text):
        for n, cells in self._rows(text).items():
            status, prs = cells[4], cells[5]
            if status == "fixed":
                assert PR_LINK.search(prs), f"DGNTY-{n} is fixed but links no PR"
            assert (status == "deferred (v1.1)") == (n in self.DEFERRED), (
                f"DGNTY-{n} reads {status!r}: it is deferred exactly when it is in "
                "TestHowItWasBuilt.DEFERRED"
            )
            if n in self.OPEN:
                assert status == "open (maintainer decision)", (
                    f"DGNTY-{n} is in TestHowItWasBuilt.OPEN but reads {status!r}"
                )
            assert (status == STATUSES[2]) == (n in self.CLOSES_WITH_EVALUATION), (
                f"DGNTY-{n} reads {status!r}: a row carries the status "
                f"{STATUSES[2]!r} exactly when its id is in "
                "TestHowItWasBuilt.CLOSES_WITH_EVALUATION; after the final "
                "evaluation set DGNTY-10 to its real status and empty that set"
            )

    def test_open_findings_link_the_prs_that_addressed_part_of_them(self, text):
        rows = self._rows(text)
        for n in self.OPEN:
            assert PR_LINK.search(rows[n][5]), (
                f"DGNTY-{n} is open: its PR cell must link the PRs that "
                "addressed part of it"
            )

    def test_this_prs_own_findings_link_its_pull_request_and_say_what_it_fixes(
        self, text
    ):
        rows = self._rows(text)
        for n in (12, 36):
            assert rows[n][4] == "fixed", f"DGNTY-{n} is fixed by this pull request"
            assert f"({BUILT_REPO}/pull/{THIS_PR})" in rows[n][5], (
                f"DGNTY-{n}: the PR cell must link this pull request (#{THIS_PR})"
            )
        flat = _flat(text)
        notes = sentences_of(flat)
        readme = (
            "the README part is fixed by this pull request (#41), which rewrites "
            "the README"
        )
        for lead in ("DGNTY-12 is fixed in parts", "DGNTY-36 is fixed in parts"):
            note = next((n for n in notes if n.startswith(lead)), "")
            assert readme in note, (
                f"the note {lead!r} (under the table) must say that {readme!r}"
            )
        sentence = (
            "A page cannot cite the CI run of the merge that contains it, so the "
            'cell of this pull request (#41) says "#41: not recorded"; its CI run '
            "is on the pull request page."
        )
        assert sentence in flat, (
            "the paragraph above the table must say, in words that hold before "
            f"and after the merge: {sentence!r}"
        )
        assert "no merge commit" not in flat, (
            "do not say that this pull request has no merge commit: that stops "
            "being true on the merge"
        )

    def test_every_pr_and_ci_link_of_every_row_is_the_pinned_one(self, text):
        problems = trail_row_problems(text)
        assert problems == [], (
            "a PR or CI cell of the finding table differs from PR_MAP / RUN_FOR in "
            "tests/test_docs.py (the pinned map of each id's PRs, from `gh pr "
            "list --state merged`): fix the cell, or, if the map is what changed, "
            f"check it with gh and update the constant: {problems}"
        )

    @pytest.mark.parametrize(
        "old, new",
        [
            ("pull/28)", "pull/12)"),  # wrong PR number behind the same text
            ("[#28](", "[#27]("),  # text and address disagree
            ("actions/runs/42574940294", "actions/runs/42574940295"),  # wrong run
            (
                "| fixed | [#28](https://github.com/crichalchemist/Dignity/pull/28) |",
                "| fixed | [#30](https://github.com/crichalchemist/Dignity/pull/30) |",
            ),  # a PR cell moved to another row
            ("| confirmed | DGNTY-9 |", "| not recorded | DGNTY-9 |"),
            ("42971898695) #41: not recorded", "42971898695) not recorded"),
            ("| none | not recorded |", "| none | |"),
        ],
    )
    def test_a_wrong_or_swapped_link_is_reported(self, text, old, new):
        assert trail_row_problems(_corrupt(text, old, new)) != []

    def test_swapped_run_links_between_two_rows_are_reported(self, text):
        run_a = f"{BUILT_REPO}/actions/runs/{RUN_FOR[28]}"
        run_b = f"{BUILT_REPO}/actions/runs/{RUN_FOR[19]}"
        swapped = text.replace(run_a, "@@").replace(run_b, run_a).replace("@@", run_b)
        assert any("run for" in p for p in trail_row_problems(swapped))

    def test_a_row_with_two_prs_and_one_run_is_reported(self, text):
        shortened = text.replace(
            f" [run for #33]({BUILT_REPO}/actions/runs/{RUN_FOR[33]})", "", 1
        )
        assert shortened != text
        assert trail_row_problems(shortened) != []

    def test_a_paraphrase_changed_to_nonsense_is_reported(self, text):
        corrupted = _corrupt(
            text, "The synthetic risk task was trivially separable", "Nothing happened"
        )
        assert any("digest" in p for p in trail_row_problems(corrupted))

    def test_the_privacy_row_is_the_curve_section_headline_and_adds_nothing(
        self, text, privacy_text
    ):
        problems = privacy_row_problems(text, privacy_text)
        assert problems == [], (
            "the DGNTY-8 row of docs/HOW-IT-WAS-BUILT.md copies the headline "
            "paragraph of the curve section of docs/PRIVACY.md word for word, "
            f"with its replicate-0 qualifiers: {problems}"
        )

    PRIVACY_FIXTURE = (
        "## Privacy-versus-utility curve\n\nThe curve answers one question. "
        "On the tuning seed, the shipped allocation leaves the risk model at "
        "chance: both point estimates are below the reference's and the interval "
        "lies wholly below the non-private reference's interval. At epsilon 30 the "
        "interval lies wholly above the prevalence. These intervals are replicate "
        "0's: one noise draw and one training seed. At epsilon 30 the result rests "
        "mainly on replicate 0's interval. Test: `tests/test_docs.py::T`. Epsilon "
        "is per released value.\n"
    )

    def privacy_row_fixture(self):
        cell = privacy_row_expected(self.PRIVACY_FIXTURE)
        row = f"| {cell} | privacy | confirmed | DGNTY-8 | open | none | not recorded |"
        return row + "\n"

    @pytest.mark.parametrize(
        "old, new",
        [
            ("both point estimates are below", "both intervals are below"),
            (
                "lies wholly below the non-private",
                "lies wholly above the non-private",
            ),
            (
                "These intervals are replicate 0's: one noise draw and one training "
                "seed.",
                "",
            ),
            (
                "At epsilon 30 the result rests mainly on",
                "At epsilon 30 the result rests on",
            ),
            ("See [Privacy-versus-utility curve]", "See [the curve]"),
            (
                "See [Privacy-versus-utility curve]",
                "It never beats chance. See [Privacy-versus-utility curve]",
            ),
        ],
    )
    def test_a_drift_between_the_privacy_row_and_the_curve_section_is_reported(
        self, old, new
    ):
        row = self.privacy_row_fixture()
        assert privacy_row_problems(row, self.PRIVACY_FIXTURE) == []
        assert old in row, old
        corrupted = row.replace(old, new, 1)
        assert privacy_row_problems(corrupted, self.PRIVACY_FIXTURE) != []

    def test_a_privacy_section_without_its_headline_paragraph_is_reported(self):
        row = self.privacy_row_fixture()
        problems = privacy_row_problems(row, "## Privacy-versus-utility curve\n\nA.\n")
        assert problems and "no longer has" in problems[0]

    def test_the_table_is_self_contained_and_names_plane_as_the_private_tracker(
        self, text
    ):
        assert "private tracker" in text
        assert "app.plane.so" not in text
        for n in (46, 47, 48):
            assert f"DGNTY-{n}" in text
            assert f"| DGNTY-{n} |" not in text

    def test_no_metric_number_appears_in_the_document(self, text):
        assert "<!--num" not in text
        assert "<!--results-start-->" not in text
        bare = re.sub(r"\(3\.1[02]\)|v1\.\d|3\.1[02]", " ", text)
        bare = re.sub(r"\(https?://\S+\)", " ", bare)
        assert re.findall(r"\d\.\d", bare) == []
        assert "1.000" not in text

    def test_the_dgnty_40_row_says_a_skeptic_checked_its_steps_as_the_prose_does(
        self, text
    ):
        row = " ".join(self._rows(text)[40])
        assert (
            "a skeptic checked its proposed steps and caught two as unsound" in row
        ), (
            "the DGNTY-40 row must say that a skeptic checked its proposed steps, "
            "as the second story of the review says"
        )
        assert "the skeptic caught two of the proposed steps as unsound" in _flat(
            text
        ), "the second story of the review must say what the skeptic caught"

    def test_the_benchmark_story_is_qualitative(self, text):
        story = section_of(text, "## The adversarial review")
        assert "depth-3 tree" in story
        assert "separated every block" in story
        assert "AUROC" not in story

    def test_the_review_counts_equal_the_pinned_constants_and_the_verdict_column(
        self, text
    ):
        assert review_count_problems(text) == []

    @pytest.mark.parametrize(
        "old, new",
        [
            (
                "42 raw findings merged into 31 defects",
                "42 raw findings merged into 30 defects",
            ),
            ("all 31 were confirmed", "all 30 were confirmed"),
            ("Five reviewers", "Six reviewers"),
            ("DGNTY-7 to DGNTY-37", "DGNTY-7 to DGNTY-36"),
            ("Eight recommendation items", "Seven recommendation items"),
            ("| confirmed | DGNTY-20 |", "| not recorded | DGNTY-20 |"),
        ],
    )
    def test_a_changed_review_count_is_reported(self, text, old, new):
        assert review_count_problems(_corrupt(text, old, new)) != []

    def test_the_agent_conventions_name_only_the_tracked_file(self, text):
        section = section_of(text, "## Agent conventions")
        assert "`.claude/CLAUDE.md`" in section
        assert ".gemini/GEMINI.md" not in text
        assert ".agents/AGENTS.md" not in text
        flat = _flat(section)
        assert "symlinks are not tracked" in flat
        assert "dot-folders carry" not in flat

    def test_the_claude_md_excerpt_is_whole_bullets_in_source_order(
        self, text, claude_md
    ):
        assert excerpt_problems(text, claude_md) == []

    @pytest.mark.parametrize(
        "old, new",
        [
            ("  keep those comments current when you add a field.\n", ""),
            (
                "  The checkpoint stores `available_features` in order (`meta.feature_names`).\n",
                "",
            ),
            ("never derive them from data.", "never derive them."),
            ("  configs stay on SystemRandom.", "  configs stay on SystemRandom"),
            ("four whole bullets", "two whole bullets"),
            ("is dropped without a warning", "is kept"),
            (
                "The privacy block is optional and its bounds are public parameters. ",
                "",
            ),
        ],
    )
    def test_a_cut_reordered_or_overclaiming_excerpt_is_reported(
        self, text, claude_md, old, new
    ):
        assert excerpt_problems(_corrupt(text, old, new), claude_md) != []

    def test_excerpt_bullets_out_of_order_or_from_another_section_are_reported(
        self, text, claude_md
    ):
        block = text.split("<!--excerpt .claude/CLAUDE.md-->")[1].split(
            "<!--/excerpt-->"
        )[0]
        body = block.replace("```text\n", "").replace("```\n", "").strip("\n")
        bullets = claude_md_bullets("\n" + body)
        reordered = text.replace(body, "\n".join(reversed(bullets)))
        assert any("order" in p for p in excerpt_problems(reordered, claude_md))
        foreign = text.replace(
            body, body + "\n- Type hints use modern union syntax (`np.ndarray | None`)"
        )
        assert excerpt_problems(foreign, claude_md) != []

    def test_the_gates_say_only_what_the_workflow_hooks_and_protection_show(
        self, text, ci_text, precommit_text, pyproject_text
    ):
        problems = gates_problems(text, ci_text, precommit_text, pyproject_text)
        assert problems == [], (
            "rewrite the Gates section of docs/HOW-IT-WAS-BUILT.md: it must name "
            "the checks, commands and hooks of ci.yml and .pre-commit-config.yaml, "
            "the coverage gate of ci.yml and pyproject.toml, and claim no "
            f"enforcement they do not show: {problems}"
        )

    @pytest.mark.parametrize(
        "old, new",
        [
            (
                "forbids deleting the branch and force-pushing",
                "forbids deleting the branch",
            ),
            (
                "It does not require a pull-request review and it",
                "It requires a review and it",
            ),
            ("`pytest tests/ --cov --cov-fail-under=100`", "`pytest tests/ -x`"),
            ("`ruff-format`", "`ruff-fmt`"),
            ("`DGNTY-n` by convention", "`DGNTY-n` always"),
            (
                "A PR title carries",
                "Every change reaches the integration branch through a pull request. A PR title carries",
            ),
            (
                "and two of them, #12 and #14, do not carry one",
                "and every PR carries one",
            ),
            (
                "At the time of writing, 28 PRs had been merged into `dev`",
                "Two PRs had been merged into `dev`",
            ),
            (
                "Both branches require four status checks",
                "Both branches require five status checks",
            ),
            (
                "The coverage gate is 100% and measures only `core/privacy.py`",
                "The coverage gate measures the whole package",
            ),
            ("the coverage configuration in `pyproject.toml` and ", ""),
            (
                "the doc checkers in `tests/test_docs.py` show",
                "the branch protection show",
            ),
        ],
    )
    def test_a_gate_claim_the_record_does_not_show_is_reported(
        self, text, ci_text, precommit_text, pyproject_text, old, new
    ):
        corrupted = _corrupt(text, old, new)
        assert gates_problems(corrupted, ci_text, precommit_text, pyproject_text) != []

    def test_a_job_the_workflow_dropped_or_a_version_it_changed_is_reported(
        self, text, ci_text, precommit_text, pyproject_text
    ):
        without_smoke = ci_text.split("  smoke:")[0]
        assert gates_problems(text, without_smoke, precommit_text, pyproject_text)
        other_version = ci_text.replace('["3.10", "3.12"]', '["3.11", "3.12"]')
        assert other_version != ci_text
        assert gates_problems(text, other_version, precommit_text, pyproject_text)

    def test_a_coverage_gate_the_record_changed_is_reported(
        self, text, ci_text, precommit_text, pyproject_text
    ):
        lower = ci_text.replace("--cov-fail-under=100", "--cov-fail-under=90")
        assert lower != ci_text
        assert gates_problems(text, lower, precommit_text, pyproject_text)
        wider = pyproject_text.replace(
            'include = ["core/privacy.py"]', 'include = ["core/*"]'
        )
        assert wider != pyproject_text
        assert gates_problems(text, ci_text, precommit_text, wider)

    def test_the_plane_section_describes_only_what_its_example_shows(self, text):
        section = section_of(text, "## Plane to GitHub")
        flat = _flat(section)
        assert PR_LINK.search(section)
        assert f"({BUILT_REPO}/pull/40)" in section
        assert 'a "Linked to Plane Work Item(s)" comment' in flat
        assert "names DGNTY-27 and DGNTY-36" in flat
        # one example shown, so no quantifier over PRs and no "gets"/"is why" claim
        assert not re.search(
            r"\b(every|each|all|any|always|whenever|gets|is why)\b", flat, re.I
        )

    def test_the_docs_index_links_the_page(self):
        assert "(HOW-IT-WAS-BUILT.md)" in (ROOT / "docs/README.md").read_text()

    def test_docs_never_cite_the_gitignored_internal_notes(self):
        for path in _public_docs():
            assert ".internal/" not in path.read_text(), path

    def test_no_private_string_appears_anywhere_in_the_document(self, text):
        assert private_string_problems(text) == []

    @pytest.mark.parametrize(
        "private",
        [
            "maintainer@example.org",
            "see TWELVEDATA here",
            "/Users/someone/work",
            "/home/someone/work",
            "~/notes",
            "the COGNEE key",
            "Courtney",
        ],
    )
    def test_a_private_string_outside_the_excerpt_is_reported(self, text, private):
        assert private_string_problems(text + f"\nNote: {private}\n") != []

    def test_the_old_checkpoint_row_says_what_v1_0_does_and_what_is_deferred(
        self, text
    ):
        import inspect

        from train import checkpoint

        row = " ".join(self._rows(text)[35])
        for part in (
            "v1.0 refuses a pre-v1.0 checkpoint with a clear error",
            "converting old checkpoints is deferred to v1.1",
        ):
            assert part in row, f"the DGNTY-35 row must say: {part!r}"
        flat = _flat(text)
        assert "deferred only in part" in flat, (
            "the note under the table must say that DGNTY-35 is deferred only in part"
        )
        results = _flat((ROOT / "docs" / "RESULTS.md").read_text())
        for name, page in (("HOW-IT-WAS-BUILT.md", flat), ("RESULTS.md", results)):
            assert CHECKPOINT_CLAIM in page, (
                f"docs/{name} must say, word for word, as the one statement of what "
                f"v1.0 does with an old checkpoint: {CHECKPOINT_CLAIM!r}"
            )
            for stale in (
                "rejects the old checkpoint format at export",
                "retraining the old checkpoints",
                "retraining or converting",
                "retrain with `dignity-train`",
            ):
                assert stale not in page, f"docs/{name} still says {stale!r}"
        assert "Checkpoints written before v1.0 are unsupported" in results
        assert "DGNTY-35" in results
        architecture = (ROOT / "docs" / "ARCHITECTURE.md").read_text()
        assert "UnsupportedCheckpointError` (DGNTY-35)" in architecture
        assert "raise UnsupportedCheckpointError" in inspect.getsource(checkpoint), (
            "train.checkpoint no longer raises UnsupportedCheckpointError: the "
            "claim that v1.0 refuses an old checkpoint with a clear error is false"
        )

    def test_m7_is_defined_once_and_v1_1_items_are_planned_not_present(self, text):
        flat = _flat(text)
        definition = "M7 is the release milestone in which the final evaluation runs"
        assert flat.count(definition) == 1
        assert flat.index(definition) < flat.index("| Finding |")
        assert (
            "v1.1 is the planned next release: a deferred item is planned for it "
            "and does not exist in v1.0" in flat
        )
        assert "the review's 45 filed items" not in flat
        assert (
            "The review filed DGNTY-7 to DGNTY-45; DGNTY-1 to DGNTY-6 were filed "
            "before it." in flat
        )

    def test_every_sentence_true_only_on_the_tuning_seed_follows_the_results_seed(
        self, text
    ):
        problems = how_built_seed_problems(text)
        assert problems == [], (
            "rewrite these sentences of docs/HOW-IT-WAS-BUILT.md and the DGNTY-10 "
            "status: the test requires them while risk.json carries the tuning "
            "seed and rejects them once it carries the evaluation seed (then "
            "DGNTY-10 takes its real status, and TestHowItWasBuilt."
            f"CLOSES_WITH_EVALUATION is emptied): {problems}"
        )

    def test_a_tuning_seed_sentence_is_required_while_the_results_use_that_seed(
        self, tmp_path
    ):
        tuning = seed_dir(tmp_path, EvalConfig().tuning_seed)
        doc = self.built_fixture(STATUSES[2])
        assert how_built_seed_problems(doc, tuning) == []
        for sentence in BUILT_TUNING_ONLY:
            gone = doc.replace(sentence, "")
            assert gone != doc
            assert how_built_seed_problems(gone, tuning) != [], sentence
        fixed = self.built_fixture("fixed")
        assert how_built_seed_problems(fixed, tuning) != []

    def test_a_tuning_seed_sentence_is_rejected_once_the_results_carry_the_eval_seed(
        self, tmp_path
    ):
        evaluation = seed_dir(tmp_path, EvalConfig().eval_seed)
        doc = self.built_fixture(STATUSES[2])
        problems = how_built_seed_problems(doc, evaluation)
        for wording in BUILT_TUNING_WORDING[:4]:
            assert any(wording in p for p in problems), wording
        rewritten = doc
        for sentence in BUILT_TUNING_ONLY[:2]:
            rewritten = rewritten.replace(sentence, "")
        rewritten = rewritten.replace(
            "| closes with the final evaluation (M7) |", "| fixed |"
        ).replace("On the tuning seed", "On the evaluation seed")
        assert how_built_seed_problems(rewritten, evaluation) == []

    def test_a_has_not_run_sentence_is_rejected_once_the_results_carry_the_eval_seed(
        self, tmp_path
    ):
        evaluation = seed_dir(tmp_path, EvalConfig().eval_seed)
        left_over = self.built_fixture("fixed") + "The final evaluation has not run.\n"
        assert any(
            "has not run" in p for p in how_built_seed_problems(left_over, evaluation)
        )
