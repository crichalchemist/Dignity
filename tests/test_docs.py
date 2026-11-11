"""Public docs must not reintroduce claims or APIs this project removed."""

import ast
import dataclasses
import importlib
import json
import os
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


FENCE_RE = re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1", re.DOTALL | re.MULTILINE)
FIX = ROOT / "tests" / "fixtures" / "docs_numbers"
FIXR = ROOT / "tests" / "fixtures" / "docs_refs"
RESULTS_DIR = ROOT / "results" / "v1.0"
NUM_RE = re.compile(r"([^\s>]+)\s*<!--num (\S+?)\.json (\S+) (\S+?)-->")
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
CODE_RE = re.compile(r"`[^`\n]*`")
# The wrappers a marker's token may carry around its number: Markdown emphasis,
# table pipes, a CI bracket or parenthesis, a glued "/" and closing punctuation.
TOKEN_RE = re.compile(
    r"(?P<prefix>[\[(*_~|/]*)"
    r"(?P<core>n/a|[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?)"
    r"(?P<suffix>[\])*_~|,.;:/]*)"
)
BANNER = (
    "<!--provisional-->**Provisional: numbers are from the tuning seed; "
    "the final evaluation has not run.**"
)


def split_token(token: str, fmt: str = "") -> tuple[str, str, str]:
    """Split a marker's token into (prefix, core, suffix).

    core is the number text, the literal n/a, or (for an "s" format) the whole
    token. prefix and suffix are the Markdown or punctuation wrappers around it.
    check_numbers compares only core, and the M6.6 refresh script must import
    this helper and rewrite only core, so that prefix + new_core + suffix keeps
    every wrapper. Raises ValueError for a token that holds no single number.
    """
    if fmt[-1:] == "s":
        return "", token, ""
    match = TOKEN_RE.fullmatch(token)
    if match:
        return match["prefix"], match["core"], match["suffix"]
    if re.search(r"\d\s*[–—-]\s*\.?\d", token):
        raise ValueError(
            f"{token} is a range: write an interval in the bracket form "
            "[low, high], one marker per bound"
        )
    raise ValueError(f"cannot read a number from {token}")


def _resolve(data, dotted: str):
    """Index dicts by key and lists by non-negative integer; KeyError if absent."""
    for part in dotted.split("."):
        if isinstance(data, dict):
            data = data[part]
        elif isinstance(data, list):
            if not part.isdigit():
                raise KeyError(part)
            data = data[int(part)]
        else:
            raise KeyError(part)
    return data


def _blank(text: str, span: tuple[int, int]) -> str:
    """Replace a span with spaces, keeping newlines so offsets stay valid."""
    start, end = span
    return text[:start] + re.sub(r"[^\n]", " ", text[start:end]) + text[end:]


def strip_code(text: str) -> str:
    """Blank fenced blocks and inline code spans: markers and zone tags there are
    documentation, not live markers."""
    for pattern in (FENCE_RE, CODE_RE):
        for match in list(pattern.finditer(text)):
            text = _blank(text, match.span())
    return text


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
            shown = "n/a" if value is None else format(value, fmt)
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
            return [f"{name}: Provisional left in a doc on the evaluation seed"]
        return []
    if BANNER not in text:
        return [f"{name}: missing the banner line {BANNER}"]
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
        for name in ("README.md", "docs/RESULTS.md"):
            text = (ROOT / name).read_text()
            assert banner_problems(name, text, final) == []


class TestResultsComeFromOneRun:
    def test_a_half_regenerated_results_set_is_not_published(self):
        assert run_identity_problems() == []

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


SYMBOL_RE = re.compile(r"^(core|data|models|train|export|eval)(\.\w+)+$")
PATH_EXT = (".py", ".yaml", ".md", ".json", ".png", ".ipynb")
# Paths a doc may name although they are gitignored and so absent from a checkout.
ALLOWED_PREFIXES = ("runs/", "checkpoints/")
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


class TestArchitectureDoc:
    """docs/ARCHITECTURE.md describes the code as it is, section by section."""

    ARCH = ROOT / "docs" / "ARCHITECTURE.md"

    def test_data_contract_table_lists_every_generator_column(self):
        from core.config import SyntheticConfig
        from data.source.synthetic import SyntheticGenerator

        frame = SyntheticGenerator(SyntheticConfig(n_blocks=2, block_len=20)).generate()
        columns = list(frame.columns)
        assert len(columns) == 10, columns
        text = self.ARCH.read_text()
        for col in columns:
            assert f"| `{col}` |" in text, col

    def test_doc_names_every_implemented_anomaly_type(self):
        from data.source.anomalies import ANOMALY_TYPES

        text = self.ARCH.read_text()
        assert all(f"`{t}`" in text for t in ANOMALY_TYPES)

    def test_doc_makes_no_trading_or_agent_claim(self):
        # Regression guard: green before the rewrite. Word-bounded so that
        # "support" and UnsupportedCheckpointError stay writable; the REMOVED
        # list already covers the forecasting phrases.
        text = self.ARCH.read_text().lower()
        assert re.search(r"\b(ppo|a3c)\b", text) is None
        assert "trading agent" not in text

    def test_architecture_doc_has_no_pre_v1_names(self):
        # Regression guard: green before the rewrite. Every name here was in
        # the old ARCHITECTURE.md or SIGNALS.md and is defined nowhere in the code.
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

    def test_scope_limits_of_v1_are_stated(self):
        # data.source is synthetic-only and nothing releases at inference time;
        # a doc that implied otherwise would describe paths that do not exist.
        text = self.ARCH.read_text()
        assert "`data.source`" in text and "`CryptoSource`" in text
        assert "DGNTY-27" in text
        assert "DGNTY-35" in text
