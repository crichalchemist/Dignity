"""Rewrite every quoted number in a doc from the committed results JSON.

A quoted number is a token directly followed by a marker
``<!--num <file>.json <dotted.path> <fmt>-->``. This module is the single
definition of that grammar: tests/test_docs.py imports it to check the docs, and
``refresh`` uses it to rewrite them, so the two cannot drift. Standard library
only, so it runs as ``python scripts/refresh_doc_numbers.py`` from anywhere.

    python scripts/refresh_doc_numbers.py README.md docs/RESULTS.md

Only the number inside a token changes. The wrappers around it (brackets, bold,
table pipes, closing punctuation), markers and zone tags inside code, and all
other text stay as written. A marker that cannot be resolved stops the run with
a non-zero exit and nothing is written.
"""

import argparse
import json
import re
import sys
from pathlib import Path

FENCE_RE = re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1", re.DOTALL | re.MULTILINE)
CODE_RE = re.compile(r"`[^`\n]*`")
NUM_RE = re.compile(r"([^\s>]+)\s*<!--num (\S+?)\.json (\S+) (\S+?)-->")
# The wrappers a marker's token may carry around its number: Markdown emphasis,
# table pipes, a CI bracket or parenthesis, a glued "/" and closing punctuation.
TOKEN_RE = re.compile(
    r"(?P<prefix>[\[(*_~|/]*)"
    r"(?P<core>n/a|[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?)"
    r"(?P<suffix>[\])*_~|,.;:/]*)"
)
DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "results" / "v1.0"


def split_token(token: str, fmt: str = "") -> tuple[str, str, str]:
    """Split a marker's token into (prefix, core, suffix).

    core is the number text, the literal n/a, or (for an "s" format) the whole
    token. prefix and suffix are the Markdown or punctuation wrappers around it.
    check_numbers compares only core, and refresh rewrites only core, so that
    prefix + new_core + suffix keeps every wrapper. Raises ValueError for a
    token that holds no single number.
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


def resolve(data, dotted: str):
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


def render(value, fmt: str) -> str:
    """The text a marker's token must show: a JSON null is n/a for every fmt."""
    return "n/a" if value is None else format(value, fmt)


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


def rewrite(md_path: Path, results_dir: Path) -> tuple[str, int]:
    """The doc text with every stale token rewritten, and how many were.

    Raises ValueError listing every marker that cannot be resolved or read.
    """
    text = Path(md_path).read_text()
    live = strip_code(text)
    loaded: dict[str, object] = {}
    problems = []
    edits = []
    for match in NUM_RE.finditer(live):
        token, name, dotted, fmt = match.groups()
        where = f"{Path(md_path).name}:{text.count(chr(10), 0, match.start()) + 1}"
        if name not in loaded:
            path = Path(results_dir) / f"{name}.json"
            loaded[name] = json.loads(path.read_text()) if path.exists() else None
        if loaded[name] is None:
            problems.append(f"{where}: no such results file {name}.json")
            continue
        try:
            shown = render(resolve(loaded[name], dotted), fmt)
        except (KeyError, IndexError):
            problems.append(f"{where}: no such path {dotted} in {name}.json")
            continue
        except (TypeError, ValueError) as error:
            problems.append(
                f"{where}: {dotted} cannot be formatted with {fmt}: {error}"
            )
            continue
        try:
            prefix, core, suffix = split_token(token, fmt)
        except ValueError as error:
            problems.append(f"{where}: {error}")
            continue
        if core != shown:
            start, end = match.span(1)
            edits.append((start, end, prefix + shown + suffix))
    if problems:
        raise ValueError("\n".join(problems))
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    return text, len(edits)


def refresh(md_path: Path, results_dir: Path) -> int:
    """Rewrite the stale tokens of one doc in place; return how many changed."""
    text, count = rewrite(md_path, results_dir)
    if count:
        Path(md_path).write_text(text)
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("docs", nargs="+", type=Path, help="Markdown files")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)

    rewritten = []
    failed = False
    for doc in args.docs:
        try:
            rewritten.append((doc, *rewrite(doc, args.results)))
        except ValueError as error:
            failed = True
            print(f"{doc}: not refreshed\n{error}", file=sys.stderr)
    if failed:
        return 1
    for doc, text, count in rewritten:
        if count:
            doc.write_text(text)
        print(f"{doc}: {count} rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
