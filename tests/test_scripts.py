"""scripts/refresh_doc_numbers.py: the one place that rewrites quoted numbers."""

import json

import pytest

from tests.test_docs import NUM_RE, _refresh, check_numbers

refresh = _refresh.refresh
main = _refresh.main
ZONE = "<!--results-start-->{}<!--results-end-->"


def _write(tmp_path, text, name="a.md", **results):
    md = tmp_path / name
    md.write_text(text)
    (tmp_path / "r.json").write_text(json.dumps(results or {"m": {"v": 0.8412}}))
    return md


class TestRefreshDocNumbers:
    def test_refresh_rewrites_a_stale_number_and_leaves_matching_ones(self, tmp_path):
        md = tmp_path / "a.md"
        md.write_text(
            "<!--results-start-->0.10 <!--num r.json m.v .2f--> "
            "0.50 <!--num r.json m.w .2f--><!--results-end-->"
        )
        (tmp_path / "r.json").write_text('{"m": {"v": 0.8412, "w": 0.5}}')
        assert refresh(md, tmp_path) == 1
        text = md.read_text()
        assert "0.84 <!--num r.json m.v .2f-->" in text
        assert "0.50 <!--num r.json m.w .2f-->" in text
        assert check_numbers(md, tmp_path) == []

    def test_refresh_keeps_the_zone_tags_the_checker_needs(self, tmp_path):
        md = _write(tmp_path, ZONE.format("0.10 <!--num r.json m.v .2f-->"))
        refresh(md, tmp_path)
        text = md.read_text()
        assert text.startswith("<!--results-start-->")
        assert text.endswith("<!--results-end-->")

    def test_a_second_refresh_changes_nothing(self, tmp_path):
        md = _write(tmp_path, "0.10 <!--num r.json m.v .2f--> and **0.2** x")
        assert refresh(md, tmp_path) == 1
        first = md.read_text()
        assert refresh(md, tmp_path) == 0
        assert md.read_text() == first

    def test_refresh_keeps_every_wrapper_around_a_token(self, tmp_path):
        text = (
            "[0.10 <!--num r.json m.v .2f-->, 0.20 <!--num r.json m.v .2f-->] "
            "**0.30 <!--num r.json m.v .2f-->** "
            "|0.40 <!--num r.json m.v .2f-->| "
            "~0.50 <!--num r.json m.v .1%-->. "
            "0.60<!--num r.json m.v .2f-->;"
        )
        md = _write(tmp_path, text)
        assert refresh(md, tmp_path) == 6
        assert md.read_text() == (
            "[0.84 <!--num r.json m.v .2f-->, 0.84 <!--num r.json m.v .2f-->] "
            "**0.84 <!--num r.json m.v .2f-->** "
            "|0.84 <!--num r.json m.v .2f-->| "
            "~84.1% <!--num r.json m.v .1%-->. "
            "0.84<!--num r.json m.v .2f-->;"
        )

    def test_markers_inside_code_are_documentation_and_stay_as_written(self, tmp_path):
        text = (
            "```\n0.10 <!--num r.json m.v .2f-->\n```\n"
            "and `0.10 <!--num r.json m.v .2f-->` and "
            "`<!--results-start-->` here"
        )
        md = _write(tmp_path, text)
        assert refresh(md, tmp_path) == 0
        assert md.read_text() == text

    def test_a_json_null_is_written_as_n_a_in_both_directions(self, tmp_path):
        md = _write(
            tmp_path,
            "0.50 <!--num r.json m.none .2f--> n/a <!--num r.json m.v .2f-->",
            m={"none": None, "v": 0.5},
        )
        assert refresh(md, tmp_path) == 2
        assert md.read_text() == (
            "n/a <!--num r.json m.none .2f--> 0.50 <!--num r.json m.v .2f-->"
        )
        assert check_numbers(md, tmp_path) == []

    def test_a_marker_whose_path_does_not_resolve_fails_and_leaves_the_file(
        self, tmp_path
    ):
        text = "0.10 <!--num r.json m.v .2f--> 0.2 <!--num r.json m.nope .2f-->"
        md = _write(tmp_path, text)
        with pytest.raises(ValueError, match="m.nope"):
            refresh(md, tmp_path)
        assert md.read_text() == text

    def test_a_marker_naming_a_missing_results_file_fails_and_leaves_the_file(
        self, tmp_path
    ):
        text = "0.10 <!--num gone.json m.v .2f-->"
        md = _write(tmp_path, text)
        with pytest.raises(ValueError, match="gone.json"):
            refresh(md, tmp_path)
        assert md.read_text() == text

    def test_main_exits_non_zero_and_writes_no_file_when_one_marker_is_bad(
        self, tmp_path, capsys
    ):
        good = _write(tmp_path, "0.10 <!--num r.json m.v .2f-->", name="good.md")
        bad_text = "0.10 <!--num r.json m.nope .2f-->"
        bad = tmp_path / "bad.md"
        bad.write_text(bad_text)
        code = main([str(good), str(bad), "--results", str(tmp_path)])
        assert code != 0
        assert good.read_text() == "0.10 <!--num r.json m.v .2f-->"
        assert bad.read_text() == bad_text
        assert "m.nope" in capsys.readouterr().err

    def test_main_reports_how_many_tokens_each_file_rewrote(self, tmp_path, capsys):
        md = _write(tmp_path, "0.10 <!--num r.json m.v .2f-->")
        assert main([str(md), "--results", str(tmp_path)]) == 0
        assert capsys.readouterr().out.strip() == f"{md}: 1 rewritten"

    def test_the_checker_and_refresh_read_one_marker_grammar(self):
        # one definition: a drift between the two would let a refresh write
        # tokens the checker then rejects, or the reverse
        assert NUM_RE is _refresh.NUM_RE
        from tests import test_docs

        assert test_docs.split_token is _refresh.split_token
        assert test_docs._resolve is _refresh.resolve
        assert test_docs.strip_code is _refresh.strip_code
