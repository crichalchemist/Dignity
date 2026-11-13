"""scripts/refresh_doc_numbers.py: the one place that rewrites quoted numbers."""

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from core.config import DignityConfig
from tests.test_docs import NUM_RE, _refresh, check_numbers
from tests.test_release import _passing

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

    def test_refresh_keeps_a_prefix_and_a_suffix_glued_to_the_token(self, tmp_path):
        md = _write(
            tmp_path,
            "**0.10**<!--num r.json m.v .2f--> (0.10),<!--num r.json m.v .2f-->",
        )
        assert refresh(md, tmp_path) == 2
        assert md.read_text() == (
            "**0.84**<!--num r.json m.v .2f--> (0.84),<!--num r.json m.v .2f-->"
        )

    def test_a_value_the_format_cannot_render_fails_and_leaves_the_file(self, tmp_path):
        text = "0.10 <!--num r.json m.v .2f-->"
        md = _write(tmp_path, text, m={"v": "text"})
        with pytest.raises(ValueError, match="m.v cannot be formatted with .2f"):
            refresh(md, tmp_path)
        assert md.read_text() == text

    @pytest.mark.parametrize(
        ("token", "message"),
        [("abc", "cannot read a number from abc"), ("0.7–0.9", "is a range")],
    )
    def test_a_token_that_holds_no_single_number_fails_and_leaves_the_file(
        self, tmp_path, token, message
    ):
        text = f"{token} <!--num r.json m.v .2f-->"
        md = _write(tmp_path, text)
        with pytest.raises(ValueError, match=message):
            refresh(md, tmp_path)
        assert md.read_text() == text

    def test_the_default_results_directory_is_the_committed_one(self):
        from tests.test_docs import ROOT

        assert _refresh.DEFAULT_RESULTS == ROOT / "results" / "v1.0"
        assert _refresh.DEFAULT_RESULTS.is_dir()

    def test_an_error_names_the_line_of_the_marker(self, tmp_path):
        md = _write(tmp_path, "intro\n0.10 <!--num r.json m.nope .2f-->")
        with pytest.raises(ValueError, match=r"a\.md:2: no such path m\.nope"):
            refresh(md, tmp_path)

    def test_a_fence_is_closed_only_by_its_own_kind_of_fence(self, tmp_path):
        for text in (
            "```\n~~~\n0.10 <!--num r.json m.v .2f-->\n```\n",
            "~~~\n```\n0.10 <!--num r.json m.v .2f-->\n~~~\n",
        ):
            md = _write(tmp_path, text)
            assert refresh(md, tmp_path) == 0
            assert md.read_text() == text

    def test_every_unresolvable_marker_is_reported_not_only_the_first(self, tmp_path):
        md = _write(
            tmp_path,
            "0.1 <!--num r.json m.nope .2f--> 0.2 <!--num r.json m.nada .2f-->",
        )
        with pytest.raises(ValueError) as error:
            refresh(md, tmp_path)
        assert "m.nope" in str(error.value) and "m.nada" in str(error.value)


REHEARSE = Path(__file__).resolve().parent.parent / "scripts" / "rehearse.py"


def _load_rehearse():
    spec = importlib.util.spec_from_file_location("rehearse", REHEARSE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _no_generation(monkeypatch):
    import eval.run

    def boom(*a, **k):
        raise AssertionError("data was generated for a refused rehearsal")

    monkeypatch.setattr(eval.run, "main", boom)


def _rehearse(monkeypatch, tmp_path):
    module = _load_rehearse()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


class TestRehearse:
    @pytest.mark.parametrize("seeds", ["1008", "2026", "2027", "999"])
    def test_a_seed_outside_the_selection_seeds_is_refused_before_any_data_is_generated(  # noqa: E501
        self, monkeypatch, tmp_path, seeds
    ):
        rehearse = _rehearse(monkeypatch, tmp_path)
        _no_generation(monkeypatch)
        argv = ["--stage", "variants", "--variant", "base", "--seeds", seeds]
        assert rehearse.main(argv) == 2
        assert not (tmp_path / "runs" / "rehearsal").exists()

    @pytest.mark.parametrize("seeds", ["1000,1000", "1000,1001,1000"])
    def test_duplicate_seeds_are_refused_before_any_data_is_generated(
        self, monkeypatch, tmp_path, seeds
    ):
        rehearse = _rehearse(monkeypatch, tmp_path)
        _no_generation(monkeypatch)
        argv = ["--stage", "variants", "--variant", "base", "--seeds", seeds]
        assert rehearse.main(argv) == 2
        assert not (tmp_path / "runs" / "rehearsal").exists()

    def test_an_unknown_variant_exits_2(self, monkeypatch, tmp_path):
        rehearse = _rehearse(monkeypatch, tmp_path)
        _no_generation(monkeypatch)
        argv = ["--stage", "variants", "--variant", "wide", "--seeds", "1000"]
        assert rehearse.main(argv) == 2
        assert not (tmp_path / "runs" / "rehearsal").exists()

    def test_the_base_variant_has_no_overrides(self):
        assert _load_rehearse().VARIANTS["base"] == {}

    def test_every_write_lands_under_runs_rehearsal(self, monkeypatch, tmp_path):
        rehearse = _rehearse(monkeypatch, tmp_path)
        path = rehearse.write_seed_config("variants", "cap10", 1000)
        root = tmp_path / "runs" / "rehearsal"
        assert root in path.parents
        folder = rehearse.seed_dir("variants", "cap10", 1000)
        assert folder == root / "variants" / "cap10" / "seed_1000"
        written = yaml.safe_load(path.read_text())
        assert written["eval"]["out_dir"] == str(folder / "results")
        assert written["eval"]["img_dir"] == str(folder / "img")
        assert written["train"]["checkpoint_dir"] == str(folder / "ckpt")
        for key in (
            written["eval"]["out_dir"],
            written["eval"]["img_dir"],
            written["train"]["checkpoint_dir"],
        ):
            assert not key.startswith(("results/", "docs/img", "checkpoints"))
        files = {p for p in tmp_path.rglob("*") if p.is_file()}
        assert all(root in p.parents for p in files)

    def test_each_variant_changes_at_most_one_forecast_or_shared_factor_and_one_policy_step(  # noqa: E501
        self,
    ):
        variants = _load_rehearse().VARIANTS
        assert set(variants) == {"base", "cold", "cap10", "cap3", "narrow", "lr1e-3"}
        allowed = {
            "train.forecast_warm_start",
            "train.forecast_pos_weight_cap",
            "model.hidden_size",
            "train.lr",
        }
        for name, overrides in variants.items():
            factors = set(overrides) - {"policy.multiplier_lr"}
            assert len(factors) <= 1, name
            assert factors <= allowed, name

    def test_the_rehearsal_config_is_the_hard_preset_plus_the_overrides_only(
        self, monkeypatch, tmp_path
    ):
        rehearse = _rehearse(monkeypatch, tmp_path)
        preset = DignityConfig.from_yaml(str(rehearse.PRESET))

        def written(variant):
            path = rehearse.write_seed_config("variants", variant, 1000)
            return DignityConfig.from_yaml(str(path))

        cold = written("cold")
        assert cold.data == preset.data
        assert cold.policy.multiplier_lr == 0.05
        assert cold.train.forecast_warm_start is False
        base = written("base")
        for block in ("model", "train", "policy"):
            a, b = asdict(getattr(base, block)), asdict(getattr(preset, block))
            if block == "train":
                a.pop("checkpoint_dir"), b.pop("checkpoint_dir")
            assert a == b, block
        assert written("cap3").policy.multiplier_lr == preset.policy.multiplier_lr


def _seed_fixtures(**over):
    """The passing release fixtures plus the series rehearse.py reports."""
    files = _passing()
    files["forecast"]["metrics"]["seed_spread"].update(
        val_lift_upper={"values": [1.2, 1.3, 1.4]},
        risk_val_lift_upper={"values": [1.0, 1.1, 1.2]},
        best_epoch={"values": [7, 8, 9]},
    )
    files["policy"]["metrics"]["seed_spread"].update(
        val_neg_cost={"values": [-0.2, -0.21, -0.19]},
        review_multiplier={"values": [1.0, 1.1, 1.2]},
    )
    for key, value in over.items():
        name, _, path = key.partition("__")
        node = files[name]
        *parents, leaf = path.split("__")
        for part in parents:
            node = node[part]
        node[leaf] = value
    return files


BEST_METRICS = (0.41, 0.43, 0.45)


class _FakeEval:
    """A boundary stand-in for eval.run.main: writes small result fixtures."""

    def __init__(self, files=None, status=0):
        self.files = files or {}
        self.status = status
        self.calls = []

    def __call__(self, argv):
        import torch

        from train.checkpoint import CheckpointMeta, save_checkpoint

        self.calls.append(list(argv))
        config = yaml.safe_load(Path(argv[argv.index("--config") + 1]).read_text())
        out = Path(config["eval"]["out_dir"])
        out.mkdir(parents=True)
        files = self.files or _seed_fixtures()
        for name in ("risk", "forecast", "policy"):
            if name in files:
                (out / f"{name}.json").write_text(json.dumps(files[name]))
        seed = argv[argv.index("--data-seed") + 1]
        model = torch.nn.Linear(1, 1)
        meta = CheckpointMeta(2, "risk", 1, ["volume"], {}, [0.0], [1.0], {})
        for k, best in enumerate(BEST_METRICS):
            run = Path(config["train"]["checkpoint_dir"]) / f"eval_tuning_{seed}"
            path = run / "risk" / f"train_seed_{config['seed'] + k}" / "best.pt"
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            scaler = torch.amp.GradScaler("cpu", enabled=False)
            save_checkpoint(path, model, opt, scaler, meta, 1, {}, best)
        return self.status


def _summary(monkeypatch, tmp_path, fake, variant="base", seeds="1000", extra=()):
    rehearse = _rehearse(monkeypatch, tmp_path)
    import eval.run

    monkeypatch.setattr(eval.run, "main", fake)
    argv = ["--stage", "variants", "--variant", variant, "--seeds", seeds, *extra]
    code = rehearse.main(argv)
    path = rehearse.seed_dir("variants", variant, 1000).parent / "summary.json"
    return code, json.loads(path.read_text())


class TestRehearseRun:
    def test_c4_fails_when_a_seed_collapses_to_never_reviewing(
        self, monkeypatch, tmp_path
    ):
        rates = {
            "policy__metrics__seed_spread__review_rate": {"values": [0.01, 0.05, 0.05]}
        }
        code, summary = _summary(
            monkeypatch, tmp_path, _FakeEval(_seed_fixtures(**rates))
        )
        assert code == 0
        assert summary["seeds"]["1000"]["C4"] is False

    def test_c4_passes_at_exactly_the_review_rate_bar(self, monkeypatch, tmp_path):
        rates = {
            "policy__metrics__seed_spread__review_rate": {"values": [0.02, 0.05, 0.05]}
        }
        _, summary = _summary(monkeypatch, tmp_path, _FakeEval(_seed_fixtures(**rates)))
        assert summary["seeds"]["1000"]["C4"] is True
        assert summary["seeds"]["1000"]["C2"] is True
        assert summary["seeds"]["1000"]["C3"] is True

    def test_the_summary_carries_each_training_seeds_risk_validation_score(
        self, monkeypatch, tmp_path
    ):
        _, summary = _summary(monkeypatch, tmp_path, _FakeEval())
        assert summary["seeds"]["1000"]["risk_val_pr_auc"] == pytest.approx(
            BEST_METRICS
        )

    def test_the_summary_reports_the_forecast_and_policy_series_it_was_given(
        self, monkeypatch, tmp_path
    ):
        _, summary = _summary(monkeypatch, tmp_path, _FakeEval())
        entry = summary["seeds"]["1000"]
        assert entry["forecast_val_lift_upper"] == [1.2, 1.3, 1.4]
        assert entry["forecast_floor"] == [1.0, 1.1, 1.2]
        assert entry["forecast_best_epoch"] == [7, 8, 9]
        assert entry["review_multiplier"] == [1.0, 1.1, 1.2]
        assert entry["policy_val_neg_cost"] == [-0.2, -0.21, -0.19]
        assert entry["review_rate"] == [0.03, 0.04, 0.05]
        assert entry["uncapped_review_share"] == [0.06, 0.08, 0.1]
        assert entry["raised"] is None

    def test_a_raised_forecast_fit_fails_c3_and_is_recorded_by_seed(
        self, monkeypatch, tmp_path
    ):
        raised = {"error": "ForecastNotEligibleError", "message": "m", "train_seed": 2}
        files = _seed_fixtures()
        files["forecast"] = {"metrics": {"raised": raised}}
        fake = _FakeEval(files, status=1)
        code, summary = _summary(monkeypatch, tmp_path, fake, seeds="1000,1001")
        assert code == 0
        assert len(fake.calls) == 2
        entry = summary["seeds"]["1000"]
        assert entry["C3"] is False
        assert entry["raised"]["forecast"]["train_seed"] == 2
        assert entry["forecast_val_lift_upper"] is None
        assert entry["forecast_floor"] is None
        assert entry["C2"] is True
        assert "1001" in summary["seeds"]

    def test_a_raised_policy_fit_fails_c4_and_hides_its_series(
        self, monkeypatch, tmp_path
    ):
        raised = {"error": "PolicyCollapseError", "message": "m", "train_seed": 1}
        files = _seed_fixtures()
        files["policy"] = {"metrics": {"raised": raised}}
        _, summary = _summary(monkeypatch, tmp_path, _FakeEval(files, status=1))
        entry = summary["seeds"]["1000"]
        assert entry["C4"] is False
        assert entry["raised"]["policy"]["error"] == "PolicyCollapseError"
        assert entry["review_rate"] is None
        assert entry["review_multiplier"] is None

    def test_a_task_that_was_not_run_has_no_verdict_and_no_series(
        self, monkeypatch, tmp_path
    ):
        files = {"risk": _seed_fixtures()["risk"]}
        _, summary = _summary(
            monkeypatch, tmp_path, _FakeEval(files), extra=("--tasks", "risk")
        )
        entry = summary["seeds"]["1000"]
        assert (entry["C2"], entry["C3"], entry["C4"]) == (True, None, None)
        assert entry["review_rate"] is None
        assert entry["forecast_best_epoch"] is None

    def test_a_refused_eval_run_is_an_error_entry_and_exit_1(
        self, monkeypatch, tmp_path
    ):
        def refuses(argv):
            raise SystemExit(2)

        code, summary = _summary(monkeypatch, tmp_path, refuses, seeds="1000,1001")
        assert code == 1
        assert "error" in summary["seeds"]["1000"]
        assert "error" in summary["seeds"]["1001"]

    def test_an_exception_in_eval_run_is_an_error_entry_and_exit_1(
        self, monkeypatch, tmp_path
    ):
        def crashes(argv):
            raise RuntimeError("out of memory")

        code, summary = _summary(monkeypatch, tmp_path, crashes)
        assert code == 1
        assert "out of memory" in summary["seeds"]["1000"]["error"]

    def test_summary_carries_the_overrides_the_sha_and_the_per_seed_wall_time(
        self, monkeypatch, tmp_path
    ):
        rehearse = _rehearse(monkeypatch, tmp_path)
        monkeypatch.setattr(
            rehearse.subprocess, "check_output", lambda *a, **k: "abc123\n"
        )
        import eval.run

        monkeypatch.setattr(eval.run, "main", _FakeEval())
        argv = ["--stage", "variants", "--variant", "cap10", "--seeds", "1000"]
        assert rehearse.main(argv) == 0
        path = rehearse.seed_dir("variants", "cap10", 1000).parent / "summary.json"
        summary = json.loads(path.read_text())
        assert summary["overrides"] == rehearse.VARIANTS["cap10"]
        assert summary["git_sha"] == "abc123"
        assert summary["stage"] == "variants"
        assert summary["variant"] == "cap10"
        assert summary["seeds"]["1000"]["wall_seconds"] > 0

    def test_a_rehearsal_never_writes_outside_runs_rehearsal(
        self, monkeypatch, tmp_path
    ):
        for folder in ("results", "docs/img", "checkpoints", "config"):
            (tmp_path / folder).mkdir(parents=True)
        before = set(tmp_path.rglob("*"))
        _summary(monkeypatch, tmp_path, _FakeEval())
        new = set(tmp_path.rglob("*")) - before
        assert new
        root = tmp_path / "runs" / "rehearsal"
        assert all(p == tmp_path / "runs" or root in (p, *p.parents) for p in new)

    def test_the_run_is_handed_its_own_config_and_the_stage_flags(
        self, monkeypatch, tmp_path
    ):
        fake = _FakeEval()
        _summary(
            monkeypatch, tmp_path, fake, extra=("--tasks", "risk", "--privacy-sweep")
        )
        argv = fake.calls[0]
        folder = tmp_path / "runs" / "rehearsal" / "variants" / "base" / "seed_1000"
        assert argv[:6] == [
            "--config",
            str(folder / "config.yaml"),
            "--seed",
            "tuning",
            "--data-seed",
            "1000",
        ]
        assert argv[-3:] == ["--tasks", "risk", "--privacy-sweep"]
