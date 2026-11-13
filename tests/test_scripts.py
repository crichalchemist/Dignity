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

    def test_the_variants_are_exactly_the_planned_single_factor_changes(self):
        assert _load_rehearse().VARIANTS == {
            "base": {},
            "cold": {"train.forecast_warm_start": False, "policy.multiplier_lr": 0.05},
            "cap10": {
                "train.forecast_pos_weight_cap": 10.0,
                "policy.multiplier_lr": 0.2,
            },
            "cap3": {"train.forecast_pos_weight_cap": 3.0},
            "narrow": {"model.hidden_size": 128},
            "lr1e-3": {"train.lr": 0.001},
        }

    def test_the_stages_are_timing_variants_and_candidate(self):
        assert _load_rehearse().STAGES == ("timing", "variants", "candidate")

    def test_the_rehearsal_runs_on_the_committed_hard_preset(self):
        rehearse = _load_rehearse()
        hard = REHEARSE.parent.parent / "config" / "synthetic_hard.yaml"
        assert hard == rehearse.PRESET

    def test_a_variant_override_naming_an_unknown_key_fails_before_any_run(
        self, monkeypatch, tmp_path
    ):
        rehearse = _rehearse(monkeypatch, tmp_path)
        monkeypatch.setitem(rehearse.VARIANTS, "typo", {"train.lerning_rate": 1.0})
        with pytest.raises(TypeError):
            rehearse.write_seed_config("variants", "typo", 1000)


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


class _RaisesOnFirstCall(_FakeEval):
    def __call__(self, argv):
        if not self.calls:
            self.calls.append(list(argv))
            raise RuntimeError("out of memory")
        return super().__call__(argv)


def _passing_policy_with_rates(values):
    return _seed_fixtures(policy__metrics__seed_spread__review_rate={"values": values})


class TestRehearseJudgement:
    def test_a_risk_that_fails_criterion_2_gives_c2_false_and_leaves_the_rest(
        self, monkeypatch, tmp_path
    ):
        files = _seed_fixtures(risk__metrics__pr_auc__ci_low=0.2)
        _, summary = _summary(monkeypatch, tmp_path, _FakeEval(files))
        entry = summary["seeds"]["1000"]
        assert (entry["C2"], entry["C3"], entry["C4"]) == (False, True, True)

    def test_the_criteria_get_the_values_verify_release_hands_them(
        self, monkeypatch, tmp_path
    ):
        from core.config import EvalConfig, PolicyConfig
        from tests.test_release import SEED, _write

        rehearse = _rehearse(monkeypatch, tmp_path)
        release = rehearse._release()
        seen = {"rehearse": [], "release": []}
        phase = ["release"]
        for name in ("_criterion_3", "_criterion_4"):
            original = getattr(release, name)

            def spy(*args, _name=name, _original=original):
                seen[phase[0]].append((_name, args[1:]))
                return _original(*args)

            monkeypatch.setattr(release, name, spy)
        _write(tmp_path)
        names = ("risk", "forecast", "policy", "privacy")
        files = {n: json.loads((tmp_path / f"{n}.json").read_text()) for n in names}
        release.verify(tmp_path, SEED, EvalConfig().epsilons)
        phase[0] = "rehearse"
        rehearse._judge(files, EvalConfig().n_train_seeds)
        assert seen["rehearse"] == seen["release"]
        assert seen["release"][1][1] == (
            PolicyConfig().review_budget,
            EvalConfig().n_train_seeds,
            PolicyConfig().collapse_review_factor,
        )

    @pytest.mark.parametrize("rates", [[], [None, 0.05, 0.05]])
    def test_c4_is_false_when_the_review_rates_are_missing_or_null(
        self, monkeypatch, tmp_path, rates
    ):
        rehearse = _rehearse(monkeypatch, tmp_path)
        release = rehearse._release()

        class Passes:
            passed = True

        monkeypatch.setattr(release, "_criterion_4", lambda *a: Passes)
        files = _passing_policy_with_rates(rates)
        files["risk"] = files["forecast"] = None
        assert rehearse._judge(files, 3)["C4"] is False

    def test_an_empty_review_rate_list_is_an_error_entry_not_a_pass(
        self, monkeypatch, tmp_path
    ):
        fake = _FakeEval(_passing_policy_with_rates([]))
        code, summary = _summary(monkeypatch, tmp_path, fake)
        assert code == 1
        assert "unusable results" in summary["seeds"]["1000"]["error"]

    def test_a_malformed_result_file_is_an_error_entry_and_the_next_seed_still_runs(
        self, monkeypatch, tmp_path
    ):
        class Corrupt(_FakeEval):
            def __call__(self, argv):
                status = super().__call__(argv)
                if len(self.calls) == 1:
                    config = yaml.safe_load(Path(argv[1]).read_text())
                    (Path(config["eval"]["out_dir"]) / "risk.json").write_text("{")
                return status

        fake = Corrupt()
        code, summary = _summary(monkeypatch, tmp_path, fake, seeds="1000,1001")
        assert code == 1
        assert "unusable results" in summary["seeds"]["1000"]["error"]
        assert summary["seeds"]["1001"]["C2"] is True

    def test_an_unreadable_checkpoint_is_an_error_entry_not_a_traceback(
        self, monkeypatch, tmp_path
    ):
        rehearse = _rehearse(monkeypatch, tmp_path)

        def broken(*args):
            raise ValueError("old checkpoint format")

        monkeypatch.setattr(rehearse, "_risk_val_pr_auc", broken)
        import eval.run

        monkeypatch.setattr(eval.run, "main", _FakeEval())
        argv = ["--stage", "variants", "--variant", "base", "--seeds", "1000"]
        assert rehearse.main(argv) == 1
        path = rehearse.seed_dir("variants", "base", 1000).parent / "summary.json"
        assert (
            "old checkpoint format"
            in json.loads(path.read_text())["seeds"]["1000"]["error"]
        )

    def test_a_raised_policy_leaves_c3_alone_and_a_raised_forecast_leaves_c4_alone(
        self, monkeypatch, tmp_path
    ):
        raised = {"error": "PolicyCollapseError", "message": "m", "train_seed": 1}
        files = _seed_fixtures()
        files["policy"] = {"metrics": {"raised": raised}}
        _, summary = _summary(monkeypatch, tmp_path, _FakeEval(files, status=1))
        assert summary["seeds"]["1000"]["C3"] is True
        files = _seed_fixtures()
        files["forecast"] = {"metrics": {"raised": raised}}
        _, summary = _summary(monkeypatch, tmp_path, _FakeEval(files, status=1))
        assert summary["seeds"]["1000"]["C4"] is True

    def test_a_policy_only_run_leaves_c2_and_c3_without_a_verdict(
        self, monkeypatch, tmp_path
    ):
        _, summary = _summary(
            monkeypatch, tmp_path, _FakeEval(), extra=("--tasks", "policy")
        )
        entry = summary["seeds"]["1000"]
        assert (entry["C2"], entry["C3"], entry["C4"]) == (None, None, True)


class TestRehearseSeeds:
    def test_every_seed_runs_on_its_own_data_seed_as_a_tuning_run(
        self, monkeypatch, tmp_path
    ):
        fake = _FakeEval()
        _summary(monkeypatch, tmp_path, fake, seeds="1000,1001")
        for call, seed in zip(fake.calls, ("1000", "1001"), strict=True):
            assert call[call.index("--data-seed") + 1] == seed
            assert call[call.index("--seed") + 1] == "tuning"
            assert "--privacy-sweep" not in call

    def test_the_privacy_sweep_runs_only_when_asked_for(self, monkeypatch, tmp_path):
        fake = _FakeEval()
        _summary(monkeypatch, tmp_path, fake, extra=("--privacy-sweep",))
        assert fake.calls[0][-1] == "--privacy-sweep"

    def test_an_eval_run_that_returns_2_is_an_error_entry_and_exit_1(
        self, monkeypatch, tmp_path
    ):
        code, summary = _summary(monkeypatch, tmp_path, _FakeEval(status=2))
        assert code == 1
        assert "exited 2" in summary["seeds"]["1000"]["error"]

    def test_an_errored_early_seed_still_exits_1_after_a_good_later_seed(
        self, monkeypatch, tmp_path
    ):
        fake = _RaisesOnFirstCall()
        code, summary = _summary(monkeypatch, tmp_path, fake, seeds="1000,1001")
        assert code == 1
        assert len(fake.calls) == 2
        assert "out of memory" in summary["seeds"]["1000"]["error"]
        assert summary["seeds"]["1001"]["C2"] is True

    def test_a_stale_result_file_from_an_earlier_run_is_never_judged(
        self, monkeypatch, tmp_path
    ):
        rehearse = _rehearse(monkeypatch, tmp_path)
        stale = rehearse.seed_dir("variants", "base", 1000) / "results"
        stale.mkdir(parents=True)
        (stale / "forecast.json").write_text(json.dumps(_seed_fixtures()["forecast"]))
        files = {k: v for k, v in _seed_fixtures().items() if k != "forecast"}
        _, summary = _summary(monkeypatch, tmp_path, _FakeEval(files))
        entry = summary["seeds"]["1000"]
        assert entry["C3"] is None
        assert entry["forecast_val_lift_upper"] is None

    def test_seeds_from_an_earlier_invocation_at_this_commit_stay_in_the_summary(
        self, monkeypatch, tmp_path
    ):
        _summary(monkeypatch, tmp_path, _FakeEval(), seeds="1000")
        _, summary = _summary(monkeypatch, tmp_path, _FakeEval(), seeds="1001")
        assert sorted(summary["seeds"]) == ["1000", "1001"]

    def test_seeds_from_another_commit_do_not_mix_into_the_summary(
        self, monkeypatch, tmp_path
    ):
        rehearse = _rehearse(monkeypatch, tmp_path)
        import eval.run

        monkeypatch.setattr(eval.run, "main", _FakeEval())
        monkeypatch.setattr(rehearse, "_git_sha", lambda: "aaa")
        base = ["--stage", "variants", "--variant", "base"]
        rehearse.main([*base, "--seeds", "1000"])
        monkeypatch.setattr(rehearse, "_git_sha", lambda: "bbb")
        rehearse.main([*base, "--seeds", "1001"])
        path = rehearse.seed_dir("variants", "base", 1000).parent / "summary.json"
        assert list(json.loads(path.read_text())["seeds"]) == ["1001"]

    def test_the_candidate_stage_runs_and_is_recorded_in_the_summary(
        self, monkeypatch, tmp_path
    ):
        rehearse = _rehearse(monkeypatch, tmp_path)
        import eval.run

        monkeypatch.setattr(eval.run, "main", _FakeEval())
        argv = ["--stage", "candidate", "--variant", "base", "--seeds", "1000"]
        assert rehearse.main(argv) == 0
        path = rehearse.seed_dir("candidate", "base", 1000).parent / "summary.json"
        assert json.loads(path.read_text())["stage"] == "candidate"


class TestRehearseEndToEnd:
    def test_the_real_rehearsal_runs_the_real_eval_through_the_colab_entry_point(
        self, monkeypatch, tmp_path
    ):
        import importlib
        import os
        import subprocess

        from tests.test_colab_eval import _load_script

        repo = REHEARSE.parent.parent
        preset = yaml.safe_load((repo / "config" / "smoke.yaml").read_text())
        preset["eval"]["n_train_seeds"] = 3
        preset_path = tmp_path / "preset.yaml"
        preset_path.write_text(yaml.safe_dump(preset))
        root = tmp_path / "root"
        root.mkdir()
        rehearse = importlib.import_module("scripts.rehearse")
        monkeypatch.setattr(rehearse, "ROOT", root)
        monkeypatch.setattr(rehearse, "PRESET", preset_path)

        monkeypatch.setenv("DIGNITY_SHA", "abc123")
        monkeypatch.delenv("DIGNITY_SEED", raising=False)
        monkeypatch.setenv(
            "DIGNITY_REHEARSAL", "--stage timing --variant base --seeds 1000"
        )
        commands = []
        script = _load_script()
        real_run = subprocess.run

        def run(cmd, *args, **kwargs):
            # the VM-only steps are recorded; git queries run for real
            if (
                cmd[0] in ("tar",)
                or cmd[:2] == ["git", "clone"]
                or cmd[3:4]
                in (
                    ["fetch"],
                    ["checkout"],
                )
            ):
                commands.append(list(cmd))
                return subprocess.CompletedProcess(cmd, 0)
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr(script.subprocess, "run", run)
        real_isdir = os.path.isdir
        monkeypatch.setattr(
            script.os.path,
            "isdir",
            lambda p: True if p == script.REPO_DIR else real_isdir(p),
        )
        monkeypatch.setattr(script.os, "chdir", lambda p: None)
        status_before = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout

        script.main()

        assert commands[-1] == [
            "tar",
            "-czf",
            "/content/dignity_rehearsal.tar.gz",
            "runs/rehearsal",
        ]
        base = root / "runs" / "rehearsal" / "timing" / "base"
        summary = json.loads((base / "summary.json").read_text())
        assert set(summary) == {"stage", "variant", "overrides", "git_sha", "seeds"}
        entry = summary["seeds"]["1000"]
        assert "error" not in entry
        assert set(entry) == {
            "C2",
            "C3",
            "C4",
            "raised",
            "forecast_val_lift_upper",
            "forecast_floor",
            "forecast_best_epoch",
            "policy_val_neg_cost",
            "review_rate",
            "review_multiplier",
            "uncapped_review_share",
            "risk_val_pr_auc",
            "wall_seconds",
        }
        assert entry["raised"] is None
        for key in (
            "forecast_val_lift_upper",
            "forecast_floor",
            "forecast_best_epoch",
            "policy_val_neg_cost",
            "review_rate",
            "review_multiplier",
            "uncapped_review_share",
            "risk_val_pr_auc",
        ):
            assert len(entry[key]) == 3, key
        written = [p for p in tmp_path.rglob("*") if p.is_file()]
        assert preset_path in written
        assert all(base in p.parents or p == preset_path for p in written), (
            "a file was written outside the rehearsal output root"
        )
        status_after = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert status_after == status_before
