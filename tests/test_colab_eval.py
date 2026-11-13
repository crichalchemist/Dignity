"""The Colab script and its runbook: what must run, in what order, and be pinned."""

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "colab_eval.py"
RUNBOOK = (
    "SHA=$(git rev-parse HEAD)        # must already be pushed to origin\n"
    "SEED=tuning                      # M7 final run only: SEED=eval\n"
    "mkdir -p runs/colab\n"
    "colab new -s dignity-eval --gpu L4\n"
    "colab exec -s dignity-eval -f scripts/colab_eval.py --timeout 43200 \\\n"
    '    --env DIGNITY_SHA="$SHA" --env DIGNITY_SEED="$SEED"\n'
    "colab download -s dignity-eval /content/dignity_results_$SEED.tar.gz "
    "runs/colab/dignity_results_$SEED.tar.gz\n"
    "colab stop -s dignity-eval\n"
    "tar -xzf runs/colab/dignity_results_$SEED.tar.gz -C runs/colab\n"
    "cp -r runs/colab/results/v1.0/. results/v1.0/\n"
    "cp -r runs/colab/docs/img/. docs/img/\n"
    "python scripts/refresh_doc_numbers.py README.md docs/RESULTS.md\n"
    "python -m pytest tests/test_docs.py\n"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("colab_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestColabEvalScript:
    def _run(
        self,
        monkeypatch,
        *,
        clone_exists,
        missing_yaml=False,
        eval_exit=0,
        events=None,
        data_seed=None,
        rehearsal=None,
        rehearse_exit=0,
    ):
        events = [] if events is None else events
        monkeypatch.setenv("DIGNITY_SHA", "abc123")
        if rehearsal is None:
            monkeypatch.setenv("DIGNITY_SEED", "tuning")
        else:
            monkeypatch.delenv("DIGNITY_SEED", raising=False)
            monkeypatch.setenv("DIGNITY_REHEARSAL", rehearsal)
        if data_seed is not None:
            monkeypatch.setenv("DIGNITY_DATA_SEED", data_seed)

        def fake_run(cmd, **kwargs):
            events.append(("run", list(cmd), kwargs))
            return subprocess.CompletedProcess(cmd, 0)

        def fake_main(argv):
            events.append(("main", argv))
            if eval_exit == "raise":
                raise RuntimeError("sweep failed")
            return eval_exit

        fake_module = types.ModuleType("eval.run")
        fake_module.main = fake_main
        monkeypatch.setitem(sys.modules, "eval.run", fake_module)

        def fake_rehearse(argv):
            events.append(("rehearse", argv))
            return rehearse_exit

        fake_rehearse_module = types.ModuleType("scripts.rehearse")
        fake_rehearse_module.main = fake_rehearse
        monkeypatch.setitem(sys.modules, "scripts.rehearse", fake_rehearse_module)
        script = _load_script()
        monkeypatch.setattr(script.subprocess, "run", fake_run)
        monkeypatch.setattr(script.os.path, "isdir", lambda p: clone_exists)
        monkeypatch.setattr(
            script.importlib.util,
            "find_spec",
            lambda name: None if missing_yaml else object(),
        )
        monkeypatch.setattr(script.os, "chdir", lambda p: events.append(("chdir", p)))
        monkeypatch.setattr(sys, "path", list(sys.path))
        script.main()
        return events

    def test_result_records_the_commit_that_ran_not_the_clone_head(self, monkeypatch):
        events = self._run(monkeypatch, clone_exists=True)
        cmds = [e[1] for e in events if e[0] == "run"]
        fetch = ["git", "-C", "/content/Dignity", "fetch", "origin"]
        checkout = ["git", "-C", "/content/Dignity", "checkout", "abc123"]
        assert cmds.index(fetch) < cmds.index(checkout)
        assert not any(c[:2] == ["git", "clone"] for c in cmds)

    def test_fresh_vm_clones_before_fetching(self, monkeypatch):
        events = self._run(monkeypatch, clone_exists=False)
        cmds = [e[1] for e in events if e[0] == "run"]
        assert cmds[0][:2] == ["git", "clone"]
        assert cmds[0][2] == "https://github.com/crichalchemist/Dignity"

    def test_every_subprocess_fails_loud(self, monkeypatch):
        events = self._run(monkeypatch, clone_exists=False, missing_yaml=True)
        runs = [e for e in events if e[0] == "run"]
        assert runs and all(e[2].get("check") is True for e in runs)

    def test_colab_numpy_stack_is_never_downgraded_by_a_requirements_install(
        self, monkeypatch
    ):
        events = self._run(monkeypatch, clone_exists=False, missing_yaml=True)
        flat = [" ".join(e[1]) for e in events if e[0] == "run"]
        assert not any("requirements" in c for c in flat)
        assert any("pip install pyyaml" in c for c in flat)

    def test_present_modules_are_not_reinstalled(self, monkeypatch):
        events = self._run(monkeypatch, clone_exists=True, missing_yaml=False)
        flat = [" ".join(e[1]) for e in events if e[0] == "run"]
        assert not any("pip" in c for c in flat)

    def test_eval_runs_on_the_requested_seed_after_entering_the_clone(
        self, monkeypatch
    ):
        events = self._run(monkeypatch, clone_exists=True)
        kinds = [e[0] for e in events]
        assert kinds.index("chdir") < kinds.index("main")
        main = next(e for e in events if e[0] == "main")
        assert main[1] == [
            "--config",
            "config/synthetic_hard.yaml",
            "--seed",
            "tuning",
            "--tasks",
            "risk,forecast,policy",
            "--privacy-sweep",
        ]
        assert sys.path[0] == "/content/Dignity"

    def test_results_are_tarred_for_download_after_the_eval(self, monkeypatch):
        events = self._run(monkeypatch, clone_exists=True)
        last = events[-1]
        assert last[0] == "run"
        assert last[1] == [
            "tar",
            "-czf",
            "/content/dignity_results_tuning.tar.gz",
            "results/v1.0",
            "docs/img",
        ]

    def test_results_on_disk_are_tarred_even_when_the_eval_fails(self, monkeypatch):
        # the sweep runs last; its failure must not lose the three task results
        for eval_exit, error in (("raise", RuntimeError), (2, AssertionError)):
            events = []
            with pytest.raises(error):
                self._run(
                    monkeypatch, clone_exists=True, eval_exit=eval_exit, events=events
                )
            assert events[-1][0] == "run"
            assert events[-1][1][:2] == ["tar", "-czf"]

    def test_data_seed_is_forwarded_and_names_the_tarball(self, monkeypatch):
        events = self._run(monkeypatch, clone_exists=True, data_seed="1004")
        main = next(e for e in events if e[0] == "main")
        assert main[1] == [
            "--config",
            "config/synthetic_hard.yaml",
            "--seed",
            "tuning",
            "--data-seed",
            "1004",
            "--tasks",
            "risk,forecast,policy",
            "--privacy-sweep",
        ]
        assert events[-1][1] == [
            "tar",
            "-czf",
            "/content/dignity_results_tuning_1004.tar.gz",
            "results/v1.0",
            "docs/img",
        ]

    def test_a_rehearsal_runs_rehearse_instead_of_the_final_command_and_tars_its_runs(
        self, monkeypatch
    ):
        events = self._run(
            monkeypatch,
            clone_exists=True,
            rehearsal="--stage variants --variant cold --seeds 1000,1001",
        )
        assert (
            "rehearse",
            ["--stage", "variants", "--variant", "cold", "--seeds", "1000,1001"],
        ) in events
        assert not any(e[0] == "main" for e in events)
        assert events[-1][1] == [
            "tar",
            "-czf",
            "/content/dignity_rehearsal.tar.gz",
            "runs/rehearsal",
        ]

    def test_a_rehearsal_that_exits_nonzero_still_tars_and_fails_loud(
        self, monkeypatch
    ):
        events = []
        with pytest.raises(AssertionError):
            self._run(
                monkeypatch,
                clone_exists=True,
                rehearsal="--stage variants",
                rehearse_exit=2,
                events=events,
            )
        assert events[-1][1][:2] == ["tar", "-czf"]

    def test_a_recorded_training_raise_is_reported_and_does_not_hide_the_tarball(
        self, monkeypatch, capsys
    ):
        events = self._run(monkeypatch, clone_exists=True, eval_exit=1)
        assert "EVAL_EXIT=1" in capsys.readouterr().out
        assert events[-1][1][:2] == ["tar", "-czf"]

    def test_stack_versions_are_printed_before_the_eval(self, monkeypatch, capsys):
        events = self._run(monkeypatch, clone_exists=True)
        out = capsys.readouterr().out
        for name in ("numpy", "torch", "pandas"):
            assert name in out
        assert events  # eval ran


class TestResultsRunbook:
    def _text(self):
        return (ROOT / "docs" / "RESULTS.md").read_text()

    def test_runbook_block_is_exactly_the_tested_command_sequence(self):
        assert "## Colab runbook" in self._text()
        assert f"```bash\n{RUNBOOK}```" in self._text()

    def test_reader_can_tell_which_commit_and_config_made_a_result(self):
        text = self._text()
        assert "Tested with google-colab-cli 0.7.2." in text
        assert "default of 30 s kills the run" in text
        assert "`git_sha`" in text and "`config_hash`" in text

    def test_runbook_moves_the_archive_into_the_tree_then_refreshes_and_tests(self):
        from tests.test_docs import runbook_problems

        assert runbook_problems(RUNBOOK) == []
        lines = RUNBOOK.splitlines()
        for i, line in enumerate(lines):
            if line.startswith(("cp ", "python ")):
                without = "\n".join(lines[:i] + lines[i + 1 :])
                assert runbook_problems(without), line
        swapped = lines[:]
        swapped[-1], swapped[-2] = swapped[-2], swapped[-1]
        assert runbook_problems("\n".join(swapped))
        assert "the refresh and the doc tests follow" in self._text()

    def test_runbook_copies_exactly_the_directories_the_script_archives(self):
        for path in ("results/v1.0", "docs/img"):
            assert f"cp -r runs/colab/{path}/. {path}/" in RUNBOOK
