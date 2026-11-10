"""The Colab script and its runbook: what must run, in what order, and be pinned."""

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "colab_eval.py"
RUNBOOK = (
    "SHA=$(git rev-parse HEAD)        # must already be pushed to origin\n"
    "SEED=tuning                      # M7 final run only: SEED=eval\n"
    "mkdir -p runs/colab\n"
    "colab new -s dignity-eval --gpu T4\n"
    "colab exec -s dignity-eval -f scripts/colab_eval.py --timeout 43200 \\\n"
    '    --env DIGNITY_SHA="$SHA" --env DIGNITY_SEED="$SEED"\n'
    "colab download -s dignity-eval /content/dignity_results_$SEED.tar.gz "
    "runs/colab/dignity_results_$SEED.tar.gz\n"
    "colab stop -s dignity-eval\n"
    "tar -xzf runs/colab/dignity_results_$SEED.tar.gz -C runs/colab\n"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("colab_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestColabEvalScript:
    def _run(self, monkeypatch, *, clone_exists, missing_yaml=False):
        events = []
        monkeypatch.setenv("DIGNITY_SHA", "abc123")
        monkeypatch.setenv("DIGNITY_SEED", "tuning")

        def fake_run(cmd, **kwargs):
            events.append(("run", list(cmd), kwargs))
            return subprocess.CompletedProcess(cmd, 0)

        def fake_main(argv):
            events.append(("main", argv))
            return 0

        fake_module = types.ModuleType("eval.run")
        fake_module.main = fake_main
        monkeypatch.setitem(sys.modules, "eval.run", fake_module)
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
