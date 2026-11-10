from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


class TestCiWorkflow:
    def test_required_check_names_survive_workflow_edits(self):
        # Branch protection on dev and main requires these exact names; renaming a
        # job or a matrix entry leaves every PR waiting on a check that never runs.
        jobs = _workflow()["jobs"]
        assert "lint" in jobs
        assert jobs["test"]["strategy"]["matrix"]["python-version"] == ["3.10", "3.12"]

    def test_pushes_to_dev_and_main_run_every_test_under_the_privacy_gate(self):
        wf = _workflow()
        on = wf.get("on", wf.get(True))  # PyYAML reads the bare key `on` as True
        assert {"dev", "main"} <= set(on["push"]["branches"])
        runs = [s.get("run", "") for s in wf["jobs"]["test"]["steps"]]
        pytest_runs = [r for r in runs if r.lstrip().startswith("pytest")]
        assert len(pytest_runs) == 1
        assert "--cov-fail-under=100" in pytest_runs[0]
        # Deselecting slow tests is allowed on pull requests only.
        if "not slow" in pytest_runs[0]:
            assert "github.event_name == 'pull_request'" in pytest_runs[0]

    def test_ci_installs_the_chart_dependency_without_making_it_a_runtime_one(self):
        runs = " ".join(s.get("run", "") for s in _workflow()["jobs"]["test"]["steps"])
        assert "requirements-dev.txt" in runs
        assert "matplotlib" in (ROOT / "requirements-dev.txt").read_text()
        assert "matplotlib" not in (ROOT / "requirements.txt").read_text()

    def test_ci_runs_installed_console_scripts_and_all_tasks_through_eval_run(self):
        # DGNTY-4 shipped because only main(argv) was tested, never the installed
        # entry points. dignity-train and dignity-export run risk; forecast and
        # policy are exercised through eval.run.
        steps = _workflow()["jobs"]["smoke"]["steps"]
        runs = "\n".join(s.get("run", "") for s in steps)
        assert "dignity-train --config config/smoke.yaml" in runs
        assert "dignity-export --checkpoint runs/smoke/train/best.pt" in runs
        eval_run = (
            "python -m eval.run --smoke --seed tuning --tasks risk,forecast,policy"
            " --privacy-sweep"
        )
        assert eval_run in runs
        assert 'test -z "$(git status --porcelain)"' in runs  # outputs stay in runs/
