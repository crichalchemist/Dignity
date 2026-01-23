# Repository Guidelines

## Project Structure & Module Organization
- `data/` contains ingestion, downloaders, and dataset preparation scripts; prepared datasets typically land under `data/data/<pair>/`.
- `models/` holds neural network architectures and model components.
- `train/` and `rl/` contain supervised training and A3C reinforcement-learning pipelines.
- `eval/` and `risk/` cover evaluation utilities and risk-related logic.
- `utils/` includes shared helpers and the end-to-end pipeline runner.
- `tests/` houses pytest suites; `docs/` and `benchmarks/` contain supporting material.
- Data artifacts live under `output_central/` (raw market data) and `checkpoints/` (saved models).

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt` installs Python dependencies (Python 3.10+).
- `python data/prepare_dataset.py --pairs gbpusd --t-in 120 --t-out 10 --task-type classification` builds datasets; add `--intrinsic-time` for directional-change bars.
- `python train/run_training.py --pairs gbpusd --epochs 50 --batch-size 64` runs supervised training.
- `python rl/run_a3c_training.py --pair gbpusd --env-mode backtesting --historical-data data/data/gbpusd/gbpusd_prepared.csv` runs RL backtesting.
- `python utils/run_training_pipeline.py --pairs gbpusd --run-histdata-download --run-rl-training` executes the full pipeline.
- `pytest` runs the full test suite; see Testing Guidelines for markers.
- Optional: `ruff check .` and `ruff format .` follow the repo’s lint/format rules.

## Coding Style & Naming Conventions
- Python formatting uses Ruff with 100-character lines, double quotes, and sorted imports (see `pyproject.toml`).
- Indentation is 4 spaces; use snake_case for functions/variables and PascalCase for classes.
- Keep module names descriptive and aligned with existing folders (e.g., `train/`, `eval/`, `rl/`).

## Testing Guidelines
- Tests use pytest with conventions defined in `pytest.ini` (`test_*.py`, `Test*`, `test_*`).
- Use markers for scope: `unit`, `integration`, `slow`, `fast`, `real_api`, `regression`, `performance`.
- Example: `pytest -m unit` or `pytest -m "not slow"`.

## Commit & Pull Request Guidelines
- Commit history is mixed: some Conventional Commit prefixes (`feat:`, `fix:`), many are plain descriptive sentences. Prefer concise, imperative summaries (and Conventional Commits when possible).
- PRs should include: a short summary, key files changed, how you tested (commands/output), and any data/API changes. Add screenshots only when UI changes exist.

## Security & Configuration Tips
- Store API keys as environment variables (e.g., `FRED_API_KEY`) and never commit secrets.
- Large datasets and model checkpoints should stay in `output_central/` and `checkpoints/`, not in git.
