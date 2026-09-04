# Privacy Honest Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every public privacy claim in Dignity true or remove it, wire the privacy primitives into `TransactionPipeline` as a real stage, add an operator-side threat model with enforcing tests, and gate all of it in CI.

**Architecture:** `core/privacy.py` becomes an instance-based `PrivacyManager` bound to a `PrivacyBudget` ledger, with keyed HMAC pseudonymization, bounded Laplace noise, and merge-based k-anonymous generalization. `TransactionPipeline` gains an `_apply_privacy` stage that runs exactly once per public call, *before* `compute_signals`, driven by a new `PrivacyConfig` dataclass in `core/config.py`. Three new test modules enforce the subject layer, the operator layer, and the docs; a GitHub Actions workflow runs them on every push with 100% coverage required on `core/privacy.py`.

**Tech Stack:** Python 3.10+, numpy, pandas, scikit-learn, PyTorch (CPU), onnx, pytest, pytest-cov, ruff, pre-commit, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-03-privacy-honest-core-design.md` — read it first; every task below cites the section it implements.

## Global Constraints

- Python floor is 3.10 (`setup.py: python_requires=">=3.10"`); type hints use `X | None` syntax; CI runs 3.10 and 3.12.
- Formatter and linter are **ruff only**; line length is **88** (spec §2). No black, no isort.
- Torch is always installed from the CPU index: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- Package imports are absolute from repo root (`from core.privacy import ...`). Run everything from the repo root with the venv active.
- Branch is `privacy-honest-core`. Commit after every task with a Conventional-Commits-style prefix matching history (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `style:`, `ci:`).
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB
  ```
- Never write these exact strings into any public doc (`README.md`, `docs/*.md` outside `docs/plans/` and `docs/superpowers/`) — `tests/test_docs.py` fails on them: `Secure Aggregation`, `hash_identifiers(`, `anonymize_addresses`, `sanitize_dataset`, `suppress_rare_events`, `differentially private model`, `.hash_identifier(`, `.add_noise(`, `.quantize_amounts(`, `anonymize_amounts(`, `add_differential_privacy_noise(`. When a doc needs to say the model is *not* DP, write "the trained model carries no DP guarantee" — never the denylisted phrase, even negated.
- `core/privacy.py` must reach **100% line coverage** (`pytest --cov=core.privacy --cov-fail-under=100`). Every branch you write in that file needs a test that exercises it.
- Do not touch: `models/`, `train/engine.py`, `data/loader.py`, `data/source/`, `export/to_onnx.py` (except the one conditional compatibility fix in Task 10), README sections unrelated to privacy, `.internal/`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `core/privacy.py` | `BudgetExhausted`, `PrivacyBudget`, `PrivacyManager` — the primitives, nothing else | Rewrite |
| `core/__init__.py` | Re-export `PrivacyBudget`, `BudgetExhausted` alongside existing names | Modify |
| `core/config.py` | Add `PrivacyFeature`, `PrivacyConfig`; wire into `DignityConfig` load/save | Modify |
| `data/pipeline.py` | Add `privacy`/`privacy_rng` kwargs, `_apply_privacy`, `_fit_on`, `_transform_on`; public `fit`/`transform`/`fit_transform` each apply privacy once | Modify |
| `train/cli.py` | Pass `config.privacy` into the pipeline; print ε spent | Modify |
| `config/base.yaml` | Commented `privacy:` example | Modify |
| `config/train_risk.yaml` | Live `privacy:` block | Modify |
| `tests/test_privacy.py` | Subject-layer primitive tests (18) | Create |
| `tests/test_operator.py` | Operator-layer tests (3) | Create |
| `tests/test_docs.py` | Removed-claims denylist test (1) | Create |
| `tests/test_core.py` | Remove `TestPrivacyManager`; add `TestPrivacyConfig` | Modify |
| `tests/test_data.py` | Add `TestPrivacyStage` | Modify |
| `tests/__init__.py` | Remove dead `get_api_url` | Modify |
| `docs/THREAT-MODEL.md` | Two adversaries, deniability defined, limitations | Create |
| `docs/PRIVACY.md` | Rewrite around the claims contract | Rewrite |
| `docs/ARCHITECTURE.md`, `docs/QUICK_START.md` | Replace phantom privacy API snippets | Modify |
| `docs/training-curriculum.md` | Forex plan from reverted direction | Delete |
| `README.md` | Privacy sections, badge, contributing | Modify |
| `.github/workflows/ci.yml` | `lint` + `test` jobs | Create |
| `.pre-commit-config.yaml` | ruff + ruff-format hooks | Create |
| `pyproject.toml`, `.editorconfig` | Width → 88 | Modify |
| `.gitignore` | `.venv/`, `.coverage`, `htmlcov/` | Modify |
| `setup.py` | `pre-commit` in dev extras | Modify |
| `.claude/CLAUDE.md` | Reflect the new state | Modify |
| `docs/superpowers/specs/2026-09-03-privacy-honest-core-design.md` | Amend test count | Modify |

---

### Task 0: Environment and baseline

No environment exists on this machine. `.venv/` is a stale macOS virtualenv with dead symlinks. Nothing can be verified until this task is done.

**Files:**
- Modify: `.gitignore` (append after line 30, the `# Virtual environments` block)

**Interfaces:**
- Produces: a working `.venv` with torch (CPU), all requirements, the package in editable mode, and pre-commit; a recorded green baseline of 31 tests.

- [ ] **Step 1: Confirm you are on the right branch with a clean tree**

Run: `git branch --show-current && git status --short`
Expected: `privacy-honest-core`, and status shows only untracked `.claude/`, `.gemini/`, `.agents/`, `CLAUDE.md`. Anything else — stop and ask.

- [ ] **Step 2: Replace the dead venv**

```bash
rm -rf .venv
/usr/bin/python3.12 -m venv .venv
source .venv/bin/activate
python --version
```
Expected: `Python 3.12.x`

- [ ] **Step 3: Install dependencies exactly as CI will**

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps
pip install pre-commit
python -c "import torch, onnx, onnxruntime, sklearn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```
Expected: last line prints a torch version and `cuda False`. If `pip install -r requirements.txt` fails on `numpy<2.0` against Python 3.12, stop — that is a real dependency conflict to surface, not to work around.

- [ ] **Step 4: Run the baseline suite**

Run: `pytest tests/ -q`
Expected: `31 passed`. If anything fails *before you have changed code*, record the failure verbatim in your report and stop — the baseline is broken and the plan's counts are wrong.

- [ ] **Step 5: Ignore the venv**

`.gitignore` lists `venv/`, `ENV/`, `env/` but not `.venv/`. Append to the `# Virtual environments` block:

```
.venv/
```

- [ ] **Step 6: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore .venv

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB"
```

---

### Task 1: Settle the formatter width at 88 and get ruff green

Spec §2, §9. The tree is black-formatted at 88 but `pyproject.toml` says 100 and `.editorconfig` says 120. The CI `lint` job runs `ruff check .` and `ruff format --check .`; neither can go green until the config agrees with the files and any pre-existing lint findings are fixed.

**Files:**
- Modify: `pyproject.toml:11` (`line-length = 100`)
- Modify: `.editorconfig:9` (`max_line_length = 120`)

- [ ] **Step 1: Set the width**

In `pyproject.toml` change:
```toml
line-length = 100
```
to:
```toml
line-length = 88
```

In `.editorconfig` change:
```
max_line_length = 120
```
to:
```
max_line_length = 88
```

- [ ] **Step 2: Check the format gate**

Run: `ruff format --check .`
Expected: `N files already formatted` with no "would reformat" lines. Black and ruff-format differ in a few edge cases, so this may list files.

- [ ] **Step 3: If Step 2 listed files, reformat in its own commit**

Only if Step 2 reported files:
```bash
ruff format .
git add -u
git commit -m "style: reformat with ruff at line-length 88

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB"
```
Re-run `ruff format --check .` and confirm it is clean.

- [ ] **Step 4: Check the lint gate**

Run: `ruff check .`
Expected: `All checks passed!`. If there are findings, read each one. Fix with `ruff check --fix .` for the mechanical ones (import order, unused imports, `UP` modernizations). Anything `--fix` does not resolve, fix by hand — but only inside the reported line; do not refactor around it. If a finding would require a behavioral change, stop and report it instead of fixing.

- [ ] **Step 5: Confirm tests still pass**

Run: `pytest tests/ -q`
Expected: `31 passed`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .editorconfig
git add -u   # picks up any lint fixes
git commit -m "chore: settle formatter width at 88 and clear ruff findings

Matches the tree as black left it; pyproject.toml, .editorconfig now agree.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB"
```

---

### Task 2: Purge dead code and the dead doc

Spec §4. Everything here has zero callers or belongs to the reverted direction. None of it is covered by the 31 tests, so they stay green.

**Files:**
- Delete: `docs/training-curriculum.md`
- Modify: `tests/__init__.py` (whole file)
- Modify: `core/privacy.py:101-158` (`suppress_rare_events` and `sanitize_dataset`)

- [ ] **Step 1: Delete the forex curriculum**

```bash
git rm docs/training-curriculum.md
```

- [ ] **Step 2: Empty the test package marker**

Replace the entire contents of `tests/__init__.py` with an empty file:

```bash
: > tests/__init__.py
```

- [ ] **Step 3: Remove the two dead methods**

In `core/privacy.py`, delete everything from the line `    @staticmethod` that precedes `def suppress_rare_events(` through the end of the file (the `sanitize_dataset` classmethod is the last thing in the file). After the edit the file ends with the `return values + noise` line of `add_noise`. Confirm:

Run: `grep -n "suppress_rare_events\|sanitize_dataset" core/privacy.py`
Expected: no output.

- [ ] **Step 4: Verify**

Run: `pytest tests/ -q && ruff check . && ruff format --check .`
Expected: `31 passed`, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add -A docs/training-curriculum.md tests/__init__.py core/privacy.py
git commit -m "chore: remove untested privacy helpers and the forex curriculum doc

suppress_rare_events and sanitize_dataset had no callers and no tests; the
latter also returned un-noised data beside noised data. training-curriculum.md
described the reverted April trading direction.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB"
```

---

### Task 3: `PrivacyBudget` and `BudgetExhausted`

Spec §5. The ledger comes first because `add_laplace_noise` depends on it.

**Files:**
- Modify: `core/privacy.py` (add two classes above `PrivacyManager`)
- Modify: `core/__init__.py` (exports)
- Create: `tests/test_privacy.py`

**Interfaces:**
- Produces: `PrivacyBudget(epsilon_total: float)` with `.spend(epsilon: float) -> None`, `.spent -> float`, `.remaining -> float`, `.epsilon_total -> float`; `BudgetExhausted(ValueError)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_privacy.py`:

```python
"""Subject-layer privacy primitives. Every test here backs one sentence in docs/PRIVACY.md."""

import pytest

from core.privacy import BudgetExhausted, PrivacyBudget


class TestPrivacyBudget:
    """Sequential composition: epsilons add, and overspending refuses."""

    def test_refuses_to_overspend(self):
        budget = PrivacyBudget(epsilon_total=1.0)
        budget.spend(0.6)
        with pytest.raises(BudgetExhausted):
            budget.spend(0.5)
        # a refused spend must not be recorded
        assert budget.spent == pytest.approx(0.6)

    def test_allows_spending_exactly_to_total(self):
        budget = PrivacyBudget(epsilon_total=1.0)
        budget.spend(0.5)
        budget.spend(0.5)
        assert budget.remaining == pytest.approx(0.0)

    def test_composes_sequentially(self):
        budget = PrivacyBudget(epsilon_total=2.0)
        for eps in (0.25, 0.5, 0.75):
            budget.spend(eps)
        assert budget.spent == pytest.approx(1.5)
        assert budget.remaining == pytest.approx(0.5)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_nonpositive_total_and_spend(self, bad):
        with pytest.raises(ValueError):
            PrivacyBudget(epsilon_total=bad)
        with pytest.raises(ValueError):
            PrivacyBudget(epsilon_total=1.0).spend(bad)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_privacy.py -q`
Expected: `ImportError: cannot import name 'BudgetExhausted'`

- [ ] **Step 3: Implement**

In `core/privacy.py`, replace the module docstring and imports at the top of the file (currently lines 1-5) with:

```python
"""Privacy-preserving utilities for transaction data.

Two families of primitives, each backed by a test in tests/test_privacy.py:

- Keyed pseudonymization (HMAC-SHA256): same key -> linkable, different key ->
  unlinkable.
- Input-level differential privacy: bounded Laplace noise spent against an
  epsilon ledger, and k-anonymous generalization by quantile binning.

Nothing here claims more than its test proves. docs/PRIVACY.md and
docs/THREAT-MODEL.md carry the exact wording of each guarantee.
"""

import hashlib
import hmac
import math
import secrets

import numpy as np

_MIN_KEY_BYTES = 16


class BudgetExhausted(ValueError):
    """Raised when a spend would exceed the configured epsilon total."""


class PrivacyBudget:
    """Sequential-composition ledger for epsilon-differential privacy.

    Epsilons add across releases. A spend that would push the total past
    ``epsilon_total`` raises instead of degrading the guarantee silently.
    """

    def __init__(self, epsilon_total: float):
        if epsilon_total <= 0:
            raise ValueError("epsilon_total must be positive")
        self.epsilon_total = float(epsilon_total)
        self._spent = 0.0

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return self.epsilon_total - self._spent

    def spend(self, epsilon: float) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self._spent + epsilon > self.epsilon_total + 1e-12:
            raise BudgetExhausted(
                f"spending {epsilon} would exceed budget: "
                f"{self._spent:.6f} spent of {self.epsilon_total:.6f}"
            )
        self._spent += epsilon
```

Leave the existing `class PrivacyManager:` and its remaining methods below this for now; Tasks 4-6 replace them.

Replace `core/__init__.py` with:

```python
"""Core utilities: config, signals, privacy."""

from .config import DignityConfig
from .privacy import BudgetExhausted, PrivacyBudget, PrivacyManager
from .signals import SignalProcessor

__all__ = [
    "DignityConfig",
    "SignalProcessor",
    "PrivacyManager",
    "PrivacyBudget",
    "BudgetExhausted",
]
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_privacy.py -q`
Expected: `5 passed` (the parametrized test counts twice).

- [ ] **Step 5: Verify the rest still passes and ruff is clean**

Run: `pytest tests/ -q && ruff check . && ruff format --check .`
Expected: `36 passed`, ruff clean. (`hmac`, `math`, `secrets` are unused until Task 4; if ruff flags F401 on them, that is expected — temporarily remove those three imports and re-add them in Task 4.)

- [ ] **Step 6: Commit**

```bash
git add core/privacy.py core/__init__.py tests/test_privacy.py
git commit -m "feat(privacy): add PrivacyBudget epsilon ledger

Sequential composition with refusal on overspend. No mechanism spends
without going through it.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB"
```

---

### Task 4: `PrivacyManager` construction and keyed pseudonymization

Spec §5. Replaces `hash_identifier` and `anonymize_addresses`. The old tests in `tests/test_core.py::TestPrivacyManager` go in this task — two of them call the API with no key, which the new design forbids.

**Files:**
- Modify: `core/privacy.py` (`PrivacyManager.__init__`, `pseudonymize`, `pseudonymize_many`; delete `hash_identifier`, `anonymize_addresses`)
- Modify: `tests/test_core.py:8` (import) and `tests/test_core.py:73-122` (delete `TestPrivacyManager`)
- Modify: `tests/test_privacy.py` (add class)

**Interfaces:**
- Consumes: `PrivacyBudget` from Task 3.
- Produces: `PrivacyManager(budget: PrivacyBudget, key: bytes | str | None = None, rng=None)`; `.pseudonymize(identifier: str) -> str`; `.pseudonymize_many(identifiers: list[str]) -> list[str]`; `.budget`. `rng` is any object with `.random() -> float in [0, 1)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_privacy.py`:

```python
def _manager(key=b"0123456789abcdef", total=1.0, rng=None):
    return PrivacyManager(PrivacyBudget(epsilon_total=total), key=key, rng=rng)


class TestPseudonymize:
    """Keyed pseudonymization: linkable under one key, unlinkable across keys."""

    def test_same_key_is_deterministic(self):
        pm = _manager()
        assert pm.pseudonymize("0xabc") == pm.pseudonymize("0xabc")
        assert len(pm.pseudonymize("0xabc")) == 64

    def test_different_keys_are_unlinkable(self):
        a = _manager(key=b"key-one-is-16-bytes!")
        b = _manager(key=b"key-two-is-16-bytes!")
        assert a.pseudonymize("0xabc") != b.pseudonymize("0xabc")

    def test_rejects_short_key(self):
        with pytest.raises(ValueError, match="16 bytes"):
            _manager(key=b"short")

    def test_requires_key_at_call_time(self):
        pm = _manager(key=None)  # construction is allowed without a key
        with pytest.raises(ValueError, match="key"):
            pm.pseudonymize("0xabc")

    def test_distinct_key_id_pairs_never_collide_on_concatenation(self):
        # Under salt + identifier concatenation these two pairs hash identically.
        # HMAC keeps key and message in separate domains.
        a = PrivacyManager(PrivacyBudget(1.0), key=b"aaaaaaaaaaaaaaaab")
        b = PrivacyManager(PrivacyBudget(1.0), key=b"aaaaaaaaaaaaaaaa")
        assert a.pseudonymize("cd") != b.pseudonymize("bcd")

    def test_pseudonymize_many_preserves_order_and_length(self):
        pm = _manager()
        ids = ["x", "y", "x"]
        out = pm.pseudonymize_many(ids)
        assert len(out) == 3
        assert out[0] == out[2] != out[1]

    def test_str_key_is_utf8_encoded(self):
        assert _manager(key="0123456789abcdef").pseudonymize("q") == _manager(
            key=b"0123456789abcdef"
        ).pseudonymize("q")
```

Change the existing import at the top of the file to `from core.privacy import BudgetExhausted, PrivacyBudget, PrivacyManager`.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_privacy.py::TestPseudonymize -q`
Expected: failures with `TypeError: PrivacyManager() takes no arguments` or `AttributeError: ... 'pseudonymize'`.

- [ ] **Step 3: Implement**

In `core/privacy.py`, replace the existing `class PrivacyManager:` header, its docstring, and the `hash_identifier` and `anonymize_addresses` static methods with:

```python
class PrivacyManager:
    """Privacy primitives bound to one budget, one optional key, and one RNG.

    ``rng`` is any object exposing ``random() -> float`` in [0, 1). Production
    uses ``secrets.SystemRandom``; tests inject ``random.Random(seed)`` or a
    constant for determinism without weakening the default.
    """

    def __init__(
        self,
        budget: PrivacyBudget,
        key: bytes | str | None = None,
        rng=None,
    ):
        if isinstance(key, str):
            key = key.encode("utf-8")
        if key is not None and len(key) < _MIN_KEY_BYTES:
            raise ValueError(f"key must be at least {_MIN_KEY_BYTES} bytes")
        self.budget = budget
        self._key = key
        self._rng = rng if rng is not None else secrets.SystemRandom()

    # -- keyed pseudonymization ---------------------------------------------

    def pseudonymize(self, identifier: str) -> str:
        """HMAC-SHA256 of ``identifier`` under the manager's key, as hex.

        Same key -> same pseudonym (linkable). Different key -> unrelated
        pseudonym (unlinkable). Requires a key; never falls back to unkeyed.
        """
        if self._key is None:
            raise ValueError("pseudonymize requires a key")
        return hmac.new(
            self._key, identifier.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def pseudonymize_many(self, identifiers: list[str]) -> list[str]:
        return [self.pseudonymize(i) for i in identifiers]
```

Keep `quantize_amounts` and `add_noise` in place for now (Tasks 5-6 replace them). If you removed `hmac`/`math`/`secrets` imports in Task 3, re-add `hmac` and `secrets` now (`math` returns in Task 5).

- [ ] **Step 4: Remove the superseded tests**

In `tests/test_core.py`:
- Delete line 8: `from core.privacy import PrivacyManager`
- Delete the whole `class TestPrivacyManager:` block (lines 73 through the blank line before `class TestDignityConfig:`).

Confirm: `grep -n "PrivacyManager" tests/test_core.py` prints nothing.

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/ -q && ruff check . && ruff format --check .`
Expected: `39 passed` (31 − 4 + 5 + 7), ruff clean.

- [ ] **Step 6: Commit**

```bash
git add core/privacy.py tests/test_privacy.py tests/test_core.py
git commit -m "feat(privacy): keyed HMAC pseudonymization replaces salted hashing

PrivacyManager is now instance-based, bound to a PrivacyBudget. hash_identifier
and anonymize_addresses are gone: the former was unsalted by default and used
ambiguous salt+id concatenation; the latter's name overclaimed. Removes the
four tests that exercised the old API without a key.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB"
```

---

### Task 5: `add_laplace_noise` — bounded, ledgered, CSPRNG-backed

Spec §5. Replaces `add_noise`. Sensitivity is derived from required `bounds`; the budget is spent *before* sampling; noise comes from an injectable RNG that defaults to `secrets.SystemRandom`.

**Files:**
- Modify: `core/privacy.py` (add `add_laplace_noise`, `_laplace`; delete `add_noise`)
- Modify: `tests/test_privacy.py` (add class)

**Interfaces:**
- Consumes: `PrivacyManager`, `PrivacyBudget` from Tasks 3-4.
- Produces: `PrivacyManager.add_laplace_noise(values: np.ndarray, *, epsilon: float, bounds: tuple[float, float]) -> np.ndarray` — returns a new float array of the same shape; input is not modified.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_privacy.py` (add `import random`, `import secrets`, `import numpy as np` to the top-of-file imports):

```python
class _ZeroNoise:
    """rng whose uniform is always 0.5 -> Laplace inverse CDF at u=0 -> exactly 0 noise."""

    def random(self) -> float:
        return 0.5


class TestAddLaplaceNoise:
    """Input-level epsilon-DP: bounded, ledgered, non-recoverable noise."""

    def test_clips_to_bounds_before_noising(self):
        pm = _manager(rng=_ZeroNoise())
        values = np.array([-50.0, 0.0, 500.0, 1500.0])
        out = pm.add_laplace_noise(values, epsilon=1.0, bounds=(0.0, 1000.0))
        np.testing.assert_array_equal(out, [0.0, 0.0, 500.0, 1000.0])
        # input untouched
        np.testing.assert_array_equal(values, [-50.0, 0.0, 500.0, 1500.0])

    def test_noise_scale_is_sensitivity_over_epsilon(self):
        # Laplace(0, b) has mean |x| = b. With bounds width 100 and eps 0.5, b = 200.
        pm = _manager(total=10.0, rng=random.Random(42))
        out = pm.add_laplace_noise(np.zeros(20_000), epsilon=0.5, bounds=(0.0, 100.0))
        assert np.mean(np.abs(out)) == pytest.approx(200.0, rel=0.05)
        # symmetric around zero
        assert np.mean(out) == pytest.approx(0.0, abs=10.0)

    def test_spends_budget(self):
        pm = _manager(total=1.0, rng=_ZeroNoise())
        pm.add_laplace_noise(np.ones(3), epsilon=0.4, bounds=(0.0, 1.0))
        assert pm.budget.spent == pytest.approx(0.4)
        pm.add_laplace_noise(np.ones(3), epsilon=0.6, bounds=(0.0, 1.0))
        with pytest.raises(BudgetExhausted):
            pm.add_laplace_noise(np.ones(3), epsilon=0.1, bounds=(0.0, 1.0))

    def test_budget_is_spent_before_any_release(self):
        # If the spend fails, nothing should have been sampled or returned.
        pm = _manager(total=0.5, rng=_ZeroNoise())
        with pytest.raises(BudgetExhausted):
            pm.add_laplace_noise(np.ones(3), epsilon=1.0, bounds=(0.0, 1.0))
        assert pm.budget.spent == 0.0

    @pytest.mark.parametrize("eps", [0.0, -0.1])
    def test_rejects_nonpositive_epsilon(self, eps):
        with pytest.raises(ValueError):
            _manager().add_laplace_noise(np.ones(2), epsilon=eps, bounds=(0.0, 1.0))

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match="lo < hi"):
            _manager().add_laplace_noise(np.ones(2), epsilon=1.0, bounds=(1.0, 1.0))

    def test_default_rng_is_system_random(self):
        pm = PrivacyManager(PrivacyBudget(1.0))
        assert isinstance(pm._rng, secrets.SystemRandom)

    def test_preserves_shape(self):
        pm = _manager(rng=_ZeroNoise())
        out = pm.add_laplace_noise(np.zeros((3, 4)), epsilon=1.0, bounds=(0.0, 1.0))
        assert out.shape == (3, 4)

    def test_uniform_at_exactly_zero_is_resampled(self):
        # rng.random() == 0.0 maps to u = -0.5 -> log(0). Must resample, not return -inf.
        class _ZeroThenHalf:
            def __init__(self):
                self.calls = 0

            def random(self):
                self.calls += 1
                return 0.0 if self.calls == 1 else 0.5

        pm = _manager(rng=_ZeroThenHalf())
        out = pm.add_laplace_noise(np.zeros(1), epsilon=1.0, bounds=(0.0, 1.0))
        assert np.isfinite(out).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_privacy.py::TestAddLaplaceNoise -q`
Expected: `AttributeError: 'PrivacyManager' object has no attribute 'add_laplace_noise'`

- [ ] **Step 3: Implement**

In `core/privacy.py`, delete the `add_noise` static method and add, inside `PrivacyManager` after `pseudonymize_many`:

```python
# -- differential privacy -------------------------------------------------


def add_laplace_noise(
    self,
    values: np.ndarray,
    *,
    epsilon: float,
    bounds: tuple[float, float],
) -> np.ndarray:
    """Clip ``values`` to ``bounds`` and add Laplace noise for epsilon-DP.

    Sensitivity is the width of ``bounds``; the caller cannot understate it
    by omission. ``epsilon`` is spent from the budget before any sampling,
    so a refused spend releases nothing. Floating-point implementation:
    see docs/THREAT-MODEL.md for the Mironov (2012) caveat.

    Returns a new array; ``values`` is not modified.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    lo, hi = bounds
    if lo >= hi:
        raise ValueError("bounds must satisfy lo < hi")
    self.budget.spend(epsilon)
    scale = (hi - lo) / epsilon
    clipped = np.clip(np.asarray(values, dtype=float), lo, hi)
    noise = np.fromiter(
        (self._laplace(scale) for _ in range(clipped.size)),
        dtype=float,
        count=clipped.size,
    ).reshape(clipped.shape)
    return clipped + noise


def _laplace(self, scale: float) -> float:
    """One Laplace(0, scale) draw by inverse CDF from a uniform in [0, 1)."""
    while True:
        u = self._rng.random() - 0.5
        if abs(u) < 0.5:
            break
    return -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))
```

Ensure `import math` is present at the top of the file.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_privacy.py -q`
Expected: `21 passed`

- [ ] **Step 5: Full suite and ruff**

Run: `pytest tests/ -q && ruff check . && ruff format --check .`
Expected: `48 passed`, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add core/privacy.py tests/test_privacy.py
git commit -m "feat(privacy): bounded Laplace mechanism with budget accounting

Replaces add_noise. Sensitivity derives from required bounds (the old
sensitivity=1.0 default silently voided the guarantee outside [0,1]); epsilon
is spent before sampling; noise is inverse-CDF from secrets.SystemRandom, not
the seeded numpy PRNG.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB"
```

---

### Task 6: `generalize_amounts` — real k-anonymity

Spec §5. Replaces `quantize_amounts`, whose docstring falsely claimed k-anonymity from equal-width bins. Quantile edges, then merge any under-k bin into its smaller neighbour until every equivalence class has ≥ k members. Output per record is the midpoint of its class's min and max — a function of the class only, so it stays k-anonymous.

**Files:**
- Modify: `core/privacy.py` (add `generalize_amounts`; delete `quantize_amounts`)
- Modify: `tests/test_privacy.py` (add class)

**Interfaces:**
- Produces: `PrivacyManager.generalize_amounts(values: np.ndarray, *, bins: int = 10, k: int = 5) -> np.ndarray` (staticmethod) — same shape as input.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_privacy.py`:

```python
def _class_sizes(out: np.ndarray) -> np.ndarray:
    _, counts = np.unique(out, return_counts=True)
    return counts


class TestGeneralizeAmounts:
    """k-anonymity for the generalized column: every output value is shared by >= k records."""

    def test_every_equivalence_class_has_at_least_k(self):
        rng = np.random.default_rng(0)
        values = rng.lognormal(3.0, 1.0, size=500)  # skewed, like transaction volumes
        out = PrivacyManager.generalize_amounts(values, bins=10, k=7)
        assert (_class_sizes(out) >= 7).all()

    def test_holds_under_heavy_ties(self):
        # 95% of records share one value, like a fee_rate column with rare spikes.
        values = np.concatenate([np.full(950, 0.001), np.linspace(0.002, 0.005, 50)])
        out = PrivacyManager.generalize_amounts(values, bins=10, k=5)
        assert (_class_sizes(out) >= 5).all()

    def test_all_identical_values_form_one_class(self):
        out = PrivacyManager.generalize_amounts(np.full(20, 42.0), bins=10, k=5)
        assert np.unique(out).size == 1
        assert out[0] == 42.0

    def test_preserves_record_count_and_shape(self):
        values = np.arange(100, dtype=float).reshape(10, 10)
        out = PrivacyManager.generalize_amounts(values, bins=5, k=5)
        assert out.shape == (10, 10)

    def test_output_values_lie_within_input_range(self):
        values = np.random.default_rng(1).uniform(10, 100, 300)
        out = PrivacyManager.generalize_amounts(values, bins=10, k=5)
        assert out.min() >= values.min() and out.max() <= values.max()

    def test_rejects_n_below_k(self):
        with pytest.raises(ValueError, match="k=5"):
            PrivacyManager.generalize_amounts(np.arange(4.0), bins=2, k=5)

    @pytest.mark.parametrize("kwargs", [{"k": 1}, {"bins": 1}])
    def test_rejects_degenerate_parameters(self, kwargs):
        with pytest.raises(ValueError):
            PrivacyManager.generalize_amounts(
                np.arange(50.0), **{"bins": 5, "k": 5, **kwargs}
            )

    def test_small_bins_merge_rather_than_drop_records(self):
        # 3 outliers can't form their own class at k=5; they must join a neighbour.
        values = np.concatenate([np.linspace(0, 10, 97), [1000.0, 1001.0, 1002.0]])
        out = PrivacyManager.generalize_amounts(values, bins=10, k=5)
        assert out.size == 100
        assert (_class_sizes(out) >= 5).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_privacy.py::TestGeneralizeAmounts -q`
Expected: `AttributeError: type object 'PrivacyManager' has no attribute 'generalize_amounts'`

- [ ] **Step 3: Implement**

In `core/privacy.py`, delete the `quantize_amounts` static method and add inside `PrivacyManager`:

```python
# -- k-anonymity ----------------------------------------------------------


@staticmethod
def generalize_amounts(
    values: np.ndarray,
    *,
    bins: int = 10,
    k: int = 5,
) -> np.ndarray:
    """Generalize ``values`` so every output value is shared by >= ``k`` records.

    Quantile (equal-frequency) edges give balanced bins; any bin with fewer
    than ``k`` members is merged into its smaller neighbour until none
    remain. Each record is replaced by the midpoint of its class's min and
    max, a function of the class alone. This is k-anonymity for this column
    as released; features derived from it downstream carry no k claim.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    if n < k:
        raise ValueError(f"need at least k={k} records, got {n}")

    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        # every value identical: one class of size n >= k
        return np.full(np.shape(values), x[0])

    # bin i covers [edges[i], edges[i+1]); the last bin is closed on the right
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.clip(idx, 0, edges.size - 2)

    while True:
        counts = np.bincount(idx, minlength=edges.size - 1)
        small = np.flatnonzero((counts > 0) & (counts < k))
        if small.size == 0:
            break
        i = small[0]
        nonempty = np.flatnonzero(counts > 0)
        lower = nonempty[nonempty < i]
        upper = nonempty[nonempty > i]
        candidates = []
        if lower.size:
            candidates.append(lower[-1])
        if upper.size:
            candidates.append(upper[0])
        target = min(candidates, key=lambda b: (counts[b], b))
        idx[idx == i] = target

    out = np.empty(n)
    for b in np.unique(idx):
        members = idx == b
        out[members] = (x[members].min() + x[members].max()) / 2.0
    return out.reshape(np.shape(values))
```

Why it terminates: each iteration merges one non-empty bin into another, so the count of non-empty bins strictly decreases. If only one non-empty bin remains, it holds all `n ≥ k` records and `small` is empty. `candidates` is never empty when `small` is non-empty, because a lone non-empty bin would have `n ≥ k` members.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_privacy.py -q`
Expected: `30 passed`

- [ ] **Step 5: Full suite, ruff, and the coverage gate for this file**

Run: `pytest tests/ -q --cov=core.privacy --cov-report=term-missing && ruff check . && ruff format --check .`
Expected: `57 passed`; the coverage table shows `core/privacy.py ... 100%`. If any line is listed under `Missing`, add a test that reaches it before committing — the CI gate in Task 13 requires 100%.

- [ ] **Step 6: Commit**

```bash
git add core/privacy.py tests/test_privacy.py
git commit -m "feat(privacy): k-anonymous generalization replaces quantization

quantize_amounts claimed k-anonymity from equal-width bins, which guarantees
nothing. generalize_amounts uses quantile edges and merges under-k bins into a
neighbour until every equivalence class has >= k records.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB"
```

---

### Task 7: `PrivacyConfig` and `PrivacyFeature`

Spec §6. Validation happens at load time so misconfiguration fails at `from_yaml`, not at epoch 3.

**Files:**
- Modify: `core/config.py` (add two dataclasses after `TrainConfig`, line 57; extend `DignityConfig` at lines 59-95)
- Modify: `tests/test_core.py` (add `TestPrivacyConfig` after `TestDignityConfig`)

**Interfaces:**
- Produces: `PrivacyFeature(mechanism: str, epsilon: float | None = None, bounds: tuple[float, float] | None = None, bins: int = 10)`; `PrivacyConfig(epsilon_total: float, k: int = 5, key_env: str | None = None, features: dict[str, PrivacyFeature] = {})` with `.to_dict() -> dict`; `DignityConfig.privacy: PrivacyConfig | None = None`, loaded from a `privacy:` YAML block and written back by `to_yaml`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_core.py` (add `from core.config import DignityConfig, PrivacyConfig` in place of the existing `DignityConfig` import at line 7, and `import yaml` after `import pytest`):

```python
def _valid_privacy_block():
    return {
        "epsilon_total": 1.0,
        "k": 5,
        "key_env": None,
        "features": {
            "volume": {"mechanism": "laplace", "epsilon": 0.5, "bounds": [0, 1000]},
            "price": {"mechanism": "laplace", "epsilon": 0.5, "bounds": [0, 500]},
            "fee_rate": {"mechanism": "generalize", "bins": 10},
        },
    }


class TestPrivacyConfig:
    """Misconfiguration fails at load time, and absence means no privacy stage."""

    def test_absent_block_yields_none(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump({"model": {"task": "risk"}}))
        assert DignityConfig.from_yaml(str(path)).privacy is None

    def test_valid_block_round_trips_through_yaml(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump({"privacy": _valid_privacy_block()}))
        cfg = DignityConfig.from_yaml(str(path))
        assert cfg.privacy is not None
        assert cfg.privacy.features["volume"].bounds == (0.0, 1000.0)

        out = tmp_path / "out.yaml"
        cfg.to_yaml(str(out))
        again = DignityConfig.from_yaml(str(out))
        assert again.privacy.to_dict() == cfg.privacy.to_dict()

    def test_shipped_configs_match_spec(self):
        root = Path(__file__).resolve().parents[1] / "config"
        assert (
            DignityConfig.from_yaml(str(root / "train_risk.yaml")).privacy is not None
        )
        assert DignityConfig.from_yaml(str(root / "base.yaml")).privacy is None

    @pytest.mark.parametrize(
        "mutate,match",
        [
            (
                lambda b: b["features"]["volume"].update(mechanism="gaussian"),
                "unknown mechanism",
            ),
            (lambda b: b["features"]["volume"].update(epsilon=0.0), "epsilon > 0"),
            (lambda b: b["features"]["volume"].pop("bounds"), "requires bounds"),
            (lambda b: b["features"]["volume"].update(bounds=[10, 10]), "lo < hi"),
            (lambda b: b["features"]["fee_rate"].update(epsilon=0.1), "bins only"),
            (lambda b: b["features"]["fee_rate"].update(bounds=[0, 1]), "bins only"),
            (lambda b: b.update(k=1), "k must be"),
            (lambda b: b["features"]["fee_rate"].update(bins=1), "bins must be"),
            (
                lambda b: b["features"]["volume"].update(epsilon=0.9),
                "exceeds epsilon_total",
            ),
        ],
    )
    def test_rejects_bad_privacy_block(self, mutate, match):
        block = _valid_privacy_block()
        mutate(block)
        with pytest.raises(ValueError, match=match):
            PrivacyConfig(**block)
```

Add `from pathlib import Path` to the imports of `tests/test_core.py`.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_core.py::TestPrivacyConfig -q`
Expected: `ImportError: cannot import name 'PrivacyConfig'`

- [ ] **Step 3: Implement the dataclasses**

In `core/config.py`, after the `TrainConfig` class (after line 57) and before `@dataclass class DignityConfig`, add:

```python
_MECHANISMS = ("laplace", "generalize")


@dataclass
class PrivacyFeature:
    """One column's privacy mechanism. Exactly one of laplace / generalize."""

    mechanism: str
    epsilon: float | None = None
    bounds: tuple[float, float] | None = None
    bins: int = 10

    def __post_init__(self):
        if self.mechanism not in _MECHANISMS:
            raise ValueError(
                f"unknown mechanism {self.mechanism!r}; expected one of {_MECHANISMS}"
            )
        if self.bounds is not None:
            self.bounds = (float(self.bounds[0]), float(self.bounds[1]))
        if self.mechanism == "laplace":
            if self.epsilon is None or self.epsilon <= 0:
                raise ValueError("laplace requires epsilon > 0")
            if self.bounds is None:
                raise ValueError("laplace requires bounds")
            if self.bounds[0] >= self.bounds[1]:
                raise ValueError("bounds must satisfy lo < hi")
        else:
            if self.epsilon is not None or self.bounds is not None:
                raise ValueError("generalize takes bins only, not epsilon or bounds")
            if self.bins < 2:
                raise ValueError("bins must be at least 2")

    def to_dict(self) -> dict:
        d = {"mechanism": self.mechanism}
        if self.mechanism == "laplace":
            d["epsilon"] = self.epsilon
            d["bounds"] = list(self.bounds)
        else:
            d["bins"] = self.bins
        return d


@dataclass
class PrivacyConfig:
    """The `privacy:` block. Absent block => no privacy stage runs."""

    epsilon_total: float
    k: int = 5
    key_env: str | None = None
    features: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.epsilon_total <= 0:
            raise ValueError("epsilon_total must be positive")
        if self.k < 2:
            raise ValueError("k must be at least 2")
        self.features = {
            name: (f if isinstance(f, PrivacyFeature) else PrivacyFeature(**f))
            for name, f in self.features.items()
        }
        total = sum(
            f.epsilon for f in self.features.values() if f.mechanism == "laplace"
        )
        if total > self.epsilon_total + 1e-12:
            raise ValueError(
                f"sum of feature epsilons {total} exceeds epsilon_total {self.epsilon_total}"
            )

    def to_dict(self) -> dict:
        return {
            "epsilon_total": self.epsilon_total,
            "k": self.k,
            "key_env": self.key_env,
            "features": {n: f.to_dict() for n, f in self.features.items()},
        }
```

- [ ] **Step 4: Wire into `DignityConfig`**

In `DignityConfig`, after `train: TrainConfig = field(default_factory=TrainConfig)` (line 65) add:

```python
    privacy: PrivacyConfig | None = None
```

In `from_yaml`, inside the `return cls(` call, after `train=TrainConfig(**config_dict.get("train", {})),` add:

```python
privacy = (
    (PrivacyConfig(**config_dict["privacy"]) if config_dict.get("privacy") else None),
)
```

In `to_yaml`, after `config_dict = {...}` is built and before the `Path(path).parent.mkdir(...)` line, add:

```python
        if self.privacy is not None:
            config_dict["privacy"] = self.privacy.to_dict()
```

- [ ] **Step 5: Run — expect one remaining failure**

Run: `pytest tests/test_core.py::TestPrivacyConfig -q`
Expected: 11 pass; `test_shipped_configs_match_spec` FAILS because `config/train_risk.yaml` has no `privacy:` block yet. That is Task 9's job. Proceed.

- [ ] **Step 6: Everything else green**

Run: `pytest tests/ -q --deselect tests/test_core.py::TestPrivacyConfig::test_shipped_configs_match_spec && ruff check . && ruff format --check .`
Expected: `68 passed, 1 deselected`, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add core/config.py tests/test_core.py
git commit -m "feat(config): add PrivacyConfig with load-time validation

privacy: block is optional; presence is the switch. Rejects unknown
mechanisms, non-positive epsilon, missing or inverted bounds, generalize with
DP parameters, k<2, bins<2, and feature epsilons summing past epsilon_total.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB"
```

---

### Task 8: The privacy stage in `TransactionPipeline`

Spec §6. The stage runs on raw columns before `compute_signals`, exactly once per public call. `fit`/`transform`/`fit_transform` each call `_apply_privacy` once and delegate to private `_fit_on`/`_transform_on`.

**Files:**
- Modify: `data/pipeline.py:1-8` (imports), `:21-56` (`__init__`), `:87-140` (`fit`, `transform`, `fit_transform`)
- Modify: `tests/test_data.py` (add `TestPrivacyStage` before `class TestDataLoader:` at line 112)

**Interfaces:**
- Consumes: `PrivacyConfig`, `PrivacyFeature` (Task 7); `PrivacyBudget`, `PrivacyManager` (Tasks 3-6).
- Produces: `TransactionPipeline(seq_len=100, features=None, scaler_type="robust", privacy: PrivacyConfig | None = None, privacy_rng=None)`; attribute `.privacy_manager: PrivacyManager | None`; `_apply_privacy(df) -> df`.

- [ ] **Step 1: Write the failing tests**

Insert into `tests/test_data.py` before `class TestDataLoader:`. Add these imports at the top of the file: `import os`, `from core.config import PrivacyConfig`, `from core.privacy import BudgetExhausted`.

```python
class _ZeroNoise:
    def random(self) -> float:
        return 0.5


def _frame(n: int = 60) -> pd.DataFrame:
    t = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "volume": 100.0 + 10.0 * np.sin(t / 3.0),
            "price": t * 3.0,  # 0 .. 177, so clipping to [50, 150] bites on both ends
            "fee_rate": np.where(t % 20 == 0, 0.005, 0.001),
            "tx_count": (50 + (t % 7)).astype(float),
        }
    )


def _laplace_price(eps_total: float = 1.0, eps: float = 1.0) -> PrivacyConfig:
    return PrivacyConfig(
        epsilon_total=eps_total,
        features={
            "price": {"mechanism": "laplace", "epsilon": eps, "bounds": [50.0, 150.0]}
        },
    )


class TestPrivacyStage:
    """The stage runs once per public call, before signals, and only when configured."""

    def test_privacy_stage_runs_before_signals(self):
        df = _frame()
        priv = TransactionPipeline(
            seq_len=10,
            features=["price", "volatility"],
            privacy=_laplace_price(),
            privacy_rng=_ZeroNoise(),
        )
        x_priv, _ = priv.process(df)

        # Zero noise => the stage is pure clipping. A plain pipeline fed pre-clipped
        # prices must produce identical output — which is only true if clipping
        # happened BEFORE volatility was computed.
        pre_clipped = df.assign(price=df["price"].clip(50.0, 150.0))
        x_ref, _ = TransactionPipeline(
            seq_len=10, features=["price", "volatility"]
        ).process(pre_clipped)
        np.testing.assert_allclose(x_priv, x_ref)

        # and it must differ from the unclipped run, or the stage did nothing
        x_raw, _ = TransactionPipeline(
            seq_len=10, features=["price", "volatility"]
        ).process(df)
        assert not np.allclose(x_priv, x_raw)

    def test_fit_transform_spends_budget_exactly_once(self):
        priv = TransactionPipeline(
            seq_len=10,
            features=["price"],
            privacy=_laplace_price(),
            privacy_rng=_ZeroNoise(),
        )
        priv.process(_frame(), fit=True)
        assert priv.privacy_manager.budget.spent == pytest.approx(1.0)

    def test_budget_exhausted_propagates_from_transform(self):
        priv = TransactionPipeline(
            seq_len=10,
            features=["price"],
            privacy=_laplace_price(),
            privacy_rng=_ZeroNoise(),
        )
        priv.process(_frame(), fit=True)  # spends the whole budget
        with pytest.raises(BudgetExhausted):
            priv.process(_frame(), fit=False)

    def test_absent_privacy_block_is_a_noop(self):
        plain = TransactionPipeline(seq_len=10, features=["price"])
        df = _frame()
        assert plain.privacy_manager is None
        assert plain._apply_privacy(df) is df

    def test_generalize_column_is_k_anonymous_in_output(self):
        cfg = PrivacyConfig(
            epsilon_total=1.0,
            k=5,
            features={"fee_rate": {"mechanism": "generalize", "bins": 4}},
        )
        priv = TransactionPipeline(seq_len=10, features=["fee_rate"], privacy=cfg)
        generalized = priv._apply_privacy(_frame())["fee_rate"].to_numpy()
        _, counts = np.unique(generalized, return_counts=True)
        assert (counts >= 5).all()

    def test_missing_privacy_column_raises(self):
        cfg = PrivacyConfig(
            epsilon_total=1.0,
            features={
                "nope": {"mechanism": "laplace", "epsilon": 1.0, "bounds": [0, 1]}
            },
        )
        priv = TransactionPipeline(seq_len=10, features=["price"], privacy=cfg)
        with pytest.raises(ValueError, match="'nope'"):
            priv.process(_frame())

    def test_missing_key_env_var_fails_at_construction(self, monkeypatch):
        monkeypatch.delenv("DIGNITY_TEST_KEY_UNSET", raising=False)
        cfg = PrivacyConfig(epsilon_total=1.0, key_env="DIGNITY_TEST_KEY_UNSET")
        with pytest.raises(ValueError, match="DIGNITY_TEST_KEY_UNSET"):
            TransactionPipeline(seq_len=10, privacy=cfg)

    def test_key_env_var_is_read_into_the_manager(self, monkeypatch):
        monkeypatch.setenv("DIGNITY_TEST_KEY", "0123456789abcdef")
        cfg = PrivacyConfig(epsilon_total=1.0, key_env="DIGNITY_TEST_KEY")
        priv = TransactionPipeline(seq_len=10, privacy=cfg)
        assert len(priv.privacy_manager.pseudonymize("x")) == 64
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_data.py::TestPrivacyStage -q`
Expected: `TypeError: TransactionPipeline.__init__() got an unexpected keyword argument 'privacy'`

- [ ] **Step 3: Implement — imports and constructor**

In `data/pipeline.py`, replace the import block (lines 3-7) with:

```python
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from core.config import PrivacyConfig
from core.privacy import PrivacyBudget, PrivacyManager
from core.signals import SignalProcessor
```

Replace the `__init__` signature and add the privacy setup. Change:

```python
    def __init__(
        self, seq_len: int = 100, features: list = None, scaler_type: str = "robust"
    ):
```
to:
```python
    def __init__(
        self,
        seq_len: int = 100,
        features: list = None,
        scaler_type: str = "robust",
        privacy: PrivacyConfig | None = None,
        privacy_rng=None,
    ):
```

Extend the docstring `Args:` with:
```
            privacy: Optional privacy stage config. None => no stage runs.
            privacy_rng: Test seam forwarded to PrivacyManager(rng=...). Leave None.
```

At the end of `__init__`, after `self.fitted = False`, add:

```python
        self.privacy = privacy
        self.privacy_manager = None
        if privacy is not None:
            key = None
            if privacy.key_env is not None:
                key = os.environ.get(privacy.key_env)
                if key is None:
                    raise ValueError(
                        f"privacy.key_env={privacy.key_env!r} is not set in the environment"
                    )
            self.privacy_manager = PrivacyManager(
                PrivacyBudget(privacy.epsilon_total), key=key, rng=privacy_rng
            )
```

- [ ] **Step 4: Implement — the stage and the once-per-call structure**

Replace the existing `fit`, `transform`, and `fit_transform` methods (lines 87-140) with:

```python
def _apply_privacy(self, df: pd.DataFrame) -> pd.DataFrame:
    """Run the configured privacy mechanisms on raw columns.

    Runs BEFORE compute_signals so derived features inherit the guarantee
    by post-processing. Returns ``df`` itself when no privacy is configured.
    """
    if self.privacy_manager is None:
        return df
    result = df.copy()
    for name, feat in self.privacy.features.items():
        if name not in result.columns:
            raise ValueError(f"privacy configured for column {name!r}, not in data")
        col = result[name].to_numpy(dtype=float)
        if feat.mechanism == "laplace":
            result[name] = self.privacy_manager.add_laplace_noise(
                col, epsilon=feat.epsilon, bounds=feat.bounds
            )
        else:
            result[name] = PrivacyManager.generalize_amounts(
                col, bins=feat.bins, k=self.privacy.k
            )
    return result


def fit(self, df: pd.DataFrame) -> "TransactionPipeline":
    """Fit the scaler on training data. Applies the privacy stage once."""
    self._fit_on(self._apply_privacy(df))
    return self


def transform(self, df: pd.DataFrame) -> np.ndarray:
    """Transform to a scaled feature array. Applies the privacy stage once."""
    return self._transform_on(self._apply_privacy(df))


def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
    """Fit and transform on one privacy release, so the scaler sees the same draw."""
    prepared = self._apply_privacy(df)
    self._fit_on(prepared)
    return self._transform_on(prepared)


def _fit_on(self, df: pd.DataFrame) -> None:
    """Fit the scaler. ``df`` must already have had privacy applied."""
    df = self.compute_signals(df)

    available_features = [f for f in self.features if f in df.columns]
    if not available_features:
        raise ValueError(
            f"None of the specified features found in data: {self.features}"
        )

    X = df[available_features].values
    self.scaler.fit(X)
    self.fitted = True
    self.available_features = available_features


def _transform_on(self, df: pd.DataFrame) -> np.ndarray:
    """Scale features. ``df`` must already have had privacy applied."""
    if not self.fitted:
        raise RuntimeError("Pipeline must be fitted before transform")

    df = self.compute_signals(df)
    X = df[self.available_features].values
    return self.scaler.transform(X)
```

`create_sequences` and `process` are unchanged; `process` still calls `fit_transform` / `transform`, which is why it spends exactly once.

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_data.py -q`
Expected: `17 passed` (9 existing + 8 new).

- [ ] **Step 6: Full suite and ruff**

Run: `pytest tests/ -q --deselect tests/test_core.py::TestPrivacyConfig::test_shipped_configs_match_spec && ruff check . && ruff format --check .`
Expected: `76 passed, 1 deselected`, ruff clean. If ruff N806 flags `X` in the two new private methods, that name is pre-existing style in this file; add `# noqa: N806` on those two lines rather than renaming.

- [ ] **Step 7: Commit**

```bash
git add data/pipeline.py tests/test_data.py
git commit -m "feat(pipeline): privacy stage runs once per call, before signals

_apply_privacy applies one mechanism per configured raw column and runs ahead
of compute_signals so derived features inherit the guarantee. fit/transform/
fit_transform each apply it exactly once via _fit_on/_transform_on; the old
fit().transform() chaining would have double-spent epsilon and fit the scaler
on a different noise draw than it transformed.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB"
```

---

### Task 9: Ship the config and make the ledger visible

Spec §6. `train_risk.yaml` turns the stage on so the default path exercises it; `base.yaml` carries a commented example; `cli.py` prints what was spent.

**Files:**
- Modify: `config/train_risk.yaml` (append before `device: cuda`)
- Modify: `config/base.yaml` (append before `device: cuda`)
- Modify: `train/cli.py:56-64`

- [ ] **Step 1: Turn it on in `train_risk.yaml`**

Insert before the `device: cuda` line:

```yaml
# Privacy stage (see docs/PRIVACY.md). Runs on raw columns before signals.
# bounds are public parameters chosen a priori for the synthetic generator:
# volume walks from 100 with sigma 20/step; price walks from 100 at 1%/step.
privacy:
  epsilon_total: 1.0
  k: 5
  features:
    volume:   {mechanism: laplace, epsilon: 0.5, bounds: [0, 1000]}
    price:    {mechanism: laplace, epsilon: 0.5, bounds: [0, 500]}
    fee_rate: {mechanism: generalize, bins: 10}

```

- [ ] **Step 2: Document it in `base.yaml`**

Insert before the `device: cuda` line:

```yaml
# Privacy stage — commented out by default. No block => no privacy stage runs.
# See docs/PRIVACY.md for what each mechanism guarantees.
# privacy:
#   key_env: DIGNITY_PRIVACY_KEY     # env var NAME holding the HMAC key; never the key itself
#   epsilon_total: 1.0
#   k: 5
#   features:
#     volume:   {mechanism: laplace, epsilon: 0.5, bounds: [0, 10000]}
#     price:    {mechanism: laplace, epsilon: 0.5, bounds: [0, 1000]}
#     fee_rate: {mechanism: generalize, bins: 10}

```

- [ ] **Step 3: The deselected test now passes**

Run: `pytest tests/test_core.py::TestPrivacyConfig::test_shipped_configs_match_spec -q`
Expected: `1 passed`

- [ ] **Step 4: Pass the config through and print the ledger in `cli.py`**

Change the pipeline construction (lines 56-58):
```python
    pipeline = TransactionPipeline(
        seq_len=config.data.seq_len, features=config.data.features
    )
```
to:
```python
    pipeline = TransactionPipeline(
        seq_len=config.data.seq_len,
        features=config.data.features,
        privacy=config.privacy,
    )
```

After the `X_train, y_train = pipeline.process(...)` call (ends line 64), add:

```python
    if pipeline.privacy_manager is not None:
        budget = pipeline.privacy_manager.budget
        print(f"Privacy: ε spent {budget.spent:.3f} of {budget.epsilon_total:.3f}")
```

- [ ] **Step 5: Smoke the CLI for two epochs on CPU**

Create a throwaway config that shortens the run:
```bash
sed -e 's/epochs: 100.*/epochs: 2/' -e 's/device: cuda/device: cpu/' -e 's/use_amp: true/use_amp: false/' config/train_risk.yaml > /tmp/smoke.yaml
python -m train.cli --config /tmp/smoke.yaml 2>&1 | grep -E "Privacy: ε spent|Training complete"
```
Expected: `Privacy: ε spent 1.000 of 1.000` followed by `Training complete!`. Delete `/tmp/smoke.yaml`. (Checkpoints land in `./checkpoints/risk/`, which is gitignored.)

- [ ] **Step 6: Full suite and ruff**

Run: `pytest tests/ -q && ruff check . && ruff format --check .`
Expected: `77 passed`, ruff clean.

- [ ] **Step 7: Commit**

```bash
git add config/train_risk.yaml config/base.yaml train/cli.py
git commit -m "feat: enable the privacy stage in train_risk.yaml and report epsilon spent

base.yaml documents the block commented out; absence means no stage runs.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB"
```

---

### Task 10: Operator-layer tests

Spec §7. These are regression guards for properties the code already has. They should pass on first run; if one fails, the failure is a real finding — report it, do not weaken the test.

**Files:**
- Create: `tests/test_operator.py`
- Possibly modify: `export/to_onnx.py:38` (only if Step 3's conditional applies)

- [ ] **Step 1: Write the tests**

Create `tests/test_operator.py`:

```python
"""Operator-layer guarantees. See docs/THREAT-MODEL.md, Adversary B.

'Deniability' means exactly the three properties tested here: no network I/O in
the inference/export path, a self-contained ONNX artifact, nothing that phones home.
"""

import ast
import re
import socket
from pathlib import Path

import onnx
import torch

from export.to_onnx import export_to_onnx
from models.dignity import Dignity

ROOT = Path(__file__).resolve().parents[1]
INFERENCE_PATH = ("core", "data", "models", "export", "train")
NETWORK_MODULES = {
    "socket",
    "http",
    "urllib",
    "ssl",
    "requests",
    "aiohttp",
    "ccxt",
    "websocket",
    "websockets",
}
ABSOLUTE_PATH = re.compile(r"(^|\s)/(home|Users|tmp|var|opt|mnt)/")


def _imported_roots(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module.split(".")[0]


def _tiny_model() -> Dignity:
    return Dignity(task="risk", input_size=4, hidden_size=16, n_layers=1)


class TestOperatorLayer:
    def test_inference_path_imports_no_network_modules(self):
        offenders = []
        for pkg in INFERENCE_PATH:
            for py in (ROOT / pkg).rglob("*.py"):
                for root in _imported_roots(py):
                    if root in NETWORK_MODULES:
                        offenders.append(f"{py.relative_to(ROOT)} imports {root}")
        assert offenders == []

    def test_onnx_export_is_self_contained(self, tmp_path):
        out = tmp_path / "tiny.onnx"
        export_to_onnx(_tiny_model(), str(out), input_shape=(1, 20, 4), verify=False)

        model = onnx.load(str(out), load_external_data=False)
        external = [
            t.name
            for t in model.graph.initializer
            if t.data_location == onnx.TensorProto.EXTERNAL
        ]
        assert external == [], "ONNX must not reference external data files"

        text = " ".join(
            [
                model.doc_string,
                model.graph.doc_string,
                *(p.value for p in model.metadata_props),
            ]
        )
        assert "://" not in text, "artifact metadata must not embed URLs"
        assert not ABSOLUTE_PATH.search(text), (
            "artifact metadata must not embed local paths"
        )

    def test_predict_succeeds_with_sockets_disabled(self, monkeypatch):
        def refuse(*args, **kwargs):
            raise AssertionError("inference attempted to open a socket")

        monkeypatch.setattr(socket, "socket", refuse)
        out = _tiny_model().predict(torch.randn(2, 20, 4))
        assert out.shape[0] == 2
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_operator.py -v`
Expected: `3 passed`.

- [ ] **Step 3: Conditional — only if `test_onnx_export_is_self_contained` errors inside `torch.onnx.export`**

Newer torch releases default `torch.onnx.export` to the dynamo exporter, which rejects or warns on `dynamic_axes`. If and only if the test fails with an error originating in `torch.onnx.export`, edit `export/to_onnx.py` at the `torch.onnx.export(` call and add `dynamo=False,` as the last keyword argument. Re-run. This is the sole permitted edit to `export/to_onnx.py`. If the failure is anything else, stop and report it.

- [ ] **Step 4: Full suite and ruff**

Run: `pytest tests/ -q && ruff check . && ruff format --check .`
Expected: `80 passed`, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add tests/test_operator.py
git add -u export/to_onnx.py 2>/dev/null || true
git commit -m "test: enforce operator-layer guarantees

No network imports on the inference/export path (AST scan), self-contained
ONNX artifact, and prediction with sockets disabled. These are what
'deniability' means in docs/THREAT-MODEL.md.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB"
```

---

### Task 11: Documentation — make the words match the code

Spec §3, §4, §7, §10. Write the threat model, rewrite `PRIVACY.md`, fix the phantom API snippets, and bring README's privacy sections in line. Re-read the Global Constraints denylist before writing each file.

**Files:**
- Create: `docs/THREAT-MODEL.md`
- Rewrite: `docs/PRIVACY.md`
- Modify: `docs/ARCHITECTURE.md:45-51`, `:333-335`
- Modify: `docs/QUICK_START.md:91-104`
- Modify: `README.md:5`, `:7`, `:14`, `:24`, `:73-96`, `:131-135`, `:186-194`, `:372`, `:380-381`

- [ ] **Step 1: Write `docs/THREAT-MODEL.md`**

```markdown
# Threat Model

Dignity makes two privacy claims for two different people. This page says exactly
what each claim covers, what it does not, and what "deniability" means here. Every
property below is enforced by a named test; if the test does not exist, the
property is not claimed.

## Adversary A — holds the training artifacts

Anyone with the trained weights, the ONNX export, or the preprocessed feature
arrays. This is the **subject layer**: it protects individuals whose transactions
appear in the training data.

**Defended.** Every raw value released into training passes through one of:

- *Bounded Laplace noise* — ε-differential privacy at the level of each released
  value. Values are clipped to public `bounds`, sensitivity is the width of those
  bounds, and every call spends ε from a per-pipeline ledger that refuses to
  exceed `epsilon_total`. Tests: `TestAddLaplaceNoise`, `TestPrivacyBudget`,
  `TestPrivacyStage::test_fit_transform_spends_budget_exactly_once`.
- *k-anonymous generalization* — every value in the generalized column is shared
  by at least k records. Test: `TestGeneralizeAmounts::test_every_equivalence_class_has_at_least_k`.

Signals derived from a noised column (volatility, momentum, directional change)
inherit its ε by the post-processing theorem, because the privacy stage runs
before signal computation. Test: `TestPrivacyStage::test_privacy_stage_runs_before_signals`.

**Not defended.**

- An adversary who already holds the raw source data. No input mechanism helps
  once the input is theirs.
- Memorization inside the trained weights beyond what input noise suppresses.
  Training is ordinary SGD; the trained model carries no DP guarantee of its own.
- k-anonymity of features *derived* from a generalized column. Rolling windows
  can re-split equivalence classes. The k claim is for the column as released.

## Adversary B — observes the operator's machine or network

Someone watching the person who runs the model. This is the **operator layer**.

**Defended — and this is the whole definition of "deniability" in this project:**

1. The inference and export path performs no network I/O.
   Test: `TestOperatorLayer::test_inference_path_imports_no_network_modules`.
2. The ONNX artifact is self-contained: no external data files, no embedded URLs,
   no embedded local paths. Test: `TestOperatorLayer::test_onnx_export_is_self_contained`.
3. Prediction works with sockets disabled; nothing phones home.
   Test: `TestOperatorLayer::test_predict_succeeds_with_sockets_disabled`.

**Not defended.**

- An adversary with access to the operator's disk. Artifacts are not encrypted
  at rest.
- Traffic analysis of whatever the operator does *with* the predictions.

## Known limitations

- **Floating-point Laplace.** The noise is sampled by inverse CDF in IEEE
  floating point, which is vulnerable to the attack in Mironov, *On Significance
  of the Least Significant Bits for Differential Privacy* (CCS 2012). The
  snapping mechanism that fixes this is not implemented.
- **`bounds` are public.** Choosing them from the data would leak its extremes,
  so they must be set a priori in config. They are visible to any adversary.
- **Pseudonymization is a primitive, not a pipeline stage.** No current data
  source carries identifier columns, so `PrivacyManager.pseudonymize` is exposed
  and tested but not wired into `TransactionPipeline`.
- **Validation is on synthetic data.** No real transaction data has passed
  through this pipeline. The mechanisms are properties of the algorithm and hold
  regardless; empirical re-identification risk on real subjects is untested.
```

- [ ] **Step 2: Rewrite `docs/PRIVACY.md`**

Replace the entire file with:

```markdown
# Privacy Operations

`core/privacy.py` implements two things, and claims exactly two things. Each
sentence in **bold** below is backed by a named test in `tests/test_privacy.py`.
For who these protect and from whom, read [THREAT-MODEL.md](THREAT-MODEL.md).

## What is claimed

| Guarantee | Mechanism | Wording we use | Wording we do not use |
|---|---|---|---|
| Identifier protection | HMAC-SHA256 with a required key | *keyed pseudonymization* | anonymization |
| Value protection | Clipped Laplace noise, ε ledgered | *input-level ε-differential privacy* | any DP claim about the trained model |
| Value protection | Quantile bins merged to ≥ k | *k-anonymity for the column as released* | k claims about derived features |

## Setup

```python
import numpy as np
from core.privacy import PrivacyBudget, PrivacyManager

# One budget per release. Epsilons add across calls; overspending raises.
budget = PrivacyBudget(epsilon_total=1.0)

# The key is only needed for pseudonymization. Load it from the environment;
# a literal here is for illustration only.
pm = PrivacyManager(budget, key=b"replace-with-16+-random-bytes")
```

## Keyed pseudonymization

**Same key, same identifier → same pseudonym. Different key → unrelated pseudonym.**
The key is required at call time; there is no unkeyed fallback. Keys under 16
bytes are rejected.

```python
pseudonym = pm.pseudonymize("0x1234abcd5678ef90")  # 64 hex chars
many = pm.pseudonymize_many(["addr_a", "addr_b", "addr_a"])  # many[0] == many[2]
```

This is pseudonymization, not anonymization: anyone holding the key can link
records. That is the intended property — it lets you join across your own
datasets while making the pseudonyms useless to anyone without the key.

## Bounded Laplace noise (input-level ε-DP)

**Values are clipped to `bounds` before noising, so sensitivity is `hi - lo` and
cannot be understated.** **ε is spent from the budget before any sample is drawn.**
**Noise comes from `secrets.SystemRandom`, not a seedable PRNG.**

```python
amounts = np.array([123.4, 789.0, 456.7, 5000.0])  # 5000 will be clipped to 1000
noisy = pm.add_laplace_noise(amounts, epsilon=0.5, bounds=(0.0, 1000.0))
budget.spent  # 0.5
budget.remaining  # 0.5
pm.add_laplace_noise(
    amounts, epsilon=0.6, bounds=(0.0, 1000.0)
)  # raises BudgetExhausted
```

Each released value is ε-differentially private for that feature (local DP).
Spending ε₁ on one column and ε₂ on another composes to ε₁ + ε₂; the ledger
enforces that the sum never exceeds `epsilon_total`.

`bounds` are **public parameters**. Do not derive them from the data — that leaks
the extremes. Choose them from domain knowledge before you look.

## k-anonymous generalization

**Every output value is shared by at least k records.** Quantile edges start the
bins balanced; any bin with fewer than k members is merged into its smaller
neighbour until none remain. Each record becomes the midpoint of its class's
min and max.

```python
volumes = np.random.default_rng(0).lognormal(3, 1, size=500)
generalized = PrivacyManager.generalize_amounts(volumes, bins=10, k=5)
```

This holds for the generalized column as released. Signals computed from it
downstream (rolling volatility, momentum) are **not** covered by the k claim.

## The privacy stage in the pipeline

`TransactionPipeline` applies these mechanisms to raw columns **before**
computing signals, so derived features inherit the DP guarantee by
post-processing. The stage runs exactly once per `fit`, `transform`, or
`fit_transform`. It is driven by a `privacy:` block in the YAML config:

```yaml
privacy:
  key_env: DIGNITY_PRIVACY_KEY       # env var NAME; never the key itself
  epsilon_total: 1.0
  k: 5
  features:
    volume:   {mechanism: laplace, epsilon: 0.5, bounds: [0, 1000]}
    price:    {mechanism: laplace, epsilon: 0.5, bounds: [0, 500]}
    fee_rate: {mechanism: generalize, bins: 10}
```

**No `privacy:` block means no privacy stage runs.** Misconfiguration — an unknown
mechanism, ε ≤ 0, missing or inverted bounds, k < 2, or feature epsilons summing
past `epsilon_total` — fails at config load, not mid-training.

`dignity-train` prints `Privacy: ε spent X of Y` after preprocessing so the
ledger is visible, not theoretical.

## What is not here

- No differential privacy on the trained weights (no DP-SGD). The model itself
  carries no DP guarantee.
- No secure aggregation, no federated learning.
- No snapping mechanism; see THREAT-MODEL.md on floating-point Laplace.

## Reference

- Dwork & Roth, *The Algorithmic Foundations of Differential Privacy* (2014) —
  Laplace mechanism (§3.3), sequential composition (§3.5), post-processing (Prop. 2.1).
- Sweeney, *k-Anonymity: A Model for Protecting Privacy* (2002).
- Mironov, *On Significance of the Least Significant Bits for Differential Privacy* (CCS 2012).
```

- [ ] **Step 3: Fix `docs/ARCHITECTURE.md`**

Replace lines 45-51 (the `**privacy.py**` block through its closing fence) with:

```markdown
**privacy.py** - Privacy primitives
```python
from core.privacy import PrivacyBudget, PrivacyManager

pm = PrivacyManager(PrivacyBudget(epsilon_total=1.0), key=b"16+-byte-secret-key")
pseudonym = pm.pseudonymize("user_id_value")
noisy = pm.add_laplace_noise(amounts, epsilon=0.5, bounds=(0.0, 1000.0))
```
```

Replace lines 333-335 (`# 2. Apply privacy` and the two calls) with:

```python
# 2. Apply privacy (see docs/PRIVACY.md) — normally done by TransactionPipeline's
#    privacy stage from the `privacy:` config block, before signals are computed
pm = PrivacyManager(PrivacyBudget(epsilon_total=1.0))
amounts = pm.add_laplace_noise(np.array(raw_data["amount"]), epsilon=1.0, bounds=(0.0, 500.0))
```

Leave the signal-processing lines that follow untouched (their drift is logged separately).

- [ ] **Step 4: Fix `docs/QUICK_START.md`**

Replace lines 91-104 (the `## Privacy Features` heading through its closing fence) with:

```markdown
## Privacy Features

Turn on the privacy stage with a `privacy:` block in your config; it runs on raw
columns before signals are computed. See `config/train_risk.yaml` for a live
example and [PRIVACY.md](PRIVACY.md) for what each mechanism guarantees.

```python
import numpy as np
from core.privacy import PrivacyBudget, PrivacyManager

pm = PrivacyManager(PrivacyBudget(epsilon_total=1.0), key=b"16+-byte-secret-key")
pseudonyms = pm.pseudonymize_many(["user_a", "merchant_1"])
noisy = pm.add_laplace_noise(np.array([100.0, 250.0]), epsilon=1.0, bounds=(0.0, 1000.0))
generalized = PrivacyManager.generalize_amounts(np.random.uniform(10, 100, 200), bins=10, k=5)
```
```

- [ ] **Step 5: Fix `README.md`**

Line 5 — replace the static badge:
```markdown
[![CI](https://github.com/crichalchemist/Dignity/actions/workflows/ci.yml/badge.svg)](https://github.com/crichalchemist/Dignity/actions/workflows/ci.yml)
```

Line 7 — change `with built-in privacy safeguards including differential privacy and secure data handling` to `with an input-level differential-privacy stage, k-anonymous generalization, and keyed pseudonymization — each backed by a test`.

Line 14 — change `Built-in hashing, anonymization, quantization, and differential privacy operations` to `Keyed pseudonymization, k-anonymous generalization, and input-level ε-DP with a budget ledger`.

Line 24 — change `Hashing, anonymization, quantization, differential privacy for sensitive transaction data` to `Keyed pseudonymization, bounded Laplace noise with ε accounting, k-anonymous generalization — see docs/PRIVACY.md`.

Lines 73-96 — replace the `#### Privacy Operations` example block with:

```markdown
#### Privacy Operations

```python
import numpy as np
from core.privacy import PrivacyBudget, PrivacyManager

# One budget per release; epsilons add and overspending raises BudgetExhausted.
pm = PrivacyManager(PrivacyBudget(epsilon_total=1.0), key=b"load-this-from-the-environment")

# Keyed pseudonymization (HMAC-SHA256): linkable under one key, unlinkable across keys
pseudonym = pm.pseudonymize("0x1234abcd5678ef90")

# Bounded Laplace noise: clipped to bounds, sensitivity = hi - lo, spends epsilon
amounts = np.array([123.456, 789.012, 456.789])
noisy = pm.add_laplace_noise(amounts, epsilon=0.5, bounds=(0.0, 1000.0))

# k-anonymous generalization: every output value is shared by >= k records
generalized = PrivacyManager.generalize_amounts(np.random.uniform(10, 100, 200), bins=10, k=5)
```

In training, none of this is called by hand: a `privacy:` block in the config
turns on a pipeline stage that runs before signals are computed. **No block, no
stage.** See `config/train_risk.yaml` and [docs/PRIVACY.md](docs/PRIVACY.md).
```

Lines 131-135 — in the Data Pipeline Flow diagram replace the `Privacy Operations (core/privacy.py)` node and its four bullets with:

```
Privacy Stage (core/privacy.py — optional, from the `privacy:` config block)
├── Bounded Laplace noise (ε-DP, budget-ledgered)
└── k-anonymous generalization (quantile bins)
```

Lines 186-194 — replace the `### Privacy Operations` section with:

```markdown
### Privacy Operations

`core/privacy.py` claims exactly three things, each enforced by a test:

- **Keyed pseudonymization** (HMAC-SHA256, key required) — linkable under one key, unlinkable across keys
- **Input-level ε-differential privacy** — clipped Laplace noise with sensitivity derived from public bounds and ε spent against a ledger that refuses to overspend
- **k-anonymity** for generalized columns — quantile bins merged until every class has ≥ k records

The stage runs on raw columns *before* signal computation, so derived features inherit the DP guarantee. What it does **not** claim — and why — is in [docs/THREAT-MODEL.md](docs/THREAT-MODEL.md).
```

Line 372 — replace the `**Deniable**` bullet with:

```markdown
2. **Deniable**: Local-only inference, self-contained artifact, nothing phones home — three properties, three tests. Defined precisely in [docs/THREAT-MODEL.md](docs/THREAT-MODEL.md).
```

Lines 380-381 — in Contributing, change:
```markdown
- Run tests before submitting: `pytest tests/ -v`
- Follow existing code style (ruff format)
```
to:
```markdown
- Install the hooks once: `pip install pre-commit && pre-commit install`
- Run tests before submitting: `pytest tests/ -v`
- Style is enforced by ruff at line length 88; the hooks run it for you
```

- [ ] **Step 6: Verify no denylisted string survives in public docs**

Run:
```bash
grep -rn "Secure Aggregation\|hash_identifiers(\|anonymize_addresses\|sanitize_dataset\|suppress_rare_events\|differentially private model\|\.hash_identifier(\|\.add_noise(\|\.quantize_amounts(\|anonymize_amounts(\|add_differential_privacy_noise(" README.md docs/ --include=*.md | grep -v "docs/plans/\|docs/superpowers/"
```
Expected: no output. Fix any hit before continuing.

- [ ] **Step 7: Commit**

```bash
git add docs/THREAT-MODEL.md docs/PRIVACY.md docs/ARCHITECTURE.md docs/QUICK_START.md README.md
git commit -m "docs: state exactly what the privacy layer guarantees, and what it does not

New THREAT-MODEL.md defines two adversaries and gives 'deniability' a testable
meaning. PRIVACY.md is rewritten around the real API. Phantom functional
snippets in ARCHITECTURE.md and QUICK_START.md are replaced. README privacy
sections match the code; the static tests badge becomes the CI badge.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB"
```

---

### Task 12: Docs regression test

Spec §8. Points the "no test, no claim" rule at the docs themselves. Historical records (`docs/plans/`, `docs/superpowers/`) are excluded — they legitimately name the removed API, and the spec for this very work would otherwise fail its own test.

**Files:**
- Create: `tests/test_docs.py`

- [ ] **Step 1: Write the test**

```python
"""Public docs must not reintroduce claims or APIs this project removed."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Removed claims and removed API names. Historical records under docs/plans/ and
# docs/superpowers/ are allowed to mention them; public docs are not.
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
)
EXCLUDED_PREFIXES = ("docs/plans/", "docs/superpowers/")


def _public_docs():
    yield ROOT / "README.md"
    for path in sorted((ROOT / "docs").rglob("*.md")):
        rel = path.relative_to(ROOT).as_posix()
        if not rel.startswith(EXCLUDED_PREFIXES):
            yield path


class TestDocs:
    def test_docs_do_not_reintroduce_removed_claims(self):
        hits = []
        for path in _public_docs():
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                for term in REMOVED:
                    if term in line:
                        hits.append(f"{path.relative_to(ROOT)}:{lineno}: {term!r}")
        assert hits == []
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_docs.py -v`
Expected: `1 passed`. If it fails, the assertion message lists file:line:term — fix the doc, not the test.

- [ ] **Step 3: Prove it bites**

```bash
echo "Secure Aggregation" >> docs/PRIVACY.md
pytest tests/test_docs.py -q; git checkout docs/PRIVACY.md
```
Expected: `1 failed` on the first command, then the file is restored. Confirm `git status --short docs/PRIVACY.md` is empty.

- [ ] **Step 4: Full suite and ruff**

Run: `pytest tests/ -q && ruff check . && ruff format --check .`
Expected: `81 passed`, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add tests/test_docs.py
git commit -m "test: fail the build if public docs reintroduce removed privacy claims

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB"
```

---

### Task 13: CI, pre-commit, and repo hygiene

Spec §9. The workflow must mirror Task 0's install steps exactly. `.coverage` is currently **tracked** in git and must be untracked, not merely ignored.

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.pre-commit-config.yaml`
- Modify: `setup.py:30-36` (`extras_require`)
- Modify: `.gitignore` (the `# Pytest` block, line 38)
- Untrack: `.coverage`

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: ci

on:
  push:
    branches: [master]
  pull_request:

permissions:
  contents: read

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff
      - run: ruff check .
      - run: ruff format --check .

  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - run: pip install -r requirements.txt
      - run: pip install -e . --no-deps
      - run: pytest tests/ --cov=core.privacy --cov-fail-under=100
```

- [ ] **Step 2: Write the pre-commit config**

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

If `pre-commit install && pre-commit run --all-files` in Step 5 reports that `v0.15.9` does not exist, replace `rev` with the output of `ruff --version` prefixed by `v`.

- [ ] **Step 3: Add pre-commit to dev extras**

In `setup.py`, change:
```python
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "ruff>=0.1.0",
        ],
```
to:
```python
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "ruff>=0.1.0",
            "pre-commit>=3.0",
        ],
```

- [ ] **Step 4: Untrack and ignore coverage output**

```bash
git rm --cached .coverage
```

In `.gitignore`, extend the `# Pytest` block:
```
# Pytest
.pytest_cache/
.cache/
.coverage
htmlcov/
```

- [ ] **Step 5: Run the gates locally exactly as CI will**

```bash
ruff check . && ruff format --check .
pytest tests/ --cov=core.privacy --cov-fail-under=100
pre-commit install && pre-commit run --all-files
```
Expected: ruff clean; `81 passed` with `Required test coverage of 100% reached`; every pre-commit hook `Passed`. If the coverage gate fails, the report lists the uncovered lines in `core/privacy.py` — add a test in `tests/test_privacy.py` that reaches them and commit that first.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/ci.yml .pre-commit-config.yaml setup.py .gitignore
git commit -m "ci: lint and test on every push with 100% coverage required on core/privacy.py

lint job runs ruff check + format --check at width 88. test job runs the suite on
Python 3.10 and 3.12 with torch from the CPU index. Adds ruff pre-commit hooks,
untracks .coverage.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB"
```

---

### Task 14: Bring `CLAUDE.md` and the spec up to date

Spec §10. `CLAUDE.md` currently documents three things this work made false. The spec's test count predates parametrization.

**Files:**
- Modify: `.claude/CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-09-03-privacy-honest-core-design.md` (§8 Count, §12 first checkbox)

- [ ] **Step 1: Fix the Commands block in `CLAUDE.md`**

Replace:
```
# Lint (ruff is the only installed tool; see "Conventions" before running `ruff format`)
ruff check .
ruff check --fix .
ruff format --line-length 88 <files-you-touched>   # NOT a blanket `ruff format .`
```
with:
```
# Lint / format — ruff only, line length 88, enforced by CI and pre-commit
ruff check .
ruff format .
pre-commit run --all-files

# The CI coverage gate (core/privacy.py must be 100%)
pytest tests/ --cov=core.privacy --cov-fail-under=100
```

Also change the tests line `# Tests (31 total: test_core 11, test_data 9, test_models 11)` to `# Tests (81 collected: test_core 19, test_data 17, test_models 11, test_privacy 30, test_operator 3, test_docs 1)`.

- [ ] **Step 2: Fix the Architecture section**

Replace the line in the flow diagram:
```
core/privacy.py                     →  optional, caller-invoked (nothing in the pipeline calls it)
```
with:
```
core/privacy.py                     →  privacy stage: runs once per fit/transform, BEFORE signals
```

Replace:
```
`core/privacy.PrivacyManager` and `core/signals.SignalProcessor` are **all-static utility classes**.
Do not instantiate them with constructor state.
```
with:
```
`core/signals.SignalProcessor` is an **all-static utility class**. `core/privacy.PrivacyManager`
is **instance-based**: it is bound to a `PrivacyBudget` (ε ledger), an optional HMAC key, and
an injectable `rng`. `TransactionPipeline` builds one per instance from `DignityConfig.privacy`.
Every public privacy claim is backed by a named test — see `docs/THREAT-MODEL.md`. If you add
a mechanism, it needs a test in `tests/test_privacy.py` or CI's 100% gate on that file fails.
```

- [ ] **Step 3: Fix the Conventions section**

Replace the two bullets beginning `- **Formatter width is contested` and `- \`ruff check .\` (the lint half)` with:

```
- `ruff format` at line length 88 is the formatter; `ruff check` (`E,W,F,I,N,UP,B,C4,SIM`) is
  the linter. Both run in CI's `lint` job and in pre-commit. Width is settled; `pyproject.toml`
  and `.editorconfig` agree.
```

Add a bullet after the `Config is dataclass-backed` bullet:

```
- A `privacy:` block in YAML is optional and validated at load (`PrivacyConfig`). No block =
  no privacy stage. `bounds` are public parameters chosen a priori — never derive them from data.
```

- [ ] **Step 4: Amend the spec's count**

In the spec, replace:
```
31 − 4 + 18 + 3 + 1 + 5 = **54**. Every new test maps to a sentence in §3, §6, or §7.
```
with:
```
70 test functions; **81 collected** once parametrized cases expand (test_privacy 30,
test_core 19, test_data 17, test_models 11, test_operator 3, test_docs 1). Every new test
maps to a sentence in §3, §6, or §7.
```
and in §12 replace `54 tests` with `81 collected`.

- [ ] **Step 5: Verify the counts you just wrote**

Run: `pytest tests/ --collect-only -q | tail -1`
Expected: `81 tests collected`. If the number differs, fix the number in both files to match reality — the number on disk is the truth, the plan is not.

- [ ] **Step 6: Commit**

```bash
git add .claude/CLAUDE.md docs/superpowers/specs/2026-09-03-privacy-honest-core-design.md
git commit -m "docs: update CLAUDE.md and spec for the instance-based privacy layer

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013oaN2krzBYvGxn9HgRPsUB"
```

Note: `.claude/CLAUDE.md` was untracked before this task. `git add` stages it for the first time. If you would rather it stay out of the repo, stop and ask before this commit.

---

### Task 15: Final verification and handoff

Spec §12. Every criterion is a command. Run them all; report the output, not a summary.

- [ ] **Step 1: The whole suite, on the CI matrix's floor if available**

```bash
pytest tests/ -v --cov=core.privacy --cov-fail-under=100 2>&1 | tail -15
```
Expected: `81 passed`, coverage 100%.

- [ ] **Step 2: Lint and format**

```bash
ruff check . && ruff format --check . && pre-commit run --all-files
```
Expected: all clean.

- [ ] **Step 3: Spec §12 grep criteria**

```bash
grep -rn "suppress_rare_events\|sanitize_dataset\|Secure Aggregation\|hash_identifiers(" README.md docs/ --include=*.md | grep -v "docs/plans/\|docs/superpowers/" ; echo "exit=$?"
grep -rn "sensitivity=" --include=*.py . | grep -v ".venv" ; echo "exit=$?"
```
Expected: both print only `exit=1` (no matches).

- [ ] **Step 4: The CLI prints the ledger**

```bash
sed -e 's/epochs: 100.*/epochs: 1/' -e 's/device: cuda/device: cpu/' -e 's/use_amp: true/use_amp: false/' config/train_risk.yaml > /tmp/smoke.yaml
python -m train.cli --config /tmp/smoke.yaml 2>&1 | grep "Privacy: ε spent"; rm /tmp/smoke.yaml
```
Expected: `Privacy: ε spent 1.000 of 1.000`

- [ ] **Step 5: Review the branch**

```bash
git log --oneline master..privacy-honest-core
git diff --stat master..privacy-honest-core | tail -1
```
Expected: roughly 16 commits; every one signed (`git log --format='%h %G? %s' master..privacy-honest-core` shows `G` on each).

- [ ] **Step 6: Hand off — do not push without the user's say-so**

Report the outputs of Steps 1-5 verbatim. Then ask the user whether to:
- `git push -u origin privacy-honest-core` and open a PR against `master` (CI runs on the PR; the spec's "green on both matrix legs" criterion is checked there), or
- hold the branch locally.

After the first green CI run, remind the user to enable branch protection on `master` requiring the `lint` and `test` checks — that is a GitHub setting, not code.

---

## Self-review against the spec

**Coverage.** §3 contract → Tasks 4-6 (mechanisms), 11 (wording), 12 (enforcement). §4 purge → Tasks 2, 11 (plus `docs/QUICK_START.md`, which the spec's list missed but the denylist test would have caught). §5 primitives → Tasks 3-6, including the key-optional-at-construction tweak from §6. §6 wiring/config → Tasks 7-9, including once-per-call and `privacy_rng`. §7 threat model → Task 11 Step 1; enforcing tests → Task 10. §8 tests → Tasks 3-8, 10, 12; the config-rejection cases moved from `test_data.py` to `test_core.py::TestPrivacyConfig` because they test `core/config.py`. §9 CI → Tasks 1, 13. §10 docs → Tasks 11, 14. §11 out of scope → untouched; Task 10 Step 3's `dynamo=False` is the single permitted exception, conditional, and named. §12 criteria → Task 15.

**Deviations from the spec, all deliberate and stated in the task that makes them:** the denylist gains the removed *old* method names and excludes `docs/plans/`+`docs/superpowers/` (Task 12); `_apply_privacy` raises on a configured column that is absent rather than skipping it (Task 8) — silent skipping is the behavior this whole project is against; the test count is 81 collected / 70 functions (Task 14 amends the spec).

**Type consistency.** `PrivacyManager(budget, key=None, rng=None)` — Tasks 4, 5, 8, 11 all use this order. `add_laplace_noise(values, *, epsilon, bounds)` — keyword-only everywhere. `generalize_amounts(values, *, bins=10, k=5)` — staticmethod, called as `PrivacyManager.generalize_amounts` in Tasks 6, 8, 11. `PrivacyConfig(epsilon_total, k=5, key_env=None, features={})` — Tasks 7, 8, 9. `PrivacyFeature.bounds` is a tuple after `__post_init__`; YAML supplies a list; `to_dict` emits a list — Task 7 round-trip test covers it. `TransactionPipeline(..., privacy=None, privacy_rng=None)` — Tasks 8, 9. `pipeline.privacy_manager.budget.spent` / `.epsilon_total` — Tasks 8, 9.

**Placeholders.** None. Every code step has code; every run step has an expected result.
