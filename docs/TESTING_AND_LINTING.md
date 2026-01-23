# Testing and Linting Workflow

This document describes the code quality infrastructure for the Sequence trading system.

## Quick Start

```bash
# Run quality gate before committing
./scripts/quality_gate.sh

# Or run components individually:
./scripts/run_lint.sh          # Check code quality
./scripts/run_tests.sh         # Run test suite
./scripts/cleanup_codebase.sh  # Auto-fix linting issues
```

## Scripts Overview

### 1. Quality Gate (`quality_gate.sh`)

**Purpose:** Pre-commit/pre-push quality check
**When to use:** Before creating commits or PRs

```bash
# Full check (lint + all tests)
./scripts/quality_gate.sh

# Fast check (lint + smoke tests only)
./scripts/quality_gate.sh --fast
```

**Exit codes:**

- `0`: All checks passed ✅
- `1`: Checks failed ❌

---

### 2. Linting (`run_lint.sh`)

**Purpose:** Code quality and style checking using [Ruff](https://github.com/astral-sh/ruff)

```bash
# Check for issues (no changes)
./scripts/run_lint.sh

# Auto-fix safe issues
./scripts/run_lint.sh --fix

# Check formatting only
./scripts/run_lint.sh --check-only
```

**What it checks:**

- Code style (PEP 8)
- Unused imports/variables
- Type hint modernization (PEP 585, PEP 604)
- Common bugs (flake8-bugbear)
- Import sorting (isort)
- Code simplifications

**Configuration:** `pyproject.toml` (`[tool.ruff]` section)

---

### 3. Testing (`run_tests.sh`)

**Purpose:** Run pytest test suite

```bash
# Full test suite
./scripts/run_tests.sh

# Fast mode (smoke tests only)
./scripts/run_tests.sh --fast

# Verbose output
./scripts/run_tests.sh --verbose

# Run specific tests
./scripts/run_tests.sh --pattern test_phase3
```

**Test categories:**

- `test_smoke.py`: Quick sanity checks
- `test_phase*.py`: Phase-specific validation
- `test_integration_*.py`: End-to-end tests

---

### 4. Codebase Cleanup (`cleanup_codebase.sh`)

**Purpose:** Bulk auto-fix linting issues

```bash
# Show what would be fixed (dry run)
./scripts/cleanup_codebase.sh --dry-run

# Apply safe auto-fixes
./scripts/cleanup_codebase.sh

# Apply safe + unsafe fixes (aggressive)
./scripts/cleanup_codebase.sh --aggressive
```

**What it does:**

1. Shows current linting statistics
2. Applies safe auto-fixes (~1,860 fixes)
3. Formats code with ruff formatter
4. Reports remaining issues

**⚠️ Warning:** Always commit before running without `--dry-run`

---

## Linting Rules

### Enabled Rule Categories

| Category | Description           | Example                         |
|----------|-----------------------|---------------------------------|
| `E`      | pycodestyle errors    | Indentation, syntax             |
| `W`      | pycodestyle warnings  | Trailing whitespace             |
| `F`      | pyflakes              | Unused imports, undefined names |
| `I`      | isort                 | Import sorting                  |
| `N`      | pep8-naming           | Variable naming conventions     |
| `UP`     | pyupgrade             | Modern Python syntax            |
| `B`      | flake8-bugbear        | Likely bugs                     |
| `C4`     | flake8-comprehensions | List/dict comprehensions        |
| `SIM`    | flake8-simplify       | Code simplifications            |

### Ignored Rules

- `E501`: Line too long (handled by formatter)
- `B008`: Function calls in argument defaults (needed for dataclasses)
- `B904`: Exception chaining with `raise ... from`
- `SIM108`: Ternary operators (can reduce readability)

### Configuration

```toml
# pyproject.toml
[tool.ruff]
target-version = "py310"
line-length = 100
extend-exclude = ["models/timesFM", ".venvx"]

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "SIM"]
ignore = ["E501", "B008", "B904", "SIM108"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

---

## Workflow Integration

### Between Development Phases

At the end of each phase (e.g., after completing Week 1):

```bash
# 1. Run cleanup to fix bulk issues
./scripts/cleanup_codebase.sh

# 2. Review and commit auto-fixes
git add -A
git commit -m "chore: lint and format codebase (end of Phase X)"

# 3. Run quality gate
./scripts/quality_gate.sh

# 4. If passed, push
git push
```

### Pre-Commit (Recommended)

Add to your workflow:

```bash
# Before every commit
./scripts/quality_gate.sh --fast

# Or add to git pre-commit hook
echo './scripts/quality_gate.sh --fast' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### CI/CD Integration

Add to GitHub Actions / CI pipeline:

```yaml
# .github/workflows/quality.yml
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run quality gate
        run: ./scripts/quality_gate.sh
```

---

## Current State (Post-Week 1)

### Linting Statistics

As of Week 1 completion:

- **Total issues:** 2,495
- **Auto-fixable:** 1,860 (75%)
- **Manual review needed:** 635

**Top issues:**

1. Blank lines with whitespace (1,291)
2. Non-PEP585 type annotations (367)
3. Non-PEP604 Optional annotations (154)
4. Deprecated imports (130)
5. Unused imports (88)

**Critical issues:**

- Syntax errors: 12
- Unused variables: 23

### Recommended Action

Run cleanup at end of current enhancement phase (Week 3):

```bash
# After Week 3 completion
./scripts/cleanup_codebase.sh
./scripts/quality_gate.sh
```

This will clean up bulk issues before Week 4 (SAC implementation).

---

## Troubleshooting

### "ruff: command not found"

```bash
pip install ruff
```

### "Too many linting errors"

Start with auto-fixes:

```bash
./scripts/cleanup_codebase.sh
```

Then address remaining critical issues manually.

### "Tests failing"

Run in verbose mode to see details:

```bash
./scripts/run_tests.sh --verbose
```

Or run specific failing test:

```bash
python -m pytest tests/test_specific.py -v
```

### "Import errors in tests"

Ensure you're in the virtual environment:

```bash
source .venvx/bin/activate
python -m pytest tests/
```

---

## Best Practices

1. **Run quality gate before pushing**
   ```bash
   ./scripts/quality_gate.sh
   ```

2. **Fix linting issues incrementally**
    - Auto-fix safe issues: `./scripts/run_lint.sh --fix`
    - Review diffs before committing
    - Fix critical issues (syntax errors) manually

3. **Write tests for new features**
    - Add to `tests/test_*.py`
    - Run tests: `./scripts/run_tests.sh`
    - Ensure tests pass before committing

4. **Use type hints**
    - Modern syntax: `list[str]` not `List[str]`
    - Optional: `str | None` not `Optional[str]`
    - Ruff will suggest upgrades

5. **Keep imports organized**
    - Ruff auto-sorts with `--fix`
    - Groups: stdlib → third-party → first-party

---

## References

- [Ruff documentation](https://docs.astral.sh/ruff/)
- [pytest documentation](https://docs.pytest.org/)
- [PEP 8 style guide](https://peps.python.org/pep-0008/)
- [PEP 585 (type hints)](https://peps.python.org/pep-0585/)
- [PEP 604 (union types)](https://peps.python.org/pep-0604/)
