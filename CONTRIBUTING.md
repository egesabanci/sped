# Contributing to sped

## Development Setup

```bash
# Clone
git clone https://github.com/egesabanci/sped
cd sped

# Create environment with uv
uv venv
source .venv/bin/activate

# Install in editable mode with dev deps
uv pip install -e ".[dev]"
```

## Code Quality

We use `ruff` for linting and formatting:

```bash
# Format code
uv run ruff format sped/

# Lint
uv run ruff check sped/

# Type check (optional)
uv run mypy sped/ --ignore-missing-imports
```

## Testing

```bash
# Run all fast tests
uv run pytest tests/ -v -m "not slow"

# Run slow tests (model loading, full training)
uv run pytest tests/ -v -m "slow"

# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=sped --cov-report=term-missing
```

### Test Philosophy

- **Fast tests** run in < 1s, no model loading, no network access
- **Slow tests** load tiny HF models and are marked `@pytest.mark.slow`
- **Network tests** are avoided where possible; use mock/synthetic data
- All tests must pass on CPU (no CUDA required)

## Branch Strategy

```
Branch format: <type>/<kebab-case-description>
```

| Type | Purpose |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `chore` | Maintenance |
| `docs` | Documentation |
| `test` | Test improvements |
| `ci` | CI/CD changes |

### Pipeline

```
main ──→ feat/xxx ──→ implement ──→ write tests ──→ all tests pass ──→ commit ──→ PR ──→ squash merge to main
```

## Commit Messages

Use conventional commits:

```
<type>: <short description>

<optional body with details>
```

Examples:
```
feat(core): implement speculate-verify-accept loop
fix(vocab): handle empty draft tokens in string alignment
docs: add CLI reference for serve command
```

## Pull Requests

1. Create a branch from `main`
2. Implement your changes
3. Write/update tests
4. Run full test suite
5. Push and create a PR
6. Squash merge when approved

## Project Structure

```
sped/
├── sped/
│   ├── cli/               — CLI commands (Typer)
│   ├── core/              — Speculative decoding engine
│   ├── vocab_agnostic/    — Cross-vocabulary alignment
│   ├── distillation/      — PEFT distillation
│   ├── adaptation/        — Online adaptation
│   ├── serving/           — Inference backends
│   └── utils/             — Shared utilities
├── tests/                 — pytest test suite
├── docs/                  — Documentation
└── pyproject.toml         — Project config
```

## Release Process

1. Update `__version__` in `sped/__init__.py`
2. Update `CHANGELOG.md`
3. Push and tag: `git tag v0.x.x && git push --tags`
4. GitHub Release triggers PyPI publish via CI
