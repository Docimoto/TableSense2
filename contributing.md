# Contributing to TableSense2

Thanks for your interest in contributing to **TableSense2**!
This project aims to advance open, reproducible spreadsheet understanding research while remaining practical for real-world use.


---

## Code of Conduct

Please be respectful and professional in all interactions. We follow a simple rule:
**be constructive, be kind, and assume good intent.**

---

## Ways to Contribute

You can help in many ways:

- **Bug reports** (repro steps, logs, minimal example files)
- **Documentation improvements**
- **New features** (models, training options, evaluators, visualization)
- **Performance improvements** (speed, memory, batching)
- **Dataset tooling** (cleaning scripts, schema validation, converters)
- **Tests** (unit tests, integration tests, regression tests)
- **Examples** (notebooks, sample configs, demos)

---

## Development Setup

TableSense2 supports Conda and venv setups. See the docs for complete instructions:

- Setup: [docs/setup.md](docs/setup.md)
- Experiments: [docs/experiment_tracking.md](docs/experiment_tracking.md)
- Cloud training: [vertex-ai/README.md](vertex-ai/README.md)
- Architecture: [docs/architecture.md](docs/architecture.md)

Quick start (Conda recommended):

```
conda env create -f environment.yml
conda activate tablesense2
pip install -e .
pytest -q
```

---

## Repository Conventions

### Branching

Create a feature branch from `main`:

- `feature/<short-description>`
- `bugfix/<short-description>`
- `docs/<short-description>`
- `perf/<short-description>`
- `refactor/<short-description>`

---

### Style & Formatting

TableSense2 does not currently enforce automatic linting or formatting.
However, the project aims to maintain a clean and readable codebase.

Please follow these general guidelines when contributing:
- use clear, descriptive naming
- keep functions and classes reasonably scoped
- add docstrings for public functions and classes
- prefer explicit, readable code over clever constructs
- include type hints where they improve clarity (not required everywhere)

#### Planned tooling

In future releases, we plan to introduce:
- **Ruff** for linting and import sorting
- **Ruff formatter** (or an equivalent) for consistent code formatting
- optional pre-commit hooks and CI enforcement

When these tools are added, contributors will be encouraged to run them locally,
and formatting-related feedback will be automated to reduce review friction.

---

### Tests

Please add or update tests for changes that affect behavior.

Run tests:

```
pytest tests/ -v
```

If your change is difficult to test automatically (for example performance work), include:
- a short benchmark script or notes
- before / after numbers if possible

---

## Pull Request Guidelines

A good pull request includes:

1. **Clear description**
   - What changed?
   - Why?
   - Any tradeoffs?

2. **Verification steps**
   - Commands to run
   - Expected outputs
   - Config used (if applicable)

3. **Tests**
   - New tests added or existing tests updated
   - Confirmation that `pytest` passes

4. **Documentation**
   - Update README or docs if user-facing behavior changed

### PR Title Suggestions

- `Fix: ...`
- `Feat: ...`
- `Docs: ...`
- `Refactor: ...`
- `Perf: ...`
- `Test: ...`

---

## Dataset Contributions

TableSense2 plans to support multiple datasets and formats, from different domains. For dataset-related contributions:

- Do **not** submit proprietary, restricted, or confidential data
- If contributing a public dataset integration:
  - include the source and license
  - provide a downloader or preparation script
  - keep large files out of git history

If you are contributing dataset tooling (recommended), please include:
- schema documentation
- validation checks (bounds, missing labels, malformed workbooks)

See [preprocess_tablesense_files.md](docs/preprocess_tablesense_files.md)

---

## Reporting Issues

When filing a bug, please include:

- OS and Python version
- environment type (Conda / venv / Docker)
- PyTorch version (and CUDA version if applicable)
- minimal reproduction steps
- logs or stack traces
- sample input files **only if you are permitted to share them**

---

## Security

If you discover a security issue (for example a dependency vulnerability or unsafe file handling), please report it privately (github@docimoto.com) rather than opening a public issue.


---

## Licensing (Apache-2.0)

This project is licensed under the **Apache License, Version 2.0**.

By contributing to this repository, you agree that your contributions will be licensed under Apache-2.0 as part of the project.

If stronger provenance guarantees are required later (common in enterprise OSS), a DCO ("Signed-off-by") workflow can be added. For now, a standard GitHub pull request is sufficient.

---

Thank you again for contributing — your help improves TableSense2 for everyone.

