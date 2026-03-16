# symexp

Typed symbolic expressions for optimization models.

## Installation

The project now uses a standard `pyproject.toml` build, so you can install it
directly with `pip`.

```bash
pip install .
```

Optional extras:

```bash
pip install ".[dev]"
pip install ".[scipy]"
pip install ".[gurobi]"
```

- `dev` installs `black`, `pre-commit`, `pytest`, and `build`
- `scipy` enables `ScipyLpSolver`
- `gurobi` enables `GurobiSolver` and still requires a valid local Gurobi setup

## Development

If you use the included Conda environment:

```bash
conda env create -f environment.yml
conda activate symexp
```

Install the git hooks after setting up the environment:

```bash
pre-commit install
```

Configured hooks include:

- `black --line-length=120`
- trailing whitespace cleanup
- end-of-file normalization
- YAML, TOML, and JSON validation
- merge-conflict and large-file checks
