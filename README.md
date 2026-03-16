# symexp

`symexp` is a typed symbolic modeling library for optimization problems. It
lets you build linear or quadratic expressions with normal Python operators,
attach them to a model, validate candidate assignments, and solve supported
models with SciPy or Gurobi.

## Installation

Install the base package:

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
- `gurobi` enables `GurobiSolver` and still requires a valid local Gurobi
  installation and license

If you use the included Conda environment:

```bash
conda env create -f environment.yml
conda activate symexp
```

## Quickstart

Build a linear model, solve it with SciPy, then evaluate expressions against
the returned solution:

```python
from symexp import Model, Sense, evaluate
from symexp.solver import ScipyLpSolver

m = Model.create("lp-demo")
x = m.add_var(name="x", lb=0)
y = m.add_var(name="y", lb=0)

m.add_constraint(x + y >= 4, name="demand")
m.add_constraint(x - y <= 1, name="balance")
m.set_objective(Sense.MINIMIZE, 2 * x + y)

solver = ScipyLpSolver(m)
solution = solver.solve()
m.set_solution(solution)

print(solution)
print(evaluate(x, y, 2 * x + y))
```

For logic-heavy models you can validate assignments without solving:

```python
from symexp import Model, VType

m = Model.create("logic")
x = m.add_vars(3, VType.BINARY, name="x")
y = m.and_(*x, name="y")

candidate = {"x[0]": 1, "x[1]": 1, "x[2]": 1, "y": 1}
assert m.check_solution_satisfies_constraints(candidate)
```

## Core Concepts

- `VType`: variable type enum with `BINARY`, `INTEGER`, and `CONTINUOUS`
- `Sense`: objective sense enum with `MINIMIZE` and `MAXIMIZE`
- `RelOp`: relation enum used by constraints
- `Var` / `BinVar`: symbolic decision variables
- `LinExpr` / `QuadExpr`: linear and quadratic symbolic expressions
- `Constr`: symbolic constraint created by `==`, `<=`, and `>=`
- `Bound`: lower and upper bound pair returned by `.bound()`

Use `Model.create(name)` for a linear model and `Model.create(name, QuadExpr)`
for a quadratic-capable model.

## Modeling API

### Model lifecycle

| API | Purpose |
| --- | --- |
| `Model.create(name, type=LinExpr, M=1e5, create_evalf=False)` | Creates a model. `type=QuadExpr` enables quadratic constraints/objectives. `M` controls big-M helper formulations. `create_evalf=True` wires helper variables with evaluation callbacks. |
| `add_var(vtype=VType.CONTINUOUS, ub=inf, lb=0, name=None, evalf=None)` | Adds one variable. Integer variables with bounds `[0, 1]` become binary variables automatically. `evalf` is an optional callback for derived variable evaluation. |
| `add_vars(n, vtype=..., ub=..., lb=..., name=None)` | Adds a list of variables with indexed names like `x[0]`, `x[1]`, ... |
| `add_constraint(constr, name=None, if_=None)` | Adds a symbolic constraint. `if_=` conditionally enforces the constraint with a binary expression. |
| `pop_constraints(num=1)` | Removes the most recently added constraints. Useful for temporary model edits. |
| `set_objective(sense, expr)` | Sets the objective expression. |
| `set_solution(solution)` | Assigns variable values from a `dict[str, float]`. |
| `check_solution_satisfies_constraints(solution=None, verbose=False, throw=False)` | Checks whether the current or provided solution satisfies all constraints. |
| `enforce_solution(solution)` | Adds equality constraints that pin variables to a given solution. |
| `get_vars()`, `get_constraints()`, `get_objective()` | Returns model contents for inspection or custom solver integrations. |
| `feasible()` | Returns whether the model has already been marked infeasible by an always-false constraint. |
| `sum(xs)` | Typed helper for summing numeric values and symbolic expressions. |

### Expressions and constraints

- Expressions use normal Python arithmetic:
  `+`, `-`, unary `-`, `*`, `/`
- Comparisons create constraints:
  `expr1 == expr2`, `expr1 <= expr2`, `expr1 >= expr2`
- Expressions can be printed directly, bounded with `.bound()`, and evaluated
  once variables have values
- Binary expressions support `~x` for logical negation

Example:

```python
from symexp import Model, QuadExpr, Sense

m = Model.create("quadratic", QuadExpr)
x = m.add_var(name="x")
y = m.add_var(name="y")

m.add_constraint(x + y >= 1)
m.set_objective(Sense.MINIMIZE, x * y + x + 2 * y)
```

## Helper Constructors

The `Model` object includes higher-level helpers for common mixed-integer
patterns:

| Helper | Purpose |
| --- | --- |
| `and_(*x, name=None)` | Creates a binary expression for logical AND. |
| `or_(*x, name=None)` | Creates a binary expression for logical OR. |
| `min_(*x, name=None)` | Creates a value variable and argmin selector binaries. |
| `max_(*x, name=None)` | Creates a value variable and argmax selector binaries. |
| `abs_(x, name=None)` | Creates a variable representing `abs(x)`. |
| `mux_((cond, expr), ..., name=None, else_=None)` | Selects one expression from multiple branches. |
| `bmux_((cond, expr), ..., name=None, else_=None)` | Binary-only version of `mux_`. |
| `muxT_(conds, exprs, name=None)` | Sequence-based variant of `mux_`. |
| `bmuxT_(conds, exprs, name=None)` | Sequence-based variant of `bmux_`. |
| `min_en_((enable, expr), ..., name=None)` | Minimum over the enabled expressions; returns `M` if none are enabled. |
| `min_enT_(enables, exprs, name=None)` | Sequence-based variant of `min_en_`. |

`min_`, `max_`, and `min_en_` return an object with:

- `.value`: the created expression/variable
- `.index`: the selector binary expressions

Example:

```python
from symexp import Model, VType

m = Model.create("helpers")
a = m.add_var(name="a")
b = m.add_var(name="b")
s = m.add_vars(2, VType.BINARY, name="s")

best = m.max_(a, b, name="best")
choice = m.mux_((s[0], a), (s[1], b), else_=0, name="choice")

m.add_constraint(best.value >= choice)
```

## Evaluation Utilities

`symexp` can evaluate expressions directly from assigned variable values:

- `evaluate(x)` returns the numeric value of one expression or variable
- `evaluate(x, y, expr)` returns a tuple of values
- `evaluate(list_or_tuple_or_dict)` recursively evaluates containers
- `evaluate(pydantic_model)` returns a new model with fields evaluated
- `evaluate_constr(constr)` returns whether a constraint is currently satisfied

If `Model.create(..., create_evalf=True)` is enabled, helper-created variables
such as `max_().value`, selector binaries, `and_()`, `or_()`, `mux_()`, and
`min_en_()` can often be evaluated from the base variable assignments alone.

```python
from symexp import Model, evaluate

m = Model.create("eval", create_evalf=True)
a = m.add_var(name="a")
b = m.add_var(name="b")
best = m.max_(a, b, name="best")

a.set_value(2)
b.set_value(5)

print(evaluate(best.value))      # 5
print(evaluate(best.index[0]))   # 0
print(evaluate(best.index[1]))   # 1
```

## Solvers

All solvers expose:

- `solve() -> Solution`
- `set_solutions(*solutions)` for warm starts when the backend supports them
- `solution_found`, `tick`, and `bound_found` callback events

The callback payload uses `SolverInfo(runtime, best_obj, obj_bound)`.

### `ScipyLpSolver`

Import with:

```python
from symexp.solver import ScipyLpSolver
```

Use it for linear programs with continuous variables only.

- accepts `Model[LinExpr]`
- rejects integer or binary variables
- does not implement warm starts
- raises solver errors for infeasible, unbounded, timeout, or numerical issues

### `GurobiSolver`

Import with:

```python
from symexp.solver import GurobiSolver
```

Use it for mixed-integer or quadratic models when Gurobi is available.

- supports continuous, integer, and binary variables
- supports quadratic models
- accepts `num_threads`, `non_convex`, and `time_limit`
- supports warm starts through `set_solutions(*solutions)`
- emits incumbent solutions, runtime ticks, and bound updates through events

Example callback registration:

```python
from symexp.solver import GurobiSolver

solver = GurobiSolver(m, time_limit=60)
solver.solution_found.register(lambda solver, sol, info: print(info.runtime, info.best_obj))
solution = solver.solve()
```

### Solver errors

The solver module exports:

- `SolverError`: base class for solver failures
- `ModelInfeasibleError`: model is infeasible
- `ModelUnboundedError`: model is unbounded
- `SolverTimeoutError`: solver hit its time limit

## Development

Install git hooks after setting up the environment:

```bash
pre-commit install
pre-commit run --all-files
```

Configured hooks include:

- `black --line-length=120`
- trailing whitespace cleanup
- end-of-file normalization
- YAML, TOML, and JSON validation
- merge-conflict and large-file checks

Run tests with:

```bash
pytest -q
```

The Gurobi-backed test module is skipped automatically when Gurobi is not
installed or the local license is unusable.
