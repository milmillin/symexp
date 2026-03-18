from importlib import import_module

from ._base import Solver, SolverError, SolverTimeoutError, ModelInfeasibleError, ModelUnboundedError, SolverInfo

__all__ = [
    "Solver",
    "SolverError",
    "SolverTimeoutError",
    "ModelInfeasibleError",
    "ModelUnboundedError",
    "SolverInfo",
    "GurobiSolver",
    "ScipyLpSolver",
    "CuOptSolver",
]

_OPTIONAL_SOLVERS = {
    "GurobiSolver": (".gurobi", {"gurobipy"}, "gurobi"),
    "ScipyLpSolver": (".scipy_lp", {"numpy", "scipy"}, "scipy"),
    "CuOptSolver": (".cuopt", {"cuopt"}, "cuopt"),
}


def __getattr__(name: str):
    if name not in _OPTIONAL_SOLVERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, import_names, extra_name = _OPTIONAL_SOLVERS[name]
    try:
        module = import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        if exc.name in import_names:
            raise ModuleNotFoundError(
                f"{name} requires the optional '{extra_name}' extra. Install it with `pip install \"symexp[{extra_name}]\"`."
            ) from exc
        raise

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
