import builtins
import importlib
import re
import sys

import pytest

from symexp import Model, Sense


def _block_imports(monkeypatch: pytest.MonkeyPatch, *prefixes: str) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _reload_solver_module():
    for name in ("symexp.solver", "symexp.solver.gurobi", "symexp.solver.highs", "symexp.solver.scipy_lp", "symexp.solver.cuopt"):
        sys.modules.pop(name, None)
    return importlib.import_module("symexp.solver")


def _make_model():
    model = Model.create("test")
    x = model.add_var(name="x")
    model.set_objective(Sense.MINIMIZE, x)
    return model


def test_solver_package_exports_all_solver_classes_without_optional_backends(monkeypatch: pytest.MonkeyPatch):
    _block_imports(monkeypatch, "gurobipy", "highspy", "numpy", "scipy", "cuopt")

    solver_module = _reload_solver_module()

    assert solver_module.GurobiSolver.__name__ == "GurobiSolver"
    assert solver_module.HighsSolver.__name__ == "HighsSolver"
    assert solver_module.ScipyLpSolver.__name__ == "ScipyLpSolver"
    assert solver_module.CuOptSolver.__name__ == "CuOptSolver"

    namespace = {}
    exec("from symexp.solver import *", {}, namespace)
    assert namespace["GurobiSolver"] is solver_module.GurobiSolver
    assert namespace["HighsSolver"] is solver_module.HighsSolver
    assert namespace["ScipyLpSolver"] is solver_module.ScipyLpSolver
    assert namespace["CuOptSolver"] is solver_module.CuOptSolver


@pytest.mark.parametrize(
    ("solver_name", "blocked_imports", "install_hint"),
    [
        ("GurobiSolver", ("gurobipy",), 'symexp[gurobi]'),
        ("HighsSolver", ("highspy",), 'symexp[highs]'),
        ("ScipyLpSolver", ("numpy",), 'symexp[scipy]'),
        ("CuOptSolver", ("cuopt",), "cuopt-cu12==26.2.*"),
    ],
)
def test_optional_solver_backend_errors_are_deferred(
    monkeypatch: pytest.MonkeyPatch,
    solver_name: str,
    blocked_imports: tuple[str, ...],
    install_hint: str,
):
    _block_imports(monkeypatch, *blocked_imports)

    solver_module = _reload_solver_module()
    solver_cls = getattr(solver_module, solver_name)

    with pytest.raises(ModuleNotFoundError, match=re.escape(install_hint)):
        solver_cls(_make_model())
