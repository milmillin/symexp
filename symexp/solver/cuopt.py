from typing import Any, Sequence

from ..expr import Model, RelOp, Sense, Solution, VType, QuadExpr, LinExpr
from ._base import (
    Solver,
    _ExprT_con,
    SolverError,
    ModelInfeasibleError,
    ModelUnboundedError,
    SolverTimeoutError,
    SolverInfo,
)

_Var = Any
_Expr = Any

_INF = float("inf")


def _load_cuopt():
    try:
        from cuopt.linear_programming import Problem, SolverSettings  # type: ignore
        from cuopt.linear_programming.internals import GetSolutionCallback  # type: ignore
        from cuopt.linear_programming.problem import VType as CuVType, sense as CuSense  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name == "cuopt" or (exc.name is not None and exc.name.startswith("cuopt.")):
            raise ModuleNotFoundError(
                "CuOptSolver requires cuOpt to be installed separately. "
                "Install it with "
                "`pip install --extra-index-url=https://pypi.nvidia.com 'cuopt-cu12==26.2.*'`."
            ) from exc
        raise

    return Problem, SolverSettings, GetSolutionCallback, CuVType, CuSense


class CuOptSolver(Solver[_ExprT_con]):
    def __init__(
        self,
        model: Model[_ExprT_con],
        time_limit: float = _INF,
        **kwargs,
    ):
        Problem, SolverSettings, GetSolutionCallback, CuVType, CuSense = _load_cuopt()
        super().__init__(model)
        self._solver_settings_cls = SolverSettings
        self._solution_callback_base = GetSolutionCallback
        self._vtype = {VType.CONTINUOUS: CuVType.CONTINUOUS, VType.INTEGER: CuVType.INTEGER, VType.BINARY: CuVType.INTEGER}
        self._sense = {Sense.MAXIMIZE: CuSense.MAXIMIZE, Sense.MINIMIZE: CuSense.MINIMIZE}
        self._problem = Problem(**kwargs)
        self._time_limit = time_limit

        inner_vars: list[_Var] = []
        for v in model.get_vars():
            lb = v.bound().min
            ub = v.bound().max
            vtype = v.type()
            # cuOpt has no native binary type; use INTEGER with [0, 1] bounds
            if vtype == VType.BINARY:
                lb, ub = 0, 1
            inner_vars.append(
                self._problem.addVariable(lb=lb, ub=ub, vtype=self._vtype[vtype], name=f"V{v.index()}")
            )

        self._var_names = [v.name() for v in model.get_vars()]
        self._inner_vars = inner_vars

        for i, constr in enumerate(model.get_constraints()):
            assert i == constr.index()
            expr = constr.get_expr()
            op = constr.get_op()
            inner_expr = _to_inner_expr(expr, inner_vars)
            match op:
                case RelOp.EQ:
                    self._problem.addConstraint(inner_expr == 0, name=f"C{i}")
                case RelOp.LE:
                    self._problem.addConstraint(inner_expr <= 0, name=f"C{i}")
                case RelOp.GE:
                    self._problem.addConstraint(inner_expr >= 0, name=f"C{i}")

        obj, obj_sense = model.get_objective()
        inner_obj = _to_inner_expr(obj, inner_vars)
        self._problem.setObjective(inner_obj, self._sense[obj_sense])

    def _solve(self):
        settings = self._solver_settings_cls()
        if self._time_limit != _INF:
            settings.set_parameter("time_limit", str(self._time_limit))

        # Set up MILP callback for solution_found events
        if self._problem.IsMIP:
            solver_self = self
            solution_callback_base = self._solution_callback_base

            class _SolCallback(solution_callback_base):
                def get_solution(self, solution, solution_cost, solution_bound, user_data):
                    sol = {
                        solver_self._var_names[i]: float(solution[i])
                        for i in range(len(solver_self._var_names))
                    }
                    obj = float(solution_cost[0]) if hasattr(solution_cost, "__getitem__") else float(solution_cost)
                    bnd = float(solution_bound[0]) if hasattr(solution_bound, "__getitem__") else float(solution_bound)
                    info = SolverInfo(0.0, obj, bnd)
                    solver_self.solution_found.invoke(solver_self, sol, info)

            settings.set_mip_callback(_SolCallback(), None)

        self._problem.solve(settings)
        status = self._problem.Status

        if status.name in ("PrimalInfeasible", "Infeasible"):
            raise ModelInfeasibleError(self)
        elif status.name in ("DualInfeasible", "Unbounded"):
            raise ModelUnboundedError(self)
        elif status.name == "TimeLimit":
            if self._problem.ObjValue is None:
                raise SolverTimeoutError(self, "No solution found within the time limit")
        elif status.name not in ("Optimal", "PrimalFeasible", "FeasibleFound"):
            raise SolverError(self, f"Solver returned status: {status.name}")

    def _get_solution(self) -> Solution:
        return {
            self._var_names[i]: float(self._inner_vars[i].getValue())
            for i in range(len(self._var_names))
        }

    def set_solutions(self, *solution: Solution) -> None:
        raise NotImplementedError(
            "cuOpt warm starts use a different mechanism (PDLP, LP-only) "
            "incompatible with the Solution type."
        )


# check if all abstract methods are implemented
if __name__ == "__main__":
    m = Model.create("")
    g = CuOptSolver(m)


# ------------
# Utils


def _to_inner_expr(expr: QuadExpr, inner_vars: Sequence[_Var]) -> _Expr:
    return sum(
        (inner_vars[v1.index()] * inner_vars[v2.index()] * coeff for (v1, v2), coeff in expr.quad_expr().items()), 0
    ) + sum((inner_vars[var.index()] * coeff for var, coeff in expr.lin_expr().items()), expr.const_expr())
