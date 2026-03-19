from typing import Any, cast, Sequence

from ..expr import Model, QuadExpr, Sense, Solution, VType
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


def _load_scip():
    try:
        from pyscipopt import Model as ScipModel, SCIP_EVENTTYPE
    except ModuleNotFoundError as exc:
        if exc.name == "pyscipopt":
            raise ModuleNotFoundError(
                'ScipSolver requires the optional \'scip\' extra. '
                'Install it with `pip install "symexp[scip]"`.'
            ) from exc
        raise

    return ScipModel, SCIP_EVENTTYPE


class ScipSolver(Solver[_ExprT_con]):
    def __init__(
        self,
        model: Model[_ExprT_con],
        time_limit: float = _INF,
        **params,
    ):
        ScipModel, SCIP_EVENTTYPE = _load_scip()
        super().__init__(model)

        self._solver = ScipModel(model.name())
        self._event_type = SCIP_EVENTTYPE
        self._solution_emitted = False
        if not any(name.startswith("display/") for name in params):
            self._solver.hideOutput()
        if time_limit != _INF:
            self._solver.setParam("limits/time", time_limit)
        for name, value in params.items():
            self._solver.setParam(name, value)

        self._vtype = {
            VType.CONTINUOUS: "C",
            VType.INTEGER: "I",
            VType.BINARY: "B",
        }
        self._vars = [self._add_var(var.name(), var.type(), var.bound().min, var.bound().max) for var in model.get_vars()]
        self._var_names = [var.name() for var in model.get_vars()]
        self._var_name_to_var = dict(zip(self._var_names, self._vars))

        for constr in model.get_constraints():
            expr = constr.get_expr()
            inner_expr = _to_inner_expr(expr, self._vars)
            self._solver.addCons(_to_constraint(inner_expr, constr.get_op()), name=constr._name or "")

        obj, obj_sense = model.get_objective()
        inner_obj = _to_inner_expr(obj, self._vars)
        if obj.quad_expr():
            aux = self._solver.addVar(name="__symexp_obj__", lb=None, ub=None)
            if obj_sense == Sense.MINIMIZE:
                self._solver.addCons(aux >= inner_obj, name="__symexp_obj_ge")
                self._solver.setObjective(aux, "minimize")
            elif obj_sense == Sense.MAXIMIZE:
                self._solver.addCons(aux <= inner_obj, name="__symexp_obj_le")
                self._solver.setObjective(aux, "maximize")
            else:
                raise ValueError(f"invalid sense: {obj_sense}")
        elif obj_sense == Sense.MINIMIZE:
            self._solver.setObjective(inner_obj, "minimize")
        elif obj_sense == Sense.MAXIMIZE:
            self._solver.setObjective(inner_obj, "maximize")
        else:
            raise ValueError(f"invalid sense: {obj_sense}")

        self._solver.attachEventHandlerCallback(_on_best_solution, [SCIP_EVENTTYPE.BESTSOLFOUND])
        self._solver.data = self

    def _add_var(self, name: str, vtype: VType, lb: float, ub: float) -> _Var:
        return self._solver.addVar(
            name=name,
            vtype=self._vtype[vtype],
            lb=None if lb == float("-inf") else lb,
            ub=None if ub == float("inf") else ub,
        )

    def _emit_solution(self) -> None:
        sol = self._solver.getBestSol()
        if sol is None:
            return
        info = SolverInfo(
            self._solver.getSolvingTime(),
            self._solver.getSolObjVal(sol),
            self._solver.getDualbound(),
        )
        solution = {name: self._solver.getSolVal(sol, var) for name, var in zip(self._var_names, self._vars)}
        self.solution_found.invoke(self, solution, info)
        self._solution_emitted = True

    def _solve(self):
        self._solver.optimize()
        status = self._solver.getStatus()

        if status == "optimal":
            if not self._solution_emitted:
                self._emit_solution()
            return
        if status == "infeasible":
            raise ModelInfeasibleError(self)
        if status == "unbounded":
            raise ModelUnboundedError(self)
        if status == "timelimit":
            if self._solver.getBestSol() is not None:
                if not self._solution_emitted:
                    self._emit_solution()
                return
            raise SolverTimeoutError(self, status)
        if self._solver.getBestSol() is not None:
            if not self._solution_emitted:
                self._emit_solution()
            return
        if status == "inforunbd":
            raise SolverError(self, "Model is infeasible or unbounded")
        raise SolverError(self, status)

    def _get_solution(self) -> Solution:
        sol = self._solver.getBestSol()
        if sol is None:
            raise SolverError(self, "SCIP did not return a feasible solution")
        return {name: self._solver.getSolVal(sol, var) for name, var in zip(self._var_names, self._vars)}

    def set_solutions(self, *solution: Solution) -> None:
        for start in solution:
            sol = self._solver.createSol()
            for name, value in start.items():
                if name not in self._var_name_to_var:
                    raise KeyError(f"Unknown variable {name!r}")
                self._solver.setSolVal(sol, self._var_name_to_var[name], float(value))
            if not self._solver.addSol(sol):
                raise SolverError(self, "SCIP rejected the provided warm start")


def _on_best_solution(model, event):
    self = cast(ScipSolver, model.data)
    self._emit_solution()


def _to_constraint(expr: _Expr, op) -> Any:
    if op.value == "==":
        return expr == 0
    if op.value == "<=":
        return expr <= 0
    if op.value == ">=":
        return expr >= 0
    raise ValueError(f"invalid operator: {op}")


def _to_inner_expr(expr: QuadExpr, inner_vars: Sequence[_Var]) -> _Expr:
    return sum(
        (inner_vars[v1.index()] * inner_vars[v2.index()] * coeff for (v1, v2), coeff in expr.quad_expr().items()), 0
    ) + sum((inner_vars[var.index()] * coeff for var, coeff in expr.lin_expr().items()), expr.const_expr())
