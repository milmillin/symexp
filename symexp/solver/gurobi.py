from typing import Any, cast, Sequence
import multiprocessing

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


def _load_gurobi():
    try:
        import gurobipy as gp
        from gurobipy import GRB  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name == "gurobipy":
            raise ModuleNotFoundError(
                'GurobiSolver requires the optional \'gurobi\' extra. '
                'Install it with `pip install "symexp[gurobi]"`.'
            ) from exc
        raise

    return gp, GRB


class GurobiSolver(Solver[_ExprT_con]):
    def __init__(
        self,
        model: Model[_ExprT_con],
        num_threads: int = multiprocessing.cpu_count(),
        non_convex: bool = False,
        time_limit: float = _INF,
        **kwargs,
    ):
        gp, grb = _load_gurobi()
        super().__init__(model)
        self._grb = grb
        self._vtype = {VType.BINARY: grb.BINARY, VType.INTEGER: grb.INTEGER, VType.CONTINUOUS: grb.CONTINUOUS}
        self._sense = {Sense.MAXIMIZE: grb.MAXIMIZE, Sense.MINIMIZE: grb.MINIMIZE}
        self.model = gp.Model(model.name(), **kwargs)  # type: ignore
        self.model.Params.Threads = num_threads
        self.model.Params.TimeLimit = time_limit
        if non_convex:
            self.model.Params.NonConvex = 2

        inner_vars = [self._add_var(f"V{v.index()}", v.type(), v.bound().min, v.bound().max) for v in model.get_vars()]
        self._var_names = [v.name() for v in model.get_vars()]
        self._var_name_map = {f"V{v.index()}": v.name() for v in model.get_vars()}
        self._var_name_imap = {v.name(): f"V{v.index()}" for v in model.get_vars()}
        self._constr_names = [c.name() for c in model.get_constraints()]
        self._constr_name_imap: dict[str, str] = {}
        for i, constr in enumerate(model.get_constraints()):
            assert i == constr.index()
            expr = constr.get_expr()
            op = constr.get_op()
            inner_expr = _to_inner_expr(expr, inner_vars)
            constr_name = f"C{i}"
            self._constr_name_imap[constr_name] = constr._name if constr._name is not None else constr_name
            match op:
                case RelOp.EQ:
                    self._add_constraint(inner_expr == 0, constr_name)
                case RelOp.LE:
                    self._add_constraint(inner_expr <= 0, constr_name)
                case RelOp.GE:
                    self._add_constraint(inner_expr >= 0, constr_name)
        obj, obj_sense = model.get_objective()
        inner_obj = _to_inner_expr(obj, inner_vars)
        self._set_objective(inner_obj, obj_sense)
        self.model.update()
        self._vars = self.model.getVars()

    def _add_var(self, name: str, vtype: VType, lb: float, ub: float) -> _Var:
        return self.model.addVar(vtype=self._vtype[vtype], name=name, lb=lb, ub=ub)

    def _add_constraint(self, constraint: Any, name: str) -> None:
        return self.model.addConstr(constraint, name)

    def _set_objective(self, objective: _Expr, sense: Sense) -> None:
        return self.model.setObjective(objective, self._sense[sense])

    def _solve(self):
        grb = self._grb
        self.model._self = self
        self.model._objbnd = float("-inf")
        self.model._sense = self._og_model.get_objective()[1].value
        self.model.optimize(_callback)
        status = self.model.getAttr(grb.Attr.Status)
        sol_count = self.model.getAttr(grb.Attr.SolCount)
        if status == grb.INF_OR_UNBD:
            self.model.Params.DualReductions = 0
            self._solve()
        if status == grb.INFEASIBLE:
            raise ModelInfeasibleError(self)
        elif status == grb.UNBOUNDED:
            raise ModelUnboundedError(self)
        elif status == grb.TIME_LIMIT and sol_count == 0:
            raise SolverTimeoutError(self, "No solution found within the time limit")
        elif sol_count == 0:
            raise SolverError(self, "No solution found")

    def _get_solution(self) -> Solution:
        return {self._var_names[int(v.VarName[1:])]: v.X for v in self.model.getVars()}

    def set_solutions(self, *solution: Solution) -> None:
        self.model.NumStart = len(solution)
        self.model.update()

        for i, s in enumerate(solution):
            s = {self._var_name_imap[k]: v for k, v in s.items()}
            self.model.params.StartNumber = i

            for v in self.model.getVars():
                if v.VarName in s:
                    v.Start = s[v.VarName]


def _callback(model, where):
    self = cast(GurobiSolver, model._self)
    grb = self._grb
    if where == grb.Callback.MIPSOL:
        xs = model.cbGetSolution(self._vars)
        runtime = model.cbGet(grb.Callback.RUNTIME)
        sol = {self._var_names[int(var.VarName[1:])]: x for var, x in zip(self._vars, xs)}
        obj = model.cbGet(grb.Callback.MIPSOL_OBJBST)
        bnd = model.cbGet(grb.Callback.MIPSOL_OBJBND)
        if obj == 1e100:
            obj = _INF
        self.solution_found.invoke(self, sol, SolverInfo(runtime, obj, bnd))
    elif where == grb.Callback.MESSAGE:
        runtime = model.cbGet(grb.Callback.RUNTIME)
        self.tick.invoke(self, runtime)
    elif where == grb.Callback.MIP:
        bnd = model.cbGet(grb.Callback.MIP_OBJBND) * model._sense
        if bnd > model._objbnd:
            model._objbnd = bnd
            runtime = model.cbGet(grb.Callback.RUNTIME)
            obj = model.cbGet(grb.Callback.MIP_OBJBST)
            if obj == 1e100:
                obj = _INF
            self.bound_found.invoke(self, SolverInfo(runtime, obj, bnd * model._sense))


# check if all abstract methods are implemented
if __name__ == "__main__":
    m = Model.create("")
    g = GurobiSolver(m)


# ------------
# Utils


def _to_inner_expr(expr: QuadExpr, inner_vars: Sequence[_Var]) -> _Expr:
    return sum(
        (inner_vars[v1.index()] * inner_vars[v2.index()] * coeff for (v1, v2), coeff in expr.quad_expr().items()), 0
    ) + sum((inner_vars[var.index()] * coeff for var, coeff in expr.lin_expr().items()), expr.const_expr())
