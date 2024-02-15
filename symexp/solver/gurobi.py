from typing import Any, cast, Sequence
import multiprocessing

import gurobipy as gp
from gurobipy import GRB  # type: ignore

from ..expr import Model, RelOp, Sense, Solution, VType, QuadExpr, LinExpr
from ._base import Solver, _ExprT_con, ModelInfeasibleError, ModelUnboundedError

_VTYPE = {VType.BINARY: GRB.BINARY, VType.INTEGER: GRB.INTEGER, VType.CONTINUOUS: GRB.CONTINUOUS}
_SENSE = {Sense.MAXIMIZE: GRB.MAXIMIZE, Sense.MINIMIZE: GRB.MINIMIZE}

_Var = Any
_Expr = Any


class GurobiSolver(Solver[_ExprT_con]):
    def __init__(
        self,
        model: Model[_ExprT_con],
        num_threads: int = multiprocessing.cpu_count(),
        non_convex: bool = False,
        time_limit: float = float("inf"),
        **kwargs,
    ):
        super().__init__(model)
        self.model = gp.Model(model.name(), **kwargs)  # type: ignore
        self.model.Params.Threads = num_threads
        self.model.Params.TimeLimit = time_limit
        if non_convex:
            self.model.Params.NonConvex = 2

        inner_vars = [self._add_var(f"V{v.index()}", v.type(), v.bound().min, v.bound().max) for v in model.get_vars()]
        self._var_names = [v.name() for v in model.get_vars()]
        self._var_name_map = {f"V{v.index()}": v.name() for v in model.get_vars()}
        self._constr_names = [c.name() for c in model.get_constraints()]
        for i, constr in enumerate(model.get_constraints()):
            assert i == constr.index()
            expr = constr.get_expr()
            op = constr.get_op()
            inner_expr = _to_inner_expr(expr, inner_vars)
            constr_name = f"C{i}"
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
        return self.model.addVar(vtype=_VTYPE[vtype], name=name, lb=lb, ub=ub)

    def _add_constraint(self, constraint: Any, name: str) -> None:
        return self.model.addConstr(constraint, name)

    def _set_objective(self, objective: _Expr, sense: Sense) -> None:
        return self.model.setObjective(objective, _SENSE[sense])

    def _solve(self):
        self.model._self = self
        self.model.optimize(_callback)
        status = self.model.getAttr(GRB.Attr.Status)
        sol_count = self.model.getAttr(GRB.Attr.SolCount)
        if status == GRB.INF_OR_UNBD:
            self.model.Params.DualReductions = 0
            self._solve()
        if status == GRB.INFEASIBLE:
            raise ModelInfeasibleError(self)
        elif status == GRB.UNBOUNDED:
            raise ModelUnboundedError(self)
        elif status == GRB.TIME_LIMIT and sol_count == 0:
            raise TimeoutError("No solution found within the time limit")
        elif sol_count == 0:
            raise ValueError("No solution found")

    def _get_solution(self) -> Solution:
        res = {}
        for v in self.model.getVars():
            res[v.VarName] = v.X
        return res

    def set_solutions(self, *solution: Solution) -> None:
        self.model.NumStart = len(solution)
        self.model.update()

        for i, s in enumerate(solution):
            s = {self._var_name_map[k]: v for k, v in s.items()}
            self.model.params.StartNumber = i

            for v in self.model.getVars():
                if v.VarName in s:
                    v.Start = s[v.VarName]


def _callback(model, where):
    if where == GRB.Callback.MIPSOL:
        self = cast(GurobiSolver, model._self)
        xs = model.cbGetSolution(self._vars)
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        sol = {self._var_names[int(var.VarName[1:])]: x for var, x in zip(self._vars, xs)}
        self.solution_found.invoke(self, sol, runtime)
    elif where == GRB.Callback.MESSAGE:
        self = cast(GurobiSolver, model._self)
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        self._invoke_tick(runtime)


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
