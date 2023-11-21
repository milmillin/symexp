from typing import Any, cast
import multiprocessing

import gurobipy as gp
from gurobipy import GRB  # type: ignore

from ..expr import Model, Sense, Solution, VType, QuadExpr, LinExpr
from ._base import Solver, _Var, _Expr, _ExprT_con, ModelInfeasibleError

_VTYPE = {VType.BINARY: GRB.BINARY, VType.INTEGER: GRB.INTEGER, VType.CONTINUOUS: GRB.CONTINUOUS}
_SENSE = {Sense.MAXIMIZE: GRB.MAXIMIZE, Sense.MINIMIZE: GRB.MINIMIZE}


class GurobiSolver(Solver[_ExprT_con]):
    def __init__(
        self,
        model: Model[_ExprT_con],
        num_threads: int = multiprocessing.cpu_count(),
        non_convex: bool = False,
        time_limit: float = float('inf'),
        **kwargs
    ):
        self.model = gp.Model(model.name(), **kwargs)  # type: ignore
        self.model.Params.Threads = num_threads
        self.model.Params.TimeLimit = time_limit
        if non_convex:
            self.model.Params.NonConvex = 2
        super().__init__(model)
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
        if status == GRB.INFEASIBLE or status == GRB.INF_OR_UNBD or status == GRB.UNBOUNDED:
            raise ModelInfeasibleError("Model is infeasible or unbounded")
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
            s = self._transform_solution(s)
            self.model.params.StartNumber = i

            for v in self.model.getVars():
                if v.VarName in s:
                    v.Start = s[v.VarName]


def _callback(model, where):
    self = cast(GurobiSolver, model._self)
    if where == GRB.Callback.MIPSOL:
        xs = model.cbGetSolution(self._vars)
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        sol = {var.VarName: x for var, x in zip(self._vars, xs)}
        self._invoke_solution_found(sol, runtime)


# check if all abstract methods are implemented
if __name__ == "__main__":
    m = Model.create("")
    g = GurobiSolver(m)
