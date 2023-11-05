from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, TypeVar, Generic

from ..expr import Model, VType, Sense, RelOp, Solution, LinExpr, QuadExpr
from ..event import Event

_Var = Any
_Expr = Any

_ExprT_con = TypeVar("_ExprT_con", LinExpr, QuadExpr, contravariant=True)

class Solver(ABC, Generic[_ExprT_con]):
    def __init__(self, model: Model[_ExprT_con]):
        self.solution_found = Event[Solution, float]()
        inner_vars = [self._add_var(v.name(), v.type(), v.bound().min, v.bound().max) for v in model.get_vars()]
        for constr in model.get_constraint():
            expr = constr.get_expr()
            op = constr.get_op()
            inner_expr = _to_inner_expr(expr, inner_vars)
            match op:
                case RelOp.EQ:
                    self._add_constraint(inner_expr == 0, constr._name or "")
                case RelOp.LE:
                    self._add_constraint(inner_expr <= 0, constr._name or "")
                case RelOp.GE:
                    self._add_constraint(inner_expr >= 0, constr._name or "")
        obj, obj_sense = model.get_objective()
        inner_obj = _to_inner_expr(obj, inner_vars)
        self._set_objective(inner_obj, obj_sense)

    def solve(self) -> Solution:
        try:
            self._solve()
        except Exception as e:
            print(e)
        return self._get_solution()

    # Solver needs to implement these functions

    @abstractmethod
    def _add_var(
        self,
        name: str,
        vtype: VType,
        lb: float,
        ub: float,
    ) -> _Var:
        ...

    @abstractmethod
    def _add_constraint(self, constraint: Any, name: str) -> None:
        ...

    @abstractmethod
    def _set_objective(self, objective: _Expr, sense: Sense) -> None:
        ...

    @abstractmethod
    def _solve(self):
        ...

    @abstractmethod
    def _get_solution(self) -> Solution:
        ...

    @abstractmethod
    def set_solutions(self, *solution: Solution) -> None:
        ...


# ------------
# Utils


def _to_inner_expr(expr: QuadExpr, inner_vars: Sequence[_Var]) -> _Expr:
    return sum(
        (inner_vars[v1.index()] * inner_vars[v2.index()] * coeff for (v1, v2), coeff in expr.quad_expr().items()), 0
    ) + sum((inner_vars[var.index()] * coeff for var, coeff in expr.lin_expr().items()), expr.const_expr())
