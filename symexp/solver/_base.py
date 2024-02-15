from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, TypeVar, Generic

from ..expr import Model, VType, Sense, RelOp, Solution, LinExpr, QuadExpr
from ..event import Event


_ExprT_con = TypeVar("_ExprT_con", LinExpr, QuadExpr, contravariant=True)


class ModelInfeasibleError(Exception):
    def __init__(self, solver: "Solver", msg: str = ""):
        super().__init__(msg)
        self.solver = solver


class ModelUnboundedError(Exception):
    def __init__(self, solver: "Solver", msg: str = ""):
        super().__init__(msg)
        self.solver = solver


class Solver(ABC, Generic[_ExprT_con]):
    def __init__(self, model: Model[_ExprT_con]):
        self.solution_found = Event["Solver", Solution, float]()
        self.tick = Event["Solver", float]()
        self._og_model = model

    def solve(self) -> Solution:
        self._solve()
        return self._get_solution()

    def _invoke_tick(self, runtime: float):
        self.tick.invoke(self, runtime)

    # Solver needs to implement these functions

    @abstractmethod
    def _solve(self): ...

    @abstractmethod
    def _get_solution(self) -> Solution: ...

    @abstractmethod
    def set_solutions(self, *solution: Solution) -> None: ...
