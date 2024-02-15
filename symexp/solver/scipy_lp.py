from scipy.optimize import linprog
from scipy.sparse import csr_array
import numpy as np

from ..expr import Model, RelOp, Sense, Solution, VType, QuadExpr, LinExpr
from ._base import Solver, _ExprT_con, SolverError, SolverTimeoutError, ModelInfeasibleError, ModelUnboundedError


class ScipyLpSolver(Solver[LinExpr]):
    def __init__(
        self,
        model: Model[LinExpr],
    ):
        super().__init__(model)
        vars = model.get_vars()
        n_vars = len(vars)
        self._var_names = [v.name() for v in vars]

        self.bounds: list[tuple[float, float]] = []
        for var in vars:
            if var.type() != VType.CONTINUOUS:
                raise ValueError("only continuous variable is supported")
            self.bounds.append(var.bound())

        A_ub_: list[list[tuple[int, float]]] = []
        B_ub_: list[float] = []
        A_eq_: list[list[tuple[int, float]]] = []
        B_eq_: list[float] = []

        for constr in model.get_constraints():
            expr = constr.get_expr()
            op = constr.get_op()
            terms = expr.lin_expr()
            rhs = expr.const_expr()
            if op == RelOp.EQ:
                A_eq_.append([(var.index(), coeff) for var, coeff in terms.items()])
                B_eq_.append(-rhs)
            elif op == RelOp.LE:
                A_ub_.append([(var.index(), coeff) for var, coeff in terms.items()])
                B_ub_.append(-rhs)
            elif op == RelOp.GE:
                A_ub_.append([(var.index(), -coeff) for var, coeff in terms.items()])
                B_ub_.append(rhs)

        expr, sense = model.get_objective()
        if sense == Sense.MAXIMIZE:
            c_ = [(var.index(), -coeff) for var, coeff in expr.lin_expr().items()]
        elif sense == Sense.MINIMIZE:
            c_ = [(var.index(), coeff) for var, coeff in expr.lin_expr().items()]
        else:
            raise ValueError(f"invalid sense: {sense}")

        self.c = _create_vector(n_vars, c_)
        self.A_ub = _create_matrix(n_vars, A_ub_)
        self.B_ub = np.array(B_ub_)
        self.A_eq = _create_matrix(n_vars, A_eq_)
        self.B_eq = np.array(B_eq_)

    def _solve(self):
        res = linprog(self.c, self.A_ub, self.B_ub, self.A_eq, self.B_eq, self.bounds)
        status = res["status"]
        if status == 1:
            raise SolverTimeoutError(self, "Iteration limit reached")
        elif status == 2:
            raise ModelInfeasibleError(self)
        elif status == 3:
            raise ModelUnboundedError(self)
        elif status == 4:
            raise SolverError(self, "Numerical difficulties encountered")
        self._sol = {self._var_names[i]: x for i, x in enumerate(res["x"])}

    def _get_solution(self) -> Solution:
        return self._sol

    def set_solutions(self, *solution: Solution) -> None:
        raise NotImplementedError()


# Utils


def _create_vector(size: int, data: list[tuple[int, float]]) -> np.ndarray:
    res = np.zeros((size,))
    for var, coeff in data:
        res[var] = coeff
    return res


def _create_matrix(m: int, data: list[list[tuple[int, float]]]) -> csr_array:
    res = csr_array((len(data), m))
    for i, terms in enumerate(data):
        for var, coeff in terms:
            res[i, var] = coeff
    return res


# check if all abstract methods are implemented
if __name__ == "__main__":
    m = Model.create("")
    g = ScipyLpSolver(m)
