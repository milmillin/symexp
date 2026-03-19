from typing import Any

from ..expr import LinExpr, Model, RelOp, Sense, Solution, VType
from ._base import ModelInfeasibleError, ModelUnboundedError, Solver, SolverError, SolverTimeoutError

_INF = float("inf")


def _load_highs():
    try:
        import highspy
    except ModuleNotFoundError as exc:
        if exc.name == "highspy":
            raise ModuleNotFoundError(
                'HighsSolver requires the optional \'highs\' extra. '
                'Install it with `pip install "symexp[highs]"`.'
            ) from exc
        raise

    return highspy


class HighsSolver(Solver[LinExpr]):
    def __init__(
        self,
        model: Model[LinExpr],
        time_limit: float = _INF,
        **options,
    ):
        highspy = _load_highs()
        super().__init__(model)

        if getattr(model, "_type", LinExpr) is not LinExpr:
            raise ValueError("HighsSolver only supports linear models")

        self._highspy = highspy
        self._solver = highspy.Highs()
        self._var_names = [var.name() for var in model.get_vars()]
        self._var_name_to_index = {name: i for i, name in enumerate(self._var_names)}
        self._vtype = {
            VType.CONTINUOUS: highspy.HighsVarType.kContinuous,
            VType.INTEGER: highspy.HighsVarType.kInteger,
            VType.BINARY: highspy.HighsVarType.kInteger,
        }

        if "output_flag" not in options:
            self._set_option("output_flag", False)
        if time_limit != _INF:
            self._set_option("time_limit", time_limit)
        for name, value in options.items():
            self._set_option(name, value)

        inner_vars = [
            self._solver.addVariable(
                lb=var.bound().min,
                ub=var.bound().max,
                type=self._vtype[var.type()],
                name=var.name(),
            )
            for var in model.get_vars()
        ]

        for constr in model.get_constraints():
            expr = constr.get_expr()
            inner_expr = _to_inner_expr(expr, inner_vars)
            name = constr._name
            match constr.get_op():
                case RelOp.EQ:
                    self._solver.addConstr(inner_expr == 0, name=name)
                case RelOp.LE:
                    self._solver.addConstr(inner_expr <= 0, name=name)
                case RelOp.GE:
                    self._solver.addConstr(inner_expr >= 0, name=name)

        obj, sense = model.get_objective()
        if obj.lin_expr():
            inner_obj = _to_inner_expr(obj, inner_vars)
            if sense == Sense.MINIMIZE:
                self._solver.minimize(inner_obj)
            elif sense == Sense.MAXIMIZE:
                self._solver.maximize(inner_obj)
            else:
                raise ValueError(f"invalid sense: {sense}")
        elif sense == Sense.MINIMIZE:
            self._solver.minimize()
        elif sense == Sense.MAXIMIZE:
            self._solver.maximize()
        else:
            raise ValueError(f"invalid sense: {sense}")

    def _set_option(self, name: str, value: Any) -> None:
        status = self._solver.setOptionValue(name, value)
        if status == self._highspy.HighsStatus.kError:
            raise ValueError(f"Failed to set HiGHS option {name!r}")

    def _has_solution(self) -> bool:
        solution = self._solver.getSolution()
        return bool(solution.value_valid)

    def _solve(self):
        run_status = self._solver.run()
        if run_status == self._highspy.HighsStatus.kError:
            raise SolverError(self, "HiGHS failed to solve the model")

        model_status = self._solver.getModelStatus()
        status_text = self._solver.modelStatusToString(model_status)
        status = self._highspy.HighsModelStatus

        if model_status == status.kOptimal:
            return
        if model_status == status.kInfeasible:
            raise ModelInfeasibleError(self)
        if model_status == status.kUnbounded:
            raise ModelUnboundedError(self)
        if model_status == status.kUnboundedOrInfeasible:
            raise SolverError(self, "Model is unbounded or infeasible")
        if model_status in (status.kTimeLimit, status.kIterationLimit, status.kSolutionLimit):
            if self._has_solution():
                return
            raise SolverTimeoutError(self, status_text)
        if model_status in (status.kObjectiveBound, status.kObjectiveTarget):
            if self._has_solution():
                return
            raise SolverError(self, status_text)
        raise SolverError(self, status_text)

    def _get_solution(self) -> Solution:
        solution = self._solver.getSolution()
        if not solution.value_valid:
            raise SolverError(self, "HiGHS did not return a valid primal solution")
        return {name: value for name, value in zip(self._var_names, solution.col_value)}

    def set_solutions(self, *solution: Solution) -> None:
        if len(solution) == 0:
            return
        if len(solution) > 1:
            raise NotImplementedError("HiGHS supports only one warm-start solution at a time")

        indices: list[int] = []
        values: list[float] = []
        for name, value in solution[0].items():
            if name not in self._var_name_to_index:
                raise KeyError(f"Unknown variable {name!r}")
            indices.append(self._var_name_to_index[name])
            values.append(float(value))

        status = self._solver.setSolution(len(indices), indices, values)
        if status == self._highspy.HighsStatus.kError:
            raise SolverError(self, "Failed to set HiGHS warm start")


def _to_inner_expr(expr: LinExpr, inner_vars: list[Any]) -> Any:
    return sum((inner_vars[var.index()] * coeff for var, coeff in expr.lin_expr().items()), expr.const_expr())
