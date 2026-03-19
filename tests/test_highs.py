import pytest

pytest.importorskip("highspy")

from symexp import Model, QuadExpr, Sense, VType
from symexp.solver import HighsSolver, ModelInfeasibleError, ModelUnboundedError


class TestHighsSolver:
    def test_solves_linear_program(self):
        m = Model.create("lp")
        x = m.add_var(name="x", lb=0)
        y = m.add_var(name="y", lb=0)
        m.add_constraint(x + y >= 4)
        m.add_constraint(x - y <= 1)
        m.set_objective(Sense.MINIMIZE, 2 * x + y)

        sol = HighsSolver(m).solve()

        assert m.check_solution_satisfies_constraints(sol)
        assert sol["x"] == pytest.approx(0.0)
        assert sol["y"] == pytest.approx(4.0)

    def test_solves_mixed_integer_program(self):
        m = Model.create("mip")
        x = m.add_vars(2, VType.BINARY, name="x")
        m.add_constraint(x[0] + x[1] >= 1)
        m.set_objective(Sense.MINIMIZE, x[0] + x[1])

        sol = HighsSolver(m).solve()

        assert m.check_solution_satisfies_constraints(sol)
        assert sol["x[0]"] + sol["x[1]"] == pytest.approx(1.0)
        assert sol["x[0]"] in (0.0, 1.0)
        assert sol["x[1]"] in (0.0, 1.0)

    def test_supports_maximize(self):
        m = Model.create("max")
        x = m.add_var(name="x", lb=0, ub=3)
        y = m.add_var(name="y", lb=0, ub=2)
        m.add_constraint(x + y <= 4)
        m.set_objective(Sense.MAXIMIZE, 3 * x + y)

        sol = HighsSolver(m).solve()

        assert m.check_solution_satisfies_constraints(sol)
        assert sol["x"] == pytest.approx(3.0)
        assert sol["y"] == pytest.approx(1.0)

    def test_supports_single_warm_start(self):
        m = Model.create("warm")
        x = m.add_vars(2, VType.BINARY, name="x")
        m.add_constraint(x[0] + x[1] >= 1)
        m.set_objective(Sense.MINIMIZE, x[0] + x[1])

        solver = HighsSolver(m)
        solver.set_solutions({"x[0]": 1.0, "x[1]": 0.0})
        sol = solver.solve()

        assert m.check_solution_satisfies_constraints(sol)
        assert sol["x[0]"] + sol["x[1]"] == pytest.approx(1.0)

    def test_rejects_quadratic_models(self):
        m = Model.create("qp", QuadExpr)
        x = m.add_var(name="x")
        m.set_objective(Sense.MINIMIZE, x * x + x)

        with pytest.raises(ValueError, match="linear models"):
            HighsSolver(m)

    def test_rejects_multiple_warm_starts(self):
        m = Model.create("warm")
        x = m.add_var(name="x")
        m.set_objective(Sense.MINIMIZE, x)

        solver = HighsSolver(m)

        with pytest.raises(NotImplementedError, match="one warm-start solution"):
            solver.set_solutions({"x": 0.0}, {"x": 1.0})

    def test_maps_infeasible_status(self):
        m = Model.create("infeasible")
        x = m.add_var(name="x", lb=0, ub=0)
        m.add_constraint(x >= 1)
        m.set_objective(Sense.MINIMIZE, x)

        with pytest.raises(ModelInfeasibleError):
            HighsSolver(m).solve()

    def test_maps_unbounded_status(self):
        m = Model.create("unbounded")
        x = m.add_var(name="x", lb=float("-inf"))
        m.set_objective(Sense.MAXIMIZE, x)

        with pytest.raises(ModelUnboundedError):
            HighsSolver(m).solve()
