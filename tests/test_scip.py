import pytest

pytest.importorskip("pyscipopt")

from symexp import Model, QuadExpr, Sense, VType, evaluate
from symexp.solver import ModelInfeasibleError, ModelUnboundedError, ScipSolver


class TestScipSolver:
    def test_solves_linear_program(self):
        m = Model.create("lp")
        x = m.add_var(name="x", lb=0)
        y = m.add_var(name="y", lb=0)
        m.add_constraint(x + y >= 4)
        m.add_constraint(x - y <= 1)
        m.set_objective(Sense.MINIMIZE, 2 * x + y)

        sol = ScipSolver(m).solve()

        assert m.check_solution_satisfies_constraints(sol)
        assert sol["x"] == pytest.approx(0.0)
        assert sol["y"] == pytest.approx(4.0)

    def test_solves_mixed_integer_program(self):
        m = Model.create("mip")
        x = m.add_vars(2, VType.BINARY, name="x")
        m.add_constraint(x[0] + x[1] >= 1)
        m.set_objective(Sense.MINIMIZE, x[0] + x[1])

        sol = ScipSolver(m).solve()

        assert m.check_solution_satisfies_constraints(sol)
        assert sol["x[0]"] + sol["x[1]"] == pytest.approx(1.0)
        assert sol["x[0]"] in (0.0, 1.0)
        assert sol["x[1]"] in (0.0, 1.0)

    def test_supports_quadratic_constraint(self):
        m = Model.create("qcp", QuadExpr)
        x = m.add_var(name="x", lb=0)
        y = m.add_var(name="y", lb=0)
        m.add_constraint(x * x + y <= 4)
        m.set_objective(Sense.MAXIMIZE, x + y)

        sol = ScipSolver(m).solve()

        assert m.check_solution_satisfies_constraints(sol)
        m.set_solution(sol)
        assert evaluate(x + y) == pytest.approx(4.25, abs=1e-3)

    def test_supports_quadratic_objective(self):
        m = Model.create("qo", QuadExpr)
        x = m.add_var(name="x", lb=0)
        y = m.add_var(name="y", lb=0)
        m.add_constraint(x + y >= 1)
        m.set_objective(Sense.MINIMIZE, x * x + y)

        sol = ScipSolver(m).solve()

        assert m.check_solution_satisfies_constraints(sol)
        m.set_solution(sol)
        assert evaluate(x * x + y) == pytest.approx(0.75, abs=1e-4)

    def test_supports_multiple_warm_starts(self):
        m = Model.create("warm")
        x = m.add_vars(2, VType.BINARY, name="x")
        m.add_constraint(x[0] + x[1] >= 1)
        m.set_objective(Sense.MINIMIZE, x[0] + x[1])

        solver = ScipSolver(m)
        solver.set_solutions({"x[0]": 1.0, "x[1]": 0.0}, {"x[0]": 0.0, "x[1]": 1.0})
        sol = solver.solve()

        assert m.check_solution_satisfies_constraints(sol)
        assert sol["x[0]"] + sol["x[1]"] == pytest.approx(1.0)

    def test_emits_solution_found_events(self):
        m = Model.create("event")
        x = m.add_vars(2, VType.BINARY, name="x")
        m.add_constraint(x[0] + x[1] >= 1)
        m.set_objective(Sense.MINIMIZE, x[0] + x[1])

        solver = ScipSolver(m)
        seen = []
        solver.solution_found.register(lambda solver_, sol, info: seen.append((sol, info)))
        sol = solver.solve()

        assert sol["x[0]"] + sol["x[1]"] == pytest.approx(1.0)
        assert len(seen) >= 1

    def test_maps_infeasible_status(self):
        m = Model.create("infeasible")
        x = m.add_var(name="x", lb=0, ub=0)
        m.add_constraint(x >= 1)
        m.set_objective(Sense.MINIMIZE, x)

        with pytest.raises(ModelInfeasibleError):
            ScipSolver(m).solve()

    def test_maps_unbounded_status(self):
        m = Model.create("unbounded")
        x = m.add_var(name="x", lb=float("-inf"))
        m.set_objective(Sense.MAXIMIZE, x)

        with pytest.raises(ModelUnboundedError):
            ScipSolver(m).solve()
