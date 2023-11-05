import pytest

from symexp import Model, VType, Var, BinVar, LinExpr, Constr, Sense
from symexp.solver import GurobiSolver


class TestMipHelper:
    @pytest.mark.parametrize("n", range(2, 5))
    def test_and(self, n: int):
        m = Model.create("test")
        x = m.add_vars(n, VType.BINARY, name="x")
        y = m.and_(*x, name="y")

        for i in range(1 << n):
            x_ = [(i >> j) & 1 for j in range(n)]
            exp = float(all(x_))
            sol = {f"x[{j}]": x_val for j, x_val in enumerate(x_)}
            good_sol = {**sol, "y": exp}
            assert m.check_solution_satisfies_constraints(good_sol)
            bad_sol = {**sol, "y": 1 - exp}
            assert not m.check_solution_satisfies_constraints(bad_sol)

    @pytest.mark.parametrize("n", range(2, 5))
    def test_or(self, n: int):
        m = Model.create("test")
        x = m.add_vars(n, VType.BINARY, name="x")
        y = m.or_(*x, name="y")

        for i in range(1 << n):
            x_ = [(i >> j) & 1 for j in range(n)]
            exp = float(any(x_))
            sol = {f"x[{j}]": x_val for j, x_val in enumerate(x_)}
            good_sol = {**sol, "y": exp}
            assert m.check_solution_satisfies_constraints(good_sol)
            bad_sol = {**sol, "y": 1 - exp}
            assert not m.check_solution_satisfies_constraints(bad_sol)

    @pytest.mark.parametrize("x", range(0, 5))
    @pytest.mark.parametrize("y", range(0, 5))
    def test_min(self, x: float, y: float):
        m = Model.create("test")
        x_ = m.add_var(name="x")
        y_ = m.add_var(name="y")
        _ = m.min_(x_, y_, name="z")
        m.set_objective(Sense.MINIMIZE, x_)
        m.add_constraint(x_ == x)
        m.add_constraint(y_ == y)
        solver = GurobiSolver(m)
        sol = solver.solve()
        assert sol["z"] == min(x, y)
        assert (sol["z/d[0]"] == 1) <= (x == min(x, y))
        assert (sol["z/d[1]"] == 1) <= (y == min(x, y))

    @pytest.mark.parametrize("x", range(0, 5))
    @pytest.mark.parametrize("y", range(0, 5))
    def test_max(self, x: float, y: float):
        m = Model.create("test")
        x_ = m.add_var(name="x")
        y_ = m.add_var(name="y")
        _ = m.max_(x_, y_, name="z")
        m.set_objective(Sense.MINIMIZE, x_)
        m.add_constraint(x_ == x)
        m.add_constraint(y_ == y)
        solver = GurobiSolver(m)
        sol = solver.solve()
        assert sol["z"] == max(x, y)
        assert (sol["z/d[0]"] == 1) <= (x == max(x, y))
        assert (sol["z/d[1]"] == 1) <= (y == max(x, y))

    @pytest.mark.parametrize("x", range(-5, 5))
    def test_abs(self, x: float):
        m = Model.create("test")
        x_ = m.add_var(name="x", lb=float("-inf"))
        m.add_constraint(x_ == x)
        y = m.abs_(x, name="y")
        m.set_objective(Sense.MINIMIZE, x_)
        solver = GurobiSolver(m)
        sol = solver.solve()
        assert sol["y"] == abs(x)


    @pytest.mark.parametrize("n", range(2, 5))
    def test_mux(self, n: int):
        m = Model.create("test")
        x = m.add_vars(n, VType.BINARY, name="x")
        y = m.mux_(*((x_, i * 10) for i, x_ in enumerate(x)), name="y")

        for i in range(n):
            x_ = {f"x[{j}]": 1.0 if j == i else 0 for j in range(n)}
            for j in range(n):
                sol = {**x_, "y": j * 10}
                assert (i == j) == m.check_solution_satisfies_constraints(sol)


    @pytest.mark.parametrize("n", range(2, 5))
    def test_min_en_(self, n: int):
        m = Model.create("test")
        x = m.add_vars(n, VType.BINARY, name="x")
        y = m.min_en_(*((x_, i * 10) for i, x_ in enumerate(x)), name="y")

        for i in range(1, 1 << n):
            x_ = {f"x[{j}]": (i >> j) & 1 for j in range(n)}
            exp = min(j * 10 for j in range(n) if (i >> j) & 1)
            sol = {**x_, "y": float(exp)}
            assert m.check_solution_satisfies_constraints(sol)

