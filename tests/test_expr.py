from ast import Assert
import pytest

from symexp import Model, VType, Var, BinVar, LinExpr, Constr


class TestModel:
    def test_init(self):
        m = Model("test")

    def test_add_var_1(self):
        m = Model("test")

        t1 = m.add_var(VType.BINARY)
        assert isinstance(t1, BinVar)

        t2 = m.add_var(VType.BINARY, ub=5)
        assert isinstance(t2, BinVar)
        assert t2.bound().max == 1
        assert t2.bound().min == 0

        t3 = m.add_var(VType.BINARY, lb=-4, ub=5)
        assert isinstance(t3, BinVar)
        assert t3.bound().max == 1
        assert t3.bound().min == 0

    def test_add_var_2(self):
        m = Model("test")

        t1 = m.add_var(VType.INTEGER, 1, 0)
        assert isinstance(t1, BinVar)

        t2 = m.add_var(VType.INTEGER, 1)
        assert isinstance(t2, BinVar)

        t3 = m.add_var(VType.INTEGER, ub=1)
        assert isinstance(t3, BinVar)

    def test_add_var_3(self):
        m = Model("test")

        t1 = m.add_var(VType.INTEGER, 5, -5)
        assert isinstance(t1, Var)
        assert t1.bound().min == -5 and t1.bound().max == 5

    def test_add_var_4(self):
        m = Model("test")

        t1 = m.add_var()
        assert isinstance(t1, Var)
        assert t1.bound().min == 0 and t1.bound().max == float("inf")

        t2 = m.add_var(lb=5)
        assert isinstance(t2, Var)
        assert t2.bound().min == 5 and t2.bound().max == float("inf")

        t3 = m.add_var(ub=5)
        assert isinstance(t3, Var)
        assert t3.bound().min == 0 and t3.bound().max == 5

        with pytest.raises(AssertionError):
            t4 = m.add_var(ub=5, lb=10)

    def test_add_vars(self):
        m = Model("test")

        t1 = m.add_vars(10)
        assert len(t1) == 10
        assert all(isinstance(t, Var) for t in t1)
        assert all(t.name() == f"T0[{i}]" for i, t in enumerate(t1))

    def test_add_var_duplicate(self):
        m = Model("test")
        a = m.add_var(name="A")
        with pytest.raises(AssertionError):
            b = m.add_var(name="A")
    
    def test_add_constraints(self):
        m = Model[LinExpr]("test")
        a = m.add_var(name="A")
        b = m.add_var(name="B")

        m.add_constraint(a <= b)



class TestVar:
    def test_var_neg(self):
        m = Model("test")
        t = m.add_var()

        expr = -t
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, -1),), 0)

    @pytest.mark.parametrize("coeff", (-1, 0, 1, 1.5))
    def test_var_num_add(self, coeff: float):
        m = Model("test")
        t = m.add_var()

        expr = t + coeff
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1),), coeff)

        expr = coeff + t
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1),), coeff)

    @pytest.mark.parametrize("coeff", (-1, 0, 1, 1.5))
    def test_var_num_sub(self, coeff: float):
        m = Model("test")
        t = m.add_var()

        expr = t - coeff
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1),), -coeff)

        expr = coeff - t
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, -1),), coeff)

    @pytest.mark.parametrize("coeff", (-1, 1, 1.5))
    def test_var_num_mul_nonzero(self, coeff: float):
        m = Model("test")
        t = m.add_var()

        expr = t * coeff
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, coeff),), 0)

        expr = coeff * t
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, coeff),), 0)

    def test_var_num_mul_zero(self):
        m = Model("test")
        t = m.add_var()

        expr = t * 0
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), (), 0)

    @pytest.mark.parametrize("coeff", (-1, 1, 1.5))
    def test_var_num_div_nonzero(self, coeff: float):
        m = Model("test")
        t = m.add_var()

        expr = t / coeff
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1 / coeff),), 0)

    def test_var_num_div_zero(self):
        m = Model("test")
        t = m.add_var()

        with pytest.raises(ZeroDivisionError):
            expr = t / 0

    def test_var_var_add(self):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()

        expr = a + b
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1), (1, 1)), 0)

        expr = a + a
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 2),), 0)

    def test_var_var_sub(self):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()

        expr = a - b
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1), (1, -1)), 0)

        expr = a - a
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), (), 0)

    def test_var_var_div(self):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()

        with pytest.raises(TypeError):
            expr = a / b  # type: ignore


class TestExpr:
    def test_expr_neg(self):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()
        c = m.add_var()

        og = a + 2 * b + 1

        expr = -og
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, -1), (1, -2)), -1)

    @pytest.mark.parametrize("coeff", (-1, 0, 1, 1.5))
    def test_expr_num_add(self, coeff: float):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()

        og = a + 2 * b + 1

        expr = og + coeff
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1), (1, 2)), 1 + coeff)

        expr = coeff + og
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1), (1, 2)), 1 + coeff)

    @pytest.mark.parametrize("coeff", (-1, 0, 1, 1.5))
    def test_expr_num_sub(self, coeff: float):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()

        og = a + 2 * b + 1

        expr = og - coeff
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1), (1, 2)), 1 - coeff)

        expr = coeff - og
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, -1), (1, -2)), -1 + coeff)

    @pytest.mark.parametrize("coeff", (-1, 1, 1.5))
    def test_expr_num_mul_nonzero(self, coeff: float):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()

        og = a + 2 * b + 1

        expr = og * coeff
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1 * coeff), (1, 2 * coeff)), coeff)

        expr = coeff * og
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1 * coeff), (1, 2 * coeff)), coeff)

    def test_expr_num_mul_zero(self):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()

        og = a + 2 * b + 1

        expr = og * 0
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), (), 0)

    @pytest.mark.parametrize("coeff", (-1, 1, 1.5))
    def test_expr_num_div_nonzero(self, coeff: float):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()

        og = a + 2 * b + 1

        expr = og / coeff
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1 / coeff), (1, 2 / coeff)), 1 / coeff)

    def test_expr_num_div_zero(self):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()

        og = a + 2 * b + 1

        with pytest.raises(ZeroDivisionError):
            expr = og / 0

    def test_expr_var_add(self):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()
        c = m.add_var()

        og = a + 2 * b + 1

        expr = og + a
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 2), (1, 2)), 1)

        expr = og + b
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1), (1, 3)), 1)

        expr = og + c
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1), (1, 2), (2, 1)), 1)

    def test_expr_var_sub(self):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()
        c = m.add_var()

        og = a + 2 * b + 1

        expr = og - a
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((1, 2),), 1)

        expr = og - b
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1), (1, 1)), 1)

        expr = og - c
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1), (1, 2), (2, -1)), 1)

    def test_expr_expr_add(self):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()
        c = m.add_var()

        og = -a + 2 * b + 1

        expr = og + (b + 2)
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, -1), (1, 3)), 3)

        expr = og + (a + 2)
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((1, 2),), 3)

        expr = og + (c + 2)
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, -1), (1, 2), (2, 1)), 3)

    def test_expr_expr_sub(self):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()
        c = m.add_var()

        og = a + 2 * b + 1

        expr = og - (b + 2)
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1), (1, 1)), -1)

        expr = og - (a + 2)
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((1, 2),), -1)

        expr = og - (c + 2)
        assert isinstance(expr, LinExpr)
        assert expr._signature() == ((), ((0, 1), (1, 2), (2, -1)), -1)


class TestConstr:
    def test_var_num(self):
        m = Model("test")
        a = m.add_var()

        assert isinstance(a == 1, Constr)
        assert isinstance(a <= 1, Constr)
        assert isinstance(a >= 1, Constr)
        assert isinstance(1 == a, Constr)
        assert isinstance(1 <= a, Constr)
        assert isinstance(1 >= a, Constr)

    def test_var_var(self):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()

        assert isinstance(a == a, Constr)
        assert isinstance(a <= a, Constr)
        assert isinstance(a >= a, Constr)
        assert isinstance(a == a, Constr)
        assert isinstance(a <= a, Constr)
        assert isinstance(a >= a, Constr)
        assert isinstance(a == b, Constr)
        assert isinstance(a <= b, Constr)
        assert isinstance(a >= b, Constr)
        assert isinstance(b == a, Constr)
        assert isinstance(b <= a, Constr)
        assert isinstance(b >= a, Constr)

    def test_expr_num(self):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()

        expr = a + 2 * b + 1

        assert isinstance(expr == 1, Constr)
        assert isinstance(expr <= 1, Constr)
        assert isinstance(expr >= 1, Constr)
        assert isinstance(1 == expr, Constr)
        assert isinstance(1 <= expr, Constr)
        assert isinstance(1 >= expr, Constr)

    def test_expr_var(self):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()
        c = m.add_var()

        expr = a + 2 * b + 1

        assert isinstance(expr == a, Constr)
        assert isinstance(expr <= a, Constr)
        assert isinstance(expr >= a, Constr)
        assert isinstance(a == expr, Constr)
        assert isinstance(a <= expr, Constr)
        assert isinstance(a >= expr, Constr)
        assert isinstance(expr == b, Constr)
        assert isinstance(expr <= b, Constr)
        assert isinstance(expr >= b, Constr)
        assert isinstance(b == expr, Constr)
        assert isinstance(b <= expr, Constr)
        assert isinstance(b >= expr, Constr)
        assert isinstance(expr == c, Constr)
        assert isinstance(expr <= c, Constr)
        assert isinstance(expr >= c, Constr)
        assert isinstance(c == expr, Constr)
        assert isinstance(c <= expr, Constr)
        assert isinstance(c >= expr, Constr)

    def test_expr_expr(self):
        m = Model("test")
        a = m.add_var()
        b = m.add_var()
        c = m.add_var()

        expr1 = a + 2 * b + 1
        expr2 = b + 2 * c

        assert isinstance(expr1 == expr2, Constr)
        assert isinstance(expr1 <= expr2, Constr)
        assert isinstance(expr1 >= expr2, Constr)
        assert isinstance(expr2 == expr1, Constr)
        assert isinstance(expr2 <= expr1, Constr)
        assert isinstance(expr2 >= expr1, Constr)
