from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Optional,
    Union,
    overload,
    TypeVar,
    Generic,
    NamedTuple,
    Literal,
    Iterable,
    Sequence,
    Callable,
    ParamSpec,
)
from dataclasses import dataclass
import math
from typeguard import typechecked, check_type
import functools


class VType(Enum):
    BINARY = 0
    INTEGER = 1
    CONTINUOUS = 2


class Sense(Enum):
    MAXIMIZE = -1
    MINIMIZE = 1


class RelOp(Enum):
    EQ = "=="
    LE = "<="
    GE = ">="


# --------------
# Interfaces

Solution = dict[str, float]
Number = Union[float, int]
QuadExprOrNumber = Union["QuadExpr", "LinExpr", float, int]
LinExprOrNumber = Union["LinExpr", float, int]
BinExprOrLiteral = Union["BinExpr", Literal[0, 1]]

_Signature = tuple[tuple[tuple[int, int, float], ...], tuple[tuple[int, float], ...], float]


_EPS = 1e-63
_INF = float("inf")


class Expr(ABC):
    @abstractmethod
    def model(self) -> "Model":
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...

    @abstractmethod
    def compact_repr(self) -> str:
        ...

    @abstractmethod
    def __float__(self) -> float:
        ...

    @abstractmethod
    def _signature(self) -> _Signature:
        ...

    def __hash__(self):
        return hash(self._signature())

    @abstractmethod
    def bound(self) -> "Bound":
        ...

    @abstractmethod
    def is_continuous(self) -> bool:
        ...

    def is_binary(self) -> bool:
        return self.bound() == (0, 1) and not self.is_continuous()


class QuadExpr(Expr):
    @abstractmethod
    def quad_expr(self) -> dict[tuple["Var", "Var"], float]:
        ...

    @abstractmethod
    def lin_expr(self) -> dict["Var", float]:
        ...

    @abstractmethod
    def const_expr(self) -> Number:
        ...

    # --------------
    # IExpr

    def __repr__(self) -> str:
        res = ""
        quad_vars = sorted(self.quad_expr().items(), key=lambda kv: (kv[0][0].index(), kv[0][1].index()))
        first = True
        for (v1, v2), coeff in quad_vars:
            sgn = "+" if coeff > 0 else "-"
            if first and sgn == "+":
                sgn = ""
            coeff_ = str(abs(coeff)) if abs(coeff) - 1 > _EPS else ""
            res += f"{sgn} {coeff_}{v1}{v2} "
            first = False
        lin_vars = sorted(self.lin_expr().items(), key=lambda kv: kv[0].index())
        for v, coeff in lin_vars:
            sgn = "+" if coeff > 0 else "-"
            if first and sgn == "+":
                sgn = ""
            coeff_ = str(abs(coeff)) if abs(coeff) - 1 > _EPS else ""
            res += f"{sgn} {coeff_}{v} "
            first = False
        const = self.const_expr()
        if abs(const) > _EPS:
            sgn = "+" if const > 0 else "-"
            if first and sgn == "+":
                sgn = ""
            res += f"{sgn} {abs(const)}"
        return res.strip()

    def compact_repr(self) -> str:
        res = ""
        quad_vars = sorted(self.quad_expr().items(), key=lambda kv: (kv[0][0].index(), kv[0][1].index()))
        first = True
        for (v1, v2), coeff in quad_vars:
            sgn = "+" if coeff > 0 else "-"
            if first and sgn == "+":
                sgn = ""
            coeff_ = str(abs(coeff)) if abs(coeff) - 1 > _EPS else ""
            res += f"{sgn}{coeff_}{v1.compact_repr()}{v2.compact_repr()}"
            first = False
        lin_vars = sorted(self.lin_expr().items(), key=lambda kv: kv[0].index())
        for v, coeff in lin_vars:
            sgn = "+" if coeff > 0 else "-"
            if first and sgn == "+":
                sgn = ""
            coeff_ = str(abs(coeff)) if abs(coeff) - 1 > _EPS else ""
            res += f"{sgn}{coeff_}{v.compact_repr()}"
            first = False
        const = self.const_expr()
        if abs(const) > _EPS:
            sgn = "+" if const > 0 else "-"
            if first and sgn == "+":
                sgn = ""
            res += f"{sgn}{abs(const)}"
        return res.strip()

    def __float__(self) -> float:
        res = self.const_expr()
        for (v1, v2), coeff in self.quad_expr().items():
            res += float(v1) * float(v2) * coeff
        for v, coeff in self.lin_expr().items():
            res += float(v) * coeff
        return res

    def _signature(self) -> _Signature:
        quad_vars = sorted(self.quad_expr().items(), key=lambda kv: (kv[0][0].index(), kv[0][1].index()))
        lin_vars = sorted(self.lin_expr().items(), key=lambda kv: kv[0].index())
        return (
            tuple((v1.index(), v2.index(), c) for (v1, v2), c in quad_vars),
            tuple((v.index(), c) for v, c in lin_vars),
            float(self.const_expr()),
        )

    def bound(self) -> "Bound":
        return (
            sum((v1.bound() * v2.bound() * coeff for (v1, v2), coeff in self.quad_expr().items()), Bound(0, 0))
            + sum((var.bound() * coeff for var, coeff in self.lin_expr().items()), Bound(0, 0))
            + self.const_expr()
        )

    def is_continuous(self) -> bool:
        return (
            any(
                v1.is_continuous() or v2.is_continuous() or isinstance(coeff, float)
                for (v1, v2), coeff in self.quad_expr().items()
            )
            or any(var.is_continuous() or isinstance(coeff, float) for var, coeff in self.lin_expr().items())
            or isinstance(self.const_expr(), float)
        )

    # --------------
    # Operators

    def __add__(self, other: QuadExprOrNumber) -> "QuadExpr":
        if isinstance(other, (float, int)):
            if abs(other) < _EPS:
                return self
            return _QuadExpr(self.model(), self.quad_expr(), self.lin_expr(), self.const_expr() + other)
        assert self.model() == other.model(), "Expressions must come from the same model"
        return _QuadExpr(
            self.model(),
            _add_expr(self.quad_expr(), other.quad_expr()),
            _add_expr(self.lin_expr(), other.lin_expr()),
            self.const_expr() + other.const_expr(),
        )

    def __mul__(self, other: Union[float, int]) -> "QuadExpr":
        if abs(other - 1) < _EPS:
            return self
        return _QuadExpr(
            self.model(),
            _mul_expr_const(self.quad_expr(), other),
            _mul_expr_const(self.lin_expr(), other),
            self.const_expr() * other,
        )

    def __radd__(self, other: QuadExprOrNumber) -> "QuadExpr":
        return self + other

    def __sub__(self, other: QuadExprOrNumber) -> "QuadExpr":
        return self + -other

    def __rsub__(self, other: QuadExprOrNumber) -> "QuadExpr":
        return -self + other

    def __rmul__(self, other: Union[float, int]) -> "QuadExpr":
        return self * other

    def __truediv__(self, other: Union[float, int]) -> "QuadExpr":
        return self * (1 / other)

    def __neg__(self) -> "QuadExpr":
        return self.__mul__(-1)

    # ---------------
    # Comparator

    def __eq__(self, other: QuadExprOrNumber) -> "Constr[QuadExpr]":
        return Constr[QuadExpr](self, RelOp.EQ, self.model()._convert(other))

    def __le__(self, other: QuadExprOrNumber) -> "Constr[QuadExpr]":
        return Constr[QuadExpr](self, RelOp.LE, self.model()._convert(other))

    def __ge__(self, other: QuadExprOrNumber) -> "Constr[QuadExpr]":
        return Constr[QuadExpr](self, RelOp.GE, self.model()._convert(other))


class LinExpr(QuadExpr):
    # -----------
    # IQuadExpr

    def quad_expr(self) -> dict[tuple["Var", "Var"], float]:
        return {}

    # --------------
    # Operators

    def __add__(self, other: LinExprOrNumber) -> "LinExpr":
        if isinstance(other, (float, int)):
            if abs(other) < _EPS:
                return self
            return _LinExpr(self.model(), self.lin_expr(), self.const_expr() + other)
        assert self.model() == other.model(), "Expressions must come from the same model"
        return _LinExpr(
            self.model(),
            _add_expr(self.lin_expr(), other.lin_expr()),
            self.const_expr() + other.const_expr(),
        )

    @overload
    def __mul__(self, other: "LinExpr") -> "QuadExpr":
        ...

    @overload
    def __mul__(self, other: Union[float, int]) -> "LinExpr":
        ...

    def __mul__(self, other: Union["LinExpr", float, int]) -> Union["LinExpr", "QuadExpr"]:
        if isinstance(other, (float, int)):
            if abs(other - 1) < _EPS:
                return self
            return _LinExpr(
                self.model(),
                _mul_expr_const(self.lin_expr(), other),
                self.const_expr() * other,
            )
        else:
            assert self.model() == other.model(), "Expressions must come from the same model"
            quad_expr: dict[tuple[Var, Var], float] = {}
            for var1, coeff1 in self.lin_expr().items():
                for var2, coeff2 in other.lin_expr().items():
                    vs = (var1, var2) if var1.index() <= var2.index() else (var2, var1)
                    quad_expr[vs] = quad_expr.get(vs, 0) + (coeff1 * coeff2)
            lin_expr = _add_expr(
                _mul_expr_const(self.lin_expr(), other.const_expr()),
                _mul_expr_const(other.lin_expr(), self.const_expr()),
            )
            return _QuadExpr(self.model(), quad_expr, lin_expr, self.const_expr() * other.const_expr())

    def __radd__(self, other: LinExprOrNumber) -> "LinExpr":
        return self + other

    def __sub__(self, other: LinExprOrNumber) -> "LinExpr":
        return self + -other

    def __rsub__(self, other: LinExprOrNumber) -> "LinExpr":
        return -self + other

    def __rmul__(self, other: Union[float, int]) -> "LinExpr":
        return self * other

    def __truediv__(self, other: Union[float, int]) -> "LinExpr":
        return self * (1 / other)

    def __neg__(self) -> "LinExpr":
        return self.__mul__(-1)

    # ---------------
    # Comparator

    def __eq__(self, other: LinExprOrNumber) -> "Constr[LinExpr]":
        return Constr[LinExpr](self, RelOp.EQ, self.model()._convert(other))

    def __le__(self, other: LinExprOrNumber) -> "Constr[LinExpr]":
        return Constr[LinExpr](self, RelOp.LE, self.model()._convert(other))

    def __ge__(self, other: LinExprOrNumber) -> "Constr[LinExpr]":
        return Constr[LinExpr](self, RelOp.GE, self.model()._convert(other))

    # ----------
    # Utils

    @abstractmethod
    def to_bin_expr(self) -> "BinExpr":
        ...


class BinExpr(LinExpr):
    @abstractmethod
    def __invert__(self) -> "BinExpr":
        ...

    def to_bin_expr(self) -> "BinExpr":
        return self


class Var(LinExpr):
    @abstractmethod
    def index(self) -> int:
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def value(self) -> Optional[float]:
        ...

    @abstractmethod
    def set_value(self, value: float) -> None:
        ...

    @abstractmethod
    def type(self) -> VType:
        ...

    def __hash__(self):
        return self.index()

    def __repr__(self) -> str:
        if self.value() is not None:
            return f"<{self.name()}={self.value()}>"
        return f"<{self.name()}>"

    def compact_repr(self) -> str:
        return f"<{self.name()}>"

    def __float__(self) -> float:
        value = self.value()
        assert value is not None, "No solution has been set"
        return value

    # -------
    # IQuadExpr

    def lin_expr(self) -> dict["Var", float]:
        return {self: 1}

    def const_expr(self) -> Number:
        return 0


class BinVar(Var, BinExpr):
    pass


# ----------------------
# Implementation


class _QuadExpr(QuadExpr):
    @typechecked
    def __init__(
        self,
        model: "Model",
        quad_expr: dict[tuple[Var, Var], float] = {},
        lin_expr: dict[Var, float] = {},
        const_expr: Number = 0,
    ):
        self._model = model
        self._quad_expr: dict[tuple[Var, Var], float] = dict(quad_expr)
        self._lin_expr: dict[Var, float] = dict(lin_expr)
        self._const_expr: float = const_expr

        assert not any(math.isnan(coeff) for var, coeff in self._quad_expr.items()), "Coeff cannot be nan"
        assert not any(math.isnan(coeff) for var, coeff in self._lin_expr.items()), "Coeff cannot be nan"
        assert not math.isnan(self._const_expr), "Const cannot be nan"

        # remove 0 coeff
        for var, coeff in quad_expr.items():
            if abs(coeff) < _EPS:
                del self._quad_expr[var]
        for var, coeff in lin_expr.items():
            if abs(coeff) < _EPS:
                del self._lin_expr[var]

    # -------------
    # IExpr

    def model(self) -> "Model":
        return self._model

    # -------------
    # IQuadExpr

    def quad_expr(self) -> dict[tuple[Var, Var], float]:
        return self._quad_expr

    def lin_expr(self) -> dict[Var, float]:
        return self._lin_expr

    def const_expr(self) -> float:
        return self._const_expr


class _LinExpr(LinExpr):
    @typechecked
    def __init__(
        self,
        model: "Model",
        lin_expr: dict[Var, float] = {},
        const_expr: Number = 0,
    ):
        self._model = model
        self._lin_expr: dict[Var, float] = dict(lin_expr)
        self._const_expr: float = const_expr

        assert not any(math.isnan(coeff) for var, coeff in self._lin_expr.items()), "Coeff cannot be nan"
        assert not math.isnan(self._const_expr), "Const cannot be nan"

        # remove 0 coeff
        for var, coeff in lin_expr.items():
            if abs(coeff) < _EPS:
                del self._lin_expr[var]

    # -------------
    # IExpr

    def model(self) -> "Model":
        return self._model

    # -------------
    # ILinExpr

    def lin_expr(self) -> dict[Var, float]:
        return self._lin_expr

    def const_expr(self) -> float:
        return self._const_expr

    def to_bin_expr(self) -> BinExpr:
        if len(self._lin_expr) == 1:
            var, coeff = next(iter(self._lin_expr.items()))
            if isinstance(var, BinVar):
                if coeff == 1 and self.const_expr() == 0:
                    return var
                if coeff == -1 and self.const_expr() == 1:
                    return _BinVarNeg(var)
        elif len(self._lin_expr) == 0:
            const = self.const_expr()
            if isinstance(const, int) and (const == 0 or const == 1):
                return _BinLiteral(self._model, const)
        return _UncheckedBinExpr(self)


class _TypVar(Var):
    def __init__(
        self,
        model: "Model",
        idx: int,
        name: str,
        lb: float,
        ub: float,
        vtype: VType,
    ):
        assert lb <= ub, "Lower bound has to be less than or equal to upper bound"
        assert vtype != VType.BINARY, "Create BinVar instead"
        self._model = model
        self._idx = idx
        self._name = name
        self._lb = lb
        self._ub = ub
        self._vtype = vtype
        self._value: Optional[float] = None

    # ------
    # IVar

    def model(self) -> "Model":
        return self._model

    def index(self) -> int:
        return self._idx

    def name(self) -> str:
        return self._name

    def bound(self) -> "Bound":
        return Bound(self._lb, self._ub)

    def type(self) -> VType:
        return self._vtype

    def value(self) -> Optional[float]:
        return self._value

    def set_value(self, value: float) -> None:
        self._value = value

    def is_continuous(self) -> bool:
        return self._vtype == VType.CONTINUOUS

    def to_bin_expr(self) -> BinExpr:
        return _UncheckedBinExpr(self)


class _BinVar(BinVar):
    def __init__(
        self,
        model: "Model",
        idx: int,
        name: str,
    ):
        self._model = model
        self._idx = idx
        self._name = name
        self._value: Optional[float] = None

    # ------
    # IVar

    def model(self) -> "Model":
        return self._model

    def index(self) -> int:
        return self._idx

    def name(self) -> str:
        return self._name

    def bound(self) -> "Bound":
        return Bound(0, 1)

    def type(self) -> VType:
        return VType.BINARY

    def value(self) -> Optional[float]:
        return self._value

    def set_value(self, value: float) -> None:
        assert value == 1 or value == 0, "Invalid value"
        self._value = value

    def is_continuous(self) -> bool:
        return False

    def is_binary(self) -> bool:
        return True

    def __invert__(self) -> BinExpr:
        return _BinVarNeg(self)


class _BinLiteral(BinExpr):
    def __init__(self, model: "Model", value: Literal[0, 1]):
        self._model = model
        self._value = value

    def model(self) -> "Model":
        return self._model

    def lin_expr(self) -> dict[Var, float]:
        return {}

    def const_expr(self) -> Number:
        return self._value

    def __invert__(self) -> BinExpr:
        return _BinLiteral(self._model, 1 if self._value == 0 else 0)


class _BinVarNeg(BinExpr):
    def __init__(self, var: BinVar):
        self._var = var

    def model(self) -> "Model":
        return self._var.model()

    def lin_expr(self) -> dict[Var, float]:
        return {self._var: -1}

    def const_expr(self) -> Number:
        return 1

    def __invert__(self) -> BinExpr:
        return self._var


class _UncheckedBinExpr(_LinExpr, BinExpr):
    def __init__(self, expr: LinExpr):
        _LinExpr.__init__(self, expr.model(), expr.lin_expr(), expr.const_expr())

    def __invert__(self) -> BinExpr:
        return _UncheckedBinExpr(1 - self)


# --------------------
# Constraint

_ExprT_co = TypeVar("_ExprT_co", QuadExpr, LinExpr, covariant=True)


class Constr(Generic[_ExprT_co]):
    def __init__(self, lhs: _ExprT_co, op: RelOp, rhs: _ExprT_co):
        assert lhs.model() == rhs.model(), "LHS and RHS must be from the same model."
        self._lhs = lhs
        self._rhs = rhs
        self._op = op
        self._name: Optional[str] = None

    def get_expr(self) -> _ExprT_co:
        return self._lhs - self._rhs

    def get_op(self) -> RelOp:
        return self._op

    def satisfied(self) -> bool:
        lhs = float(self._lhs)
        rhs = float(self._rhs)
        match self._op:
            case RelOp.EQ:
                return abs(lhs - rhs) < 1e-7
            case RelOp.LE:
                return lhs <= rhs + 1e-7
            case RelOp.GE:
                return lhs >= rhs - 1e-7

    def __repr__(self):
        return f"{self._name}:\t{self._lhs} {self._op.value} {self._rhs}"


# -------------
# Model


def _gen_var_name(func: str, var_list: Iterable[Union[Expr, Number]]):
    return func + "(" + ",".join(var.compact_repr() if isinstance(var, Expr) else str(var) for var in var_list) + ")"


@dataclass
class _MinMaxResult(Generic[_ExprT_co]):
    value: _ExprT_co
    index: Sequence[BinExpr]


_P = ParamSpec("_P")
_T = TypeVar("_T")


def _cache(func: Callable[_P, _T]) -> Callable[_P, _T]:
    func_2 = functools.cache(func)

    @functools.wraps(func)
    def func_(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        return func_2(*args, **kwargs)

    return func_


_ExprT = TypeVar("_ExprT", QuadExpr, LinExpr)


class Model(Generic[_ExprT]):
    def __init__(self, name: str, M: float = 1e5):
        self._name = name
        self._vars: list[Var] = []
        self._constrs: list[Constr[QuadExpr]] = []
        self._obj: Optional[QuadExpr] = None
        self._obj_sense: Optional[Sense] = None

        self._tmp_count = 0
        self._var_dict: dict[str, Var] = {}

        # big_M
        self._M = M

    # -------------
    # add_var

    # fmt: off
    @overload
    def add_var(self, vtype: Literal[VType.BINARY], ub: float = 1, lb: float = 0, name: Optional[str] = None) -> BinVar: ...
    @overload
    def add_var(self, vtype: Literal[VType.INTEGER], ub: Literal[1], lb: Literal[0] = 0, name: Optional[str] = None) -> BinVar: ...
    @overload
    def add_var(self, vtype: Literal[VType.INTEGER], ub: float = _INF, lb: float = 0, name: Optional[str] = None) -> Var: ...
    @overload
    def add_var(self, vtype: VType = VType.CONTINUOUS, ub: float = _INF, lb: float = 0, name: Optional[str] = None) -> Var: ...
    # fmt: on
    def add_var(
        self, vtype: VType = VType.CONTINUOUS, ub: float = _INF, lb: float = 0, name: Optional[str] = None
    ) -> Var:
        name = name or self._gen_name()
        assert name not in self._var_dict, f"Variable with name `{name}` already exists"
        assert lb <= ub, f"Upper bound must be greater than lower bound"

        idx = len(self._vars)
        if vtype == VType.BINARY or (vtype == VType.INTEGER and lb == 0 and ub == 1):
            res = _BinVar(self, idx, name)
        else:
            res = _TypVar(self, idx, name, lb, ub, vtype)
        self._vars.append(res)
        self._var_dict[res._name] = res
        return res

    # -------------
    # add_vars

    # fmt: off
    @overload
    def add_vars(self, n: int, vtype: Literal[VType.BINARY], ub: float = 1, lb: float = 0, name: Optional[str] = None) -> list[BinVar]: ...
    @overload
    def add_vars(self, n: int, vtype: Literal[VType.INTEGER], ub: Literal[1], lb: Literal[0] = 0, name: Optional[str] = None) -> list[BinVar]: ...
    @overload
    def add_vars(self, n: int, vtype: Literal[VType.INTEGER], ub: float = _INF, lb: float = 0, name: Optional[str] = None) -> list[Var]: ...
    @overload
    def add_vars(self, n: int, vtype: VType = VType.CONTINUOUS, ub: float = _INF, lb: float = 0, name: Optional[str] = None) -> list[Var]: ...
    # fmt: on
    def add_vars(
        self, n: int, vtype: VType = VType.CONTINUOUS, ub: float = _INF, lb: float = 0, name: Optional[str] = None
    ) -> Union[list[Var], list[BinVar]]:
        name = name or self._gen_name()
        return [self.add_var(vtype, ub, lb, f"{name}[{i}]") for i in range(n)]

    # ----------------
    # Member functions

    def add_constraint(self, constr: Constr[_ExprT], name: Optional[str] = None, if_: Optional[BinExprOrLiteral] = None):
    # def add_constraint(self, constr: Constr, name: Optional[str] = None, if_: Optional[BinExprOrLiteral] = None):
        constr = check_type(constr, Constr[_ExprT])
        name = name or self._gen_name("C")

        if isinstance(if_, BinExpr):
            expr = constr._lhs - constr._rhs
            if constr._op == RelOp.EQ or constr._op == RelOp.GE:
                constr_ge = constr._lhs >= constr._rhs - self._clip_M(-expr.bound().min) * (1 - if_)
                constr_ge._name = f"{name}/ge"
                self._constrs.append(constr_ge)
            if constr._op == RelOp.EQ or constr._op == RelOp.LE:
                constr_le = constr._lhs <= constr._rhs + self._clip_M(expr.bound().max) * (1 - if_)
                constr_le._name = f"{name}/le"
                self._constrs.append(constr_le)
        elif if_ is None or if_ == 1:
            constr._name = name
            self._constrs.append(constr)

    def set_objective(self, sense: Sense, expr: _ExprT):
        self._obj_sense = sense
        self._obj = expr

    def set_solution(self, solution: Solution):
        if (our := set(self._var_dict.keys())) != (other := set(solution.keys())):
            print(f"Warning: Variable names do not match:\n  missing: {our - other}\n  extra: {other - our}.")
        for k, v in solution.items():
            self._var_dict[k].set_value(v)

    def check_solution_satisfies_constraints(self, solution: Solution, verbose: bool = False) -> bool:
        self.set_solution(solution)
        all_res = True
        for constr in self._constrs:
            try:
                res = constr.satisfied()
                all_res = all_res and res
                if verbose:
                    print(f"[{'OK' if res else 'FAILED'}]: {constr}")
            except RuntimeError as e:
                if verbose:
                    print(f"[SKIPPED]: {constr}")
        return all_res

    def enforce_solution(self, solution: Solution):
        for i, (k, v) in enumerate(solution.items()):
            if k not in self._var_dict:
                print(f"Warning: variable {k} not found")
            else:
                self.add_constraint(self._var_dict[k] == v, name=f"E{i}")

    # ----------------
    # Getters

    def get_vars(self) -> list[Var]:
        return [*self._vars]

    def get_constraint(self) -> list[Constr]:
        return [*self._constrs]

    def get_objective(self) -> _ExprT:
        assert self._obj is not None and self._obj_sense is not None, "No objective set"
        return check_type(self._obj, _ExprT)

    def get_objective_sense(self) -> Sense:
        assert self._obj is not None and self._obj_sense is not None, "No objective set"
        return self._obj_sense

    # ----------------
    # Syntactic Sugar

    @overload
    def sum(self, xs: Iterable[LinExpr]) -> LinExpr:
        ...

    @overload
    def sum(self, xs: Iterable[QuadExpr]) -> QuadExpr:
        ...

    def sum(self, xs: Iterable[Union[LinExpr, QuadExpr]]) -> Union[LinExpr, QuadExpr]:
        return sum(xs, _LinExpr(self, {}, 0))

    @_cache
    def and_(self, *x: BinExprOrLiteral, name: Optional[str] = None) -> BinExpr:
        """Returns a new variable y := and(x_0, ..., x_{n-1}).

        If n is 1, returns x_0.
        """
        assert len(x) > 0, "Requires at least 1 expression."
        name = name or _gen_var_name("and", x)
        if len(x) == 1:
            return self._convert_bin(x[0])

        d = self.add_var(VType.BINARY, name=name)
        for i, var in enumerate(x):
            self.add_constraint(d <= var, f"{name}/C1[{i}]")
        self.add_constraint(d >= self.sum(self._convert_bin(x_) for x_ in x) - (len(x) - 1), f"{name}/C2")
        return d

    @_cache
    def or_(self, *x: BinExprOrLiteral, name: Optional[str] = None) -> BinExpr:
        """Returns a new variable y := or(x_0, ..., x_{n-1})

        If n is 1, returns x_0.
        """
        assert len(x) > 0, "Requires at least 1 expression."
        name = name or _gen_var_name("or", x)
        if len(x) == 1:
            return self._convert_bin(x[0])

        d = self.add_var(VType.BINARY, name=name)
        for i, var in enumerate(x):
            self.add_constraint(d >= var, f"{name}/C1[{i}]")
        self.add_constraint(d <= self.sum(self._convert_bin(x_) for x_ in x), f"{name}/C2")
        return d

    @_cache
    def min_(self, *x: Union[_ExprT, Number], name: Optional[str] = None) -> _MinMaxResult[_ExprT]:
        """Creates a new variable y := min(x_0, ..., x_{n-1}).
        Note: n auxilary binary variables are also created.

        Returns: (y, argmin).

        If n is 1, returns (x_0, [1]) (no variables created).
        """
        assert len(x) > 0, "Requires at least 1 expression."
        name = name or _gen_var_name("min", x)
        if len(x) == 1:
            return _MinMaxResult(self._convert(x[0]), [self._convert_bin(1)])

        xs = tuple(self._convert(x_) for x_ in x)
        n = len(xs)
        L_min = min(x_.bound().min for x_ in xs)
        U_min = min(x_.bound().max for x_ in xs)
        y = self.add_var(lb=L_min, ub=U_min, name=name)
        d = self.add_vars(n, VType.BINARY, name=f"{name}/d")
        for i, x_ in enumerate(xs):
            self.add_constraint(y <= x_, f"{name}/C1[{i}]")
            self.add_constraint(y >= x_ - self._clip_M(x_.bound().max - L_min) * (1 - d[i]), f"{name}/C2[{i}]")
        self.add_constraint(self.sum(d) == 1, f"{name}/C3")
        return check_type(_MinMaxResult(y, d), _MinMaxResult[_ExprT])

    @_cache
    def max_(self, *x: Union[_ExprT, Number], name: Optional[str] = None) -> _MinMaxResult[_ExprT]:
        """Creates a new variable y := max(x_0, ..., x_{n-1}).
        Note: n auxilary binary variables are also created.

        Returns: (y, argmax).

        If n is 1, returns (x_0, [1]) (no variables created).
        """
        assert len(x) > 0, "Requires at least 1 expression."
        name = name or _gen_var_name("max", x)
        if len(x) == 1:
            return _MinMaxResult(self._convert(x[0]), [self._convert_bin(1)])

        xs = tuple(self._convert(x_) for x_ in x)
        n = len(xs)
        L_max = max(x_.bound().min for x_ in xs)
        U_max = max(x_.bound().max for x_ in xs)
        y = self.add_var(lb=L_max, ub=U_max, name=name)
        d = self.add_vars(n, VType.BINARY, name=f"{name}/d")
        for i, x_ in enumerate(xs):
            self.add_constraint(y >= x_, f"{name}/C1[{i}]")
            self.add_constraint(y <= x_ + self._clip_M(U_max - x_.bound().min) * (1 - d[i]), f"{name}/C2[{i}]")
        self.add_constraint(self.sum(d) == 1, f"{name}/C3")
        return check_type(_MinMaxResult(y, d), _MinMaxResult[_ExprT])

    @_cache
    def abs_(self, x: Union[_ExprT, Number], name: Optional[str] = None) -> _ExprT:
        """Returns a new variable y := abs(x).
        Note: two auxilary binary variables are also created."""
        x_ = self._convert(x)
        name = name or f"abs({x_.compact_repr()})"
        U = max(0, x_.bound().min, -x_.bound().max)
        y = self.add_var(lb=0, ub=U, name=name)
        d = self.add_var(VType.BINARY, name=f"{name}/d")

        self.add_constraint(y >= x_, f"{name}/C1[0]")
        self.add_constraint(y >= -x_, f"{name}/C1[1]")

        self.add_constraint(y <= x_ + self._clip_M(U - x_.bound().min) * ~d, f"{name}/C2[0]")
        self.add_constraint(y <= -x_ + self._clip_M(U - (-x_).bound().max) * d, f"{name}/C2[1]")

        return check_type(y, _ExprT)

    @_cache
    def mux_(
        self,
        *cond_expr: tuple[BinExprOrLiteral, Union[_ExprT, Number]],
        name: Optional[str] = None,
        else_: Optional[Union[_ExprT, Number]] = None,
    ) -> _ExprT:
        """Creates a new variable y := expr_0 if cond_0; ...; expr_{n-1} if cond_{n-1}.
        Assumes exactly one cond_i is true or at most one cond_i if else_ is specified.

        if n is 1, returns expr_0 (no variables created).
        """
        assert len(cond_expr) > 0, "Requires at least one expression."
        cond_expr = tuple((self._convert_bin(cond), self._convert(expr)) for cond, expr in cond_expr)
        if else_ is not None:
            cond_expr = cond_expr + (
                ((1 - self.sum(cond for cond, _ in cond_expr)).to_bin_expr(), self._convert(else_)),
            )
        if len(cond_expr) == 1:
            return cond_expr[0][1]

        if name is None:
            name = (
                "mux(" + ",".join(f"{cond.compact_repr()}->({expr.compact_repr()})" for cond, expr in cond_expr) + ")"
            )

        L_min = min(x_.bound().min for _, x_ in cond_expr)
        U_max = max(x_.bound().max for _, x_ in cond_expr)
        y = self.add_var(ub=U_max, lb=L_min, name=name)
        for i, (cond, expr) in enumerate(cond_expr):
            self.add_constraint(y <= expr + self._clip_M(U_max - expr.bound().min) * (1 - cond), f"{name}/C1[{i}]")
            self.add_constraint(y >= expr - self._clip_M(expr.bound().max - L_min) * (1 - cond), f"{name}/C2[{i}]")
        return check_type(y, _ExprT)

    @_cache
    def bmux_(
        self,
        *cond_expr: tuple[BinExprOrLiteral, BinExprOrLiteral],
        name: Optional[str] = None,
        else_: Optional[BinExprOrLiteral] = None,
    ) -> BinExpr:
        """Creates a new variable y := expr_0 if cond_0; ...; expr_{n-1} if cond_{n-1}.
        Assumes exactly one cond_i is true.

        if n is 1, returns expr_0 (no variables created).
        """
        assert len(cond_expr) > 0, "Requires at least one expression."
        cond_expr = tuple((self._convert_bin(cond), self._convert_bin(expr)) for cond, expr in cond_expr)
        if else_ is not None:
            else_cond = ~(self.sum(cond for cond, _ in cond_expr)).to_bin_expr()
            cond_expr = (*cond_expr, (else_cond, self._convert_bin(else_)))
        if len(cond_expr) == 1:
            return cond_expr[0][1]

        if name is None:
            name = (
                "bmux(" + ",".join(f"{cond.compact_repr()}->({expr.compact_repr()})" for cond, expr in cond_expr) + ")"
            )

        y = self.add_var(VType.BINARY, name=name)
        for i, (cond, expr) in enumerate(cond_expr):
            self.add_constraint(y <= expr + (1 - cond), f"{name}/C1[{i}]")
            self.add_constraint(y >= expr - (1 - cond), f"{name}/C2[{i}]")
        return y

    def muxT_(
        self, cond: Sequence[BinExprOrLiteral], expr: Sequence[Union[_ExprT, Number]], name: Optional[str] = None
    ) -> _ExprT:
        """Creates a new variable y := expr_0 if cond_0; ...; expr_{n-1} if cond_{n-1}.

        Assumes exactly one cond_i is true.
        """
        assert len(cond) == len(expr), f"Length of cond and expr must match. Got {len(cond)} and {len(expr)}"
        return self.mux_(*zip(cond, expr), name=name)

    def bmuxT_(
        self, cond: Sequence[BinExprOrLiteral], expr: Sequence[BinExprOrLiteral], name: Optional[str] = None
    ) -> BinExpr:
        """Creates a new variable y := expr_0 if cond_0; ...; expr_{n-1} if cond_{n-1}.

        Assumes exactly one cond_i is true.
        """
        assert len(cond) == len(expr), f"Length of cond and expr must match. Got {len(cond)} and {len(expr)}"
        return self.bmux_(*zip(cond, expr), name=name)

    @_cache
    def min_en_(
        self, *en_x: tuple[BinExprOrLiteral, Union[_ExprT, Number]], name: Optional[str] = None
    ) -> _MinMaxResult[_ExprT]:
        """Creates a new variable y := min({x_i if en_i}).
        Assumes at most en_i is true.
        Note: n + 1 auxilary binary variables are also created.

        Returns: (y, argmin).
        If all en_i are false, Model._M is returned.
        """
        en_x = tuple((self._convert_bin(en), self._convert(x)) for en, x in en_x)
        n = len(en_x)

        assert n > 0, "Requires at least 1 expression."
        if name is None:
            name = "min_en(" + ",".join(f"{en.compact_repr()}->({x.compact_repr()})" for en, x in en_x) + ")"

        L_min = min(x.bound().min for _, x in en_x)
        y = self.add_var(lb=L_min, name=name)
        d = self.add_vars(n + 1, VType.BINARY, name=f"{name}/d")
        for i, (en, x) in enumerate(en_x):
            self.add_constraint(y <= x + self._M * (1 - en), f"{name}/C1[{i}]")
            self.add_constraint(y >= x - self._clip_M(x.bound().max - L_min) * (1 - d[i]), f"{name}/C2[{i}]")
            self.add_constraint(d[i] <= en, f"{name}/C3[{i}]")
        self.add_constraint(self.sum(d) == 1, f"{name}/C4")
        self.add_constraint(y == self._M, if_=d[n], name=f"{name}/C5")
        return check_type(_MinMaxResult(y, d), _MinMaxResult[_ExprT])

    def min_enT_(
        self, en: Sequence[BinExprOrLiteral], x: Sequence[Union[_ExprT, Number]], name: Optional[str] = None
    ) -> _MinMaxResult[_ExprT]:
        """Creates a new variable y := min({x_i if en_i}).
        Assumes at most en_i is true.
        Note: n + 1 auxilary binary variables are also created.

        Returns: (y, argmin).
        If all en_i are false, Model._M is returned.
        """
        assert len(en) == len(x), f"Length of en and x must match. Got {len(en)} and {len(x)}"
        return self.min_en_(*zip(en, x), name=name)

    # ------------
    # Utils

    def __repr__(self) -> str:
        res = f"Model `{self._name}`:\n"
        if self._obj_sense is not None:
            res += f"  {self._obj_sense.name}:\n"
            res += f"    {self._obj}\n"
        if len(self._vars) > 0:
            res += "  Variables:\n"
            for var in self._vars:
                res += f"    {var.name()}: {var.type().name} ({var.bound().min}, {var.bound().max})\n"
        if len(self._constrs) > 0:
            res += "  Constraints:\n"
            for constr in self._constrs:
                res += f"    {constr}\n"
        return res

    def _gen_name(self, pref: str = "T") -> str:
        res = f"{pref}{self._tmp_count}"
        self._tmp_count += 1
        return res

    def _clip_M(self, cand_M: float) -> float:
        cand_M = max(cand_M, 0)
        return self._M if math.isinf(cand_M) else cand_M

    def _convert(self, expr: Union[_ExprT, Number]) -> _ExprT:
        if isinstance(expr, (float, int)):
            return check_type(_LinExpr(self, {}, expr), _ExprT)
        else:
            assert self == expr.model(), "Expressions must come from the same model"
            return expr

    def _convert_bin(self, x: BinExprOrLiteral) -> BinExpr:
        """
        Convert x to IBinExpr
        """
        if isinstance(x, BinExpr):
            assert self == x.model(), "Expressions must be from the same model"
            return x
        elif x == 0 or x == 1:
            return _BinLiteral(self, x)
        else:
            raise TypeError(f"{type(x).__name__} cannot be converted to IBinExpr")


# --------------
# Utils

_T = TypeVar("_T")


def _add_expr(a: dict[_T, float], b: dict[_T, float]) -> dict[_T, float]:
    """returns: a + b"""
    if a == {}:
        return b
    if b == {}:
        return a
    res = {**a}
    for var, coeff in b.items():
        res[var] = res.get(var, 0) + coeff
    return res


def _mul_expr_const(a: dict[_T, float], c: Union[float, int]) -> dict[_T, float]:
    """returns a * c"""
    if a == {}:
        return a
    if abs(c - 1) < _EPS:
        return a
    return {k: v * c for k, v in a.items()}


class Bound(NamedTuple):
    min: Number
    max: Number

    def __add__(self, other: Union["Bound", Number]) -> "Bound":
        if isinstance(other, "Bound"):
            return Bound(self.min + other.min, self.max + other.max)
        else:
            return Bound(self.min + other, self.max + other)

    def __mul__(self, other: Union["Bound", Number]) -> "Bound":
        if isinstance(other, Bound):
            return _minmax(self.min * other.min, self.max * other.min, self.min * other.max, self.max * other.max)
        else:
            return _minmax(self.min * other, self.max * other)


def _minmax(arg: Number, *args: Number) -> Bound:
    if len(args) == 0:
        return Bound(arg, arg)
    return Bound(min(arg, *args), max(arg, *args))


if __name__ == "__main__":
    model = Model("Hello")
    var = _TypVar(model, 0, "", 0, 1, VType.BINARY)
    var = _BinVar(model, 0, "")
