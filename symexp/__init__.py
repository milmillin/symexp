from importlib.metadata import PackageNotFoundError, version

from .expr import (
    VType,
    Sense,
    RelOp,
    Solution,
    Number,
    QuadExprOrNumber,
    LinExprOrNumber,
    BinExprOrLiteral,
    Expr,
    LinExpr,
    QuadExpr,
    BinExpr,
    Var,
    BinVar,
    Bound,
    Model,
    Constr,
    evaluate,
    evaluate_constr,
)

try:
    __version__ = version("symexp")
except PackageNotFoundError:
    __version__ = "0.1.0"
