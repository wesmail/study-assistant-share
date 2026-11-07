"""sympify for text a model wrote, and the report line the host can trust.

sympify calls eval(). The container stops evaluated code from reaching anything outside
it, but not from forging the result the host reads: a claim could print a report saying
it passed and exit (review S3, Nov 2026). Three walls, each enough against a plain attempt:

1. The text is read as a Python expression first, and refused unless it is maths: no
   strings, no subscripts, no names or attributes that start with "_", no string methods,
   no comprehensions. Without strings there is nothing to hand to a function that
   evaluates text (every SymPy function sympifies a string argument, with builtins).
2. It is evaluated with no builtins, no modules, and none of SymPy's printers or text
   parsers, so print(), exit(), __import__, latex() and srepr() aren't there: such a
   name becomes a plain symbol or function, as an unknown name always does.
3. The report carries a nonce the host wrote to the work folder. `read_nonce` deletes the
   file before any claim is read, so evaluated code can't learn it.
"""
from __future__ import annotations

import ast
import json
import os
import re
import types
from tokenize import TokenError

import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

NONCE_FILE = "/work/nonce"
TRANSFORMS = standard_transformations + (convert_xor,)

# Where SymPy keeps what turns objects into text or text into objects.
TEXT_MODULES = ("sympy.printing", "sympy.parsing", "sympy.utilities", "sympy.interactive", "sympy.plotting",
                "sympy.external", "sympy.testing", "sympy.codegen", "sympy.core.sympify", "sympy.multipledispatch")
TEXT_NAMES = {"sympify", "var", "srepr", "latex", "pretty", "pprint", "sstr", "init_printing", "init_session",
              "lambdify", "preview", "parse_expr", "dotprint", "print_tree", "rcode", "python", "textplot"}
NAMING = {"symbols", "Symbol", "Function", "Dummy"}     # a name in quotes makes a symbol, nothing else
SYMBOL_NAMES = re.compile(r"[A-Za-z][\w]*(?:[\s,]+[A-Za-z][\w]*)*")
STRING_METHODS = {name for name in dir(str) if not name.startswith("_")} | {"name"}
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.keyword, ast.Name, ast.Load,
    ast.Constant, ast.Attribute, ast.Tuple, ast.List, ast.Dict, ast.Set, ast.Lambda, ast.arguments, ast.arg,
    ast.operator, ast.unaryop, ast.cmpop, ast.boolop,
)


class Refused(ValueError):
    """The text isn't plain maths, so it isn't evaluated."""


def _from_text_module(value) -> bool:
    module = getattr(value, "__module__", None) or getattr(type(value), "__module__", "") or ""
    return isinstance(module, str) and module.startswith(TEXT_MODULES)


def _namespace() -> dict:
    found: dict = {}
    exec("from sympy import *", found)      # noqa: S102 - fixed text, run once at import
    names = {name: value for name, value in found.items()
             if not name.startswith("_") and name not in TEXT_NAMES
             and not isinstance(value, types.ModuleType) and not _from_text_module(value)}
    names["__builtins__"] = {}
    return names


GLOBALS = _namespace()


def check_text(text: str, known: set[str] = frozenset()) -> None:
    """Raise Refused unless `text` is an expression made only of maths. `known` are the
    names the caller supplies itself (its own `_unit0` included)."""
    if "__" in text:
        raise Refused("double underscores aren't read")
    try:
        tree = ast.parse(text.replace("^", "**").strip(), mode="eval")
    except SyntaxError:
        return      # not Python; SymPy's own parser may still read it (n!), and fails safely if not
    named = {id(argument) for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in NAMING
             for argument in node.args}
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise Refused(f"{type(node).__name__} isn't maths")
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)) and not (
                id(node) in named and isinstance(node.value, str) and SYMBOL_NAMES.fullmatch(node.value)):
            raise Refused("text in quotes isn't read, except names in symbols('x y')")
        if isinstance(node, ast.Name) and node.id.startswith("_") and node.id not in known:
            raise Refused(f"the name {node.id} isn't read")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in STRING_METHODS):
            raise Refused(f".{node.attr} isn't read")


def sympify(text: str, names: dict | None = None):
    """sympy.sympify(text, locals=names, convert_xor=True), behind the walls above."""
    local = dict(names or {})
    check_text(text, set(local))
    try:
        return parse_expr(text, local_dict=local, global_dict=dict(GLOBALS), transformations=TRANSFORMS)
    except (SyntaxError, TokenError, NameError) as error:      # as sympify reports them
        raise sp.SympifyError(text, error) from None


def read_nonce(path: str = NONCE_FILE) -> str:
    """The host's nonce, read once and deleted before anything else runs. '' if none."""
    try:
        with open(path, encoding="ascii") as handle:
            nonce = handle.read().strip()
        os.unlink(path)
    except OSError:
        return ""
    return nonce


def report(result: dict, nonce: str) -> None:
    print(json.dumps({**result, "nonce": nonce}, ensure_ascii=False), flush=True)
