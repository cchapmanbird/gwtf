#!/usr/bin/env python
"""
Translate the scalar post-Newtonian expression functions of the legacy (numba)
models into C++ ``GWTF_HD`` (host + device) functions.

The PN series are hundreds of lines of literal coefficients, so hand-porting
them is error-prone. This tool walks the Python AST and emits fully
parenthesised C++ that preserves Python's evaluation order:

* every numeric literal is emitted as a double, so ``743 / 252`` stays true
  division;
* ``x ** n`` with integer ``n`` becomes ``gwtf::ipow<n>(x)`` (multiplications),
  half-integer ``n`` becomes ``ipow<k>(x) * sqrt(x)``;
* ``x ** (k/8)`` or ``x ** (k/4)`` with ``x`` a function argument becomes
  ``ipow<k>(x_r8)`` with ``x_r8 = sqrt(sqrt(sqrt(x)))`` computed once -- the
  TaylorT3 series evaluate a dozen such powers of tau, and double-precision
  ``pow`` dominates the model cost on GPUs (a few ulp, well below the
  phase accuracy of the PN series);
* anything else becomes ``pow``;
* ``pi`` and ``gamma_e`` map to ``gwtf::constants``.

Only straight-line functions made of assignments, ``if``/``else`` and a single
``return`` of a scalar are supported -- which covers every PN helper used by
the built-in models. The generated headers are checked in; re-run this script
if the Python expressions change::

    python tools/translate_pn.py

and validate with ``pytest tests/test_models.py`` (which compares the C++
model tracks against the legacy numba functions).
"""

import ast
import fractions
import inspect
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEGACY_MODELS = REPO / "src" / "pygwtf" / "legacy" / "models"
OUT_DIR = REPO / "src" / "cpp" / "include" / "gwtf" / "models"

NAME_MAP = {
    "pi": "gwtf::constants::pi",
    "gamma_e": "gwtf::constants::gamma_e",
}
FUNC_MAP = {"log": "log", "sqrt": "sqrt", "abs": "fabs", "exp": "exp"}


class TranslationError(RuntimeError):
    pass


def _const_value(node):
    """Evaluate a constant-only expression, or return None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return fractions.Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.USub, ast.UAdd)
    ):
        v = _const_value(node.operand)
        if v is None:
            return None
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        a, b = _const_value(node.left), _const_value(node.right)
        if a is None or b is None:
            return None
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        return a / b
    return None


def _literal(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value}.0"
    r = repr(float(value))
    if "e" not in r and "." not in r and "inf" not in r:
        r += ".0"
    return r


class ExprTranslator:
    def __init__(self, known_functions, bool_names, params=()):
        self.known_functions = known_functions
        self.bool_names = bool_names
        self.params = set(params)
        self.eighth_roots = []  # arguments needing x_r8 = x^(1/8)

    def expr(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (bool, int, float)):
                return _literal(node.value)
            raise TranslationError(f"unsupported constant {node.value!r}")
        if isinstance(node, ast.Name):
            return NAME_MAP.get(node.id, node.id)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return f"(-{self.expr(node.operand)})"
            if isinstance(node.op, ast.UAdd):
                return self.expr(node.operand)
            if isinstance(node.op, ast.Not):
                return f"(!{self.expr(node.operand)})"
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                return self.power(node)
            ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
            op = ops.get(type(node.op))
            if op is None:
                raise TranslationError(f"unsupported operator {node.op}")
            return f"({self.expr(node.left)} {op} {self.expr(node.right)})"
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise TranslationError(
                    "only simple positional calls supported"
                )
            name = node.func.id
            args = ", ".join(self.expr(a) for a in node.args)
            if name in FUNC_MAP:
                return f"{FUNC_MAP[name]}({args})"
            if name in self.known_functions:
                return f"{name}({args})"
            raise TranslationError(f"call to unknown function {name}")
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise TranslationError("chained comparisons unsupported")
            ops = {
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
                ast.Eq: "==",
                ast.NotEq: "!=",
            }
            return (
                f"({self.expr(node.left)} {ops[type(node.ops[0])]} "
                f"{self.expr(node.comparators[0])})"
            )
        raise TranslationError(f"unsupported expression {ast.dump(node)}")

    def power(self, node):
        base = self.expr(node.left)
        value = _const_value(node.right)
        if value is None:
            return f"pow({base}, {self.expr(node.right)})"
        if value.denominator == 1:
            n = int(value)
            if n == 0:
                return "1.0"
            if n > 0:
                return f"gwtf::ipow<{n}>({base})"
            return f"(1.0 / gwtf::ipow<{-n}>({base}))"
        if (
            value.denominator in (4, 8)
            and isinstance(node.left, ast.Name)
            and node.left.id in self.params
        ):
            name = node.left.id
            if name not in self.eighth_roots:
                self.eighth_roots.append(name)
            k = int(value * 8)
            term = f"gwtf::ipow<{abs(k)}>({name}_r8)"
            return term if k > 0 else f"(1.0 / {term})"
        if value.denominator == 2:
            k = abs(value.numerator) // 2  # floor(|n|)
            term = (
                f"(gwtf::ipow<{k}>({base}) * sqrt({base}))"
                if k > 0
                else f"sqrt({base})"
            )
            return term if value > 0 else f"(1.0 / {term})"
        return f"pow({base}, {self.expr(node.right)})"


def translate_function(func, known_functions):
    src = textwrap.dedent(inspect.getsource(func.py_func))
    fdef = ast.parse(src).body[0]
    params = [a.arg for a in fdef.args.args]
    bool_names = {p for p in params if p.startswith("include_")}
    tr = ExprTranslator(known_functions, bool_names, params)
    declared = set(params)
    lines = []

    def stmt(node, indent):
        pad = "    " * indent
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            return  # docstring
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(
                node.targets[0], ast.Name
            ):
                raise TranslationError(
                    "only single-name assignments supported"
                )
            name = node.targets[0].id
            rhs = tr.expr(node.value)
            if name in declared:
                lines.append(f"{pad}{name} = {rhs};")
            else:
                lines.append(f"{pad}double {name} = {rhs};")
                declared.add(name)
            return
        if isinstance(node, ast.AugAssign):
            ops = {ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*=", ast.Div: "/="}
            lines.append(
                f"{pad}{node.target.id} {ops[type(node.op)]} {tr.expr(node.value)};"
            )
            return
        if isinstance(node, ast.Return):
            lines.append(f"{pad}return {tr.expr(node.value)};")
            return
        if isinstance(node, ast.If):
            # Hoist declarations assigned in either branch so they outlive the if.
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and isinstance(
                    sub.targets[0], ast.Name
                ):
                    n = sub.targets[0].id
                    if n not in declared:
                        lines.append(f"{pad}double {n} = 0.0;")
                        declared.add(n)
            lines.append(f"{pad}if ({tr.expr(node.test)}) {{")
            for s in node.body:
                stmt(s, indent + 1)
            if node.orelse:
                lines.append(f"{pad}}} else {{")
                for s in node.orelse:
                    stmt(s, indent + 1)
            lines.append(f"{pad}}}")
            return
        raise TranslationError(f"unsupported statement {ast.dump(node)[:80]}")

    for s in fdef.body:
        stmt(s, 1)

    roots = [
        f"    const double {n}_r8 = sqrt(sqrt(sqrt({n})));"
        for n in tr.eighth_roots
    ]
    sig = ", ".join(
        ("bool " if p in bool_names else "double ") + p for p in params
    )
    return (
        f"GWTF_HD inline double {fdef.name}({sig})\n{{\n"
        + "\n".join(roots + lines)
        + "\n}\n"
    )


HEADER = """\
// AUTO-GENERATED by tools/translate_pn.py from
//   {source}
// Do not edit by hand: change the Python expressions and re-run the tool.
#pragma once

#include "gwtf/math.hpp"

namespace gwtf {{
namespace pn {{
namespace {namespace} {{

using std::log;
using std::pow;
using std::sqrt;
using std::fabs;
using std::exp;

{body}
}} // namespace {namespace}
}} // namespace pn
}} // namespace gwtf
"""


def generate(module, function_names, namespace, out_name, source_rel):
    known = set(function_names)
    body = "\n".join(
        translate_function(getattr(module, name), known)
        for name in function_names
    )
    out = OUT_DIR / out_name
    out.write_text(
        HEADER.format(source=source_rel, namespace=namespace, body=body)
    )
    print(f"wrote {out.relative_to(REPO)} ({len(function_names)} functions)")


def main():
    from pygwtf.legacy.models.taylort2ecc import common as t2
    from pygwtf.legacy.models.taylort3spin import common as t3

    generate(
        t2,
        [
            "_calculate_T",
            "_Lambda_f_vec",
            "_Xi_ecc_term",
            "_Xi",
            "_dXi_dtheta",
        ],
        "taylort2ecc",
        "taylort2ecc_pn.hpp",
        "src/pygwtf/legacy/models/taylort2ecc/common.py",
    )
    generate(
        t3,
        [
            "phase",
            "frequency",
            "frequency_derivative",
            "time_to_merger",
            "tau_to_x",
        ],
        "taylort3spin",
        "taylort3spin_pn.hpp",
        "src/pygwtf/legacy/models/taylort3spin/common.py",
    )


if __name__ == "__main__":
    main()
