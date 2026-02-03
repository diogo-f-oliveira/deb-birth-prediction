# Add SymPy-based conversion utilities for gplearn programs.
# This module provides functions to parse a gplearn Program string (e.g. "add(X0, X1)")
# into a sympy expression, simplify it, and export its srepr.

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re
from pathlib import Path

import sympy as sp

# New: import GP constants and build a default constants map (include all named constants)
from .constants import EXTENDED_CONSTANT_SET

ALL_CONSTANTS_MAP = {
    c.name: float(c.value) for c in EXTENDED_CONSTANT_SET
}

# Mapping from gplearn primitive names to functions that produce SymPy expressions.
GP_TO_SYMPY: Dict[str, Any] = {
    # binary
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "pdiv": lambda a, b: a / b,
    # unary
    "neg": lambda a: -a,
    "inv": lambda a: sp.Integer(1) / a,
    "pinv": lambda a: sp.Integer(1) / a,
    "sqrt": lambda a: sp.sqrt(a),
    "log": lambda a: sp.log(a),
    "plog": lambda a: sp.log(a),
    "cbrt": lambda a: sp.root(a, 3),
    "square": lambda a: a ** 2,
    "cube": lambda a: a ** 3,
    "atan": lambda a: sp.atan(a),
    # Add min/max mappings (sympy.Min/Max) so GP programs using min/max are handled
    "min": lambda a, b: sp.Min(a, b),
    "max": lambda a, b: sp.Max(a, b),
}

_token_re = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*|X\d+|[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?|\(|\)|,)\s*")


class ParseError(ValueError):
    pass


def tokenize(program: str) -> List[str]:
    pos = 0
    tokens: List[str] = []
    while pos < len(program):
        m = _token_re.match(program, pos)
        if not m:
            raise ParseError(f"Unexpected token at position {pos}: '{program[pos:]}'")
        tok = m.group(1)
        tokens.append(tok)
        pos = m.end()
    return tokens


def _parse_tokens(tokens: List[str], i: int = 0) -> Tuple[sp.Expr, int]:
    """Recursive parser that consumes tokens starting at index i and returns (expr, next_index).

    Grammar (simplified):
      expr := func_call | terminal
      func_call := NAME '(' args ')'
      args := expr (',' expr)*
      terminal := XNUMBER | NAME | NUMBER
    """
    if i >= len(tokens):
        raise ParseError("Unexpected end of tokens")

    tok = tokens[i]

    # Function call or terminal name/variable/constant/number
    # If next token is '(', it's a call
    if i + 1 < len(tokens) and tokens[i + 1] == '(':
        name = tok
        # consume name and '('
        i += 2
        args: List[sp.Expr] = []
        # handle empty arg list (unlikely in gplearn) but be robust
        if i < len(tokens) and tokens[i] == ')':
            i += 1
        else:
            while True:
                expr, i = _parse_tokens(tokens, i)
                args.append(expr)
                if i >= len(tokens):
                    raise ParseError("Unclosed '(' in expression")
                if tokens[i] == ',':
                    i += 1
                    continue
                if tokens[i] == ')':
                    i += 1
                    break
                raise ParseError(f"Expected ',' or ')' but got '{tokens[i]}'")

        # Build sympy expression from function name
        name_lower = name.lower()
        fn = GP_TO_SYMPY.get(name_lower)
        if fn is None:
            # Unknown function: try to call sympy function by same name if exists
            if hasattr(sp, name_lower):
                sym_fn = getattr(sp, name_lower)
                try:
                    return sym_fn(*args), i
                except Exception as ex:
                    raise ParseError(f"Failed to apply sympy.{name_lower} to args: {ex}")
            raise ParseError(f"Unknown function '{name}' in program")

        # Apply mapping; allow variable arity if mapping supports it
        try:
            expr = fn(*args)
        except TypeError as ex:
            raise ParseError(f"Invalid arity for function '{name}': {ex}")
        return expr, i

    # Terminal (X#, NAME, or number)
    # X## -> feature index
    if tok.startswith('X') and tok[1:].isdigit():
        idx = int(tok[1:])
        # We represent feature terminals as a placeholder Symbol: Feature(i)
        # The caller will substitute real names using the provided mapping.
        return sp.Symbol(f"__X{idx}__"), i + 1

    # Numeric literal
    if re.fullmatch(r"[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?", tok):
        # Convert to rational if possible or Float
        try:
            if '.' in tok or 'e' in tok or 'E' in tok:
                return sp.nsimplify(float(tok)), i + 1
            else:
                return sp.Integer(int(tok)), i + 1
        except Exception:
            return sp.Float(tok), i + 1

    # Name (could be a constant terminal like c1 or bare symbol)
    name = tok
    return sp.Symbol(name), i + 1


def program_str_to_sympy(program_str: str) -> sp.Expr:
    """Parse a gplearn Program string into a SymPy expression where feature terminals are
    represented by placeholder Symbols '__X{idx}__'.

    This function does not attempt to map placeholders to human-readable feature names.
    Use `substitute_feature_names(expr, feature_names, constants_map)` to replace them.
    """
    tokens = tokenize(program_str)
    expr, next_i = _parse_tokens(tokens, 0)
    if next_i != len(tokens):
        raise ParseError(f"Extra tokens after parsing: {tokens[next_i:]}")
    return expr


def substitute_feature_names(
        expr: sp.Expr,
        feature_names: Iterable[str],
        feature_names_no_constants: Optional[Iterable[str]] = None,
        constants_map: Optional[Dict[str, float]] = None,
) -> sp.Expr:
    """Replace placeholders __X{idx}__ with Symbols for base features and numeric constants.

    Assumes `feature_names` is the full feature list (base features first, then constants).
    If `feature_names_no_constants` is given, its length indicates number of base features.
    """
    # Use provided constants_map or fall back to the module-wide ALL_CONSTANTS_MAP
    if constants_map is None:
        constants_map = dict(ALL_CONSTANTS_MAP)

    feats = list(feature_names)
    bcount = len(list(feature_names_no_constants)) if feature_names_no_constants is not None else None

    def _repl(sym: sp.Symbol) -> sp.Expr:
        name = str(sym)
        m = re.fullmatch(r"__X(\d+)__", name)
        if not m:
            # Non-placeholder: replace named constant if present, otherwise keep symbol
            if name in constants_map:
                return sp.nsimplify(constants_map[name])
            return sp.Symbol(name)

        idx = int(m.group(1))
        # If we have a base-count, map indices < bcount to base feature names, and >= bcount to constants
        if bcount is not None:
            if idx < bcount:
                return sp.Symbol(feats[idx]) if idx < len(feats) else sp.Symbol(name)
            else:
                # constant terminal
                if idx < len(feats):
                    terminal_name = feats[idx]
                    if terminal_name in constants_map:
                        return sp.nsimplify(constants_map[terminal_name])
                    return sp.Symbol(terminal_name)
                return sp.Symbol(name)

        # Fallback: map directly using feature_names list
        if idx < 0 or idx >= len(feats):
            return sp.Symbol(name)
        terminal_name = feats[idx]
        if terminal_name in constants_map:
            return sp.nsimplify(constants_map[terminal_name])
        return sp.Symbol(terminal_name)

    return expr.xreplace({s: _repl(s) for s in expr.free_symbols})


def simplify_program(
        program_str: str,
        feature_names: Iterable[str],
        feature_names_no_constants: Optional[Iterable[str]] = None,
        constants: Optional[Dict[str, float]] = None,
) -> str:
    """Parse program_str, substitute feature and constant names/values and return a simplified string.

    Returns the simplified expression as a string.
    """
    expr = program_str_to_sympy(program_str)
    # substitute placeholders with feature symbols / constants; rely on provided feature lists
    sym_expr = substitute_feature_names(expr, feature_names, feature_names_no_constants, constants)
    simplified = sp.simplify(sym_expr)
    return str(simplified)


def program_to_srepr(program_str: str) -> str:
    """Return sympy.srepr of parsed program (placeholders will remain)."""
    expr = program_str_to_sympy(program_str)
    return sp.srepr(expr)


def model_program_to_sympy_strings(model: Any) -> Optional[Dict[str, str]]:
    """Return simplified expression string and srepr for a model's program.

    Assumes model.feature_names and model.feature_names_no_constants are present.
    """
    if not hasattr(model, "_program") or getattr(model, "_program") is None:
        return None

    program_str = str(model._program)

    # Build constants mapping from model.constants if available
    constants_map: Dict[str, float] = {}
    if hasattr(model, "constants") and model.constants is not None:
        try:
            for c in model.constants:
                constants_map[c.name] = float(c.value)
        except Exception:
            constants_map = {}

    feature_names_full = list(model.feature_names)
    feature_names_no_consts = list(model.feature_names_no_constants)

    simplified = simplify_program(program_str, feature_names_full, feature_names_no_consts, constants_map)
    srepr = program_to_srepr(program_str)

    return {"simplified": simplified, "srepr": srepr}


def model_program_to_sympy_expr(model: Any) -> Optional[sp.Expr]:
    """Return a SymPy expression for the model's program with feature names/constants substituted.

    Returns None if the model has no program.
    """
    if not hasattr(model, "_program") or getattr(model, "_program") is None:
        return None

    program_str = str(model._program)

    # Build constants mapping from model.constants if available
    constants_map: Dict[str, float] = {}
    if hasattr(model, "constants") and model.constants is not None:
        try:
            for c in model.constants:
                constants_map[c.name] = float(c.value)
        except Exception:
            constants_map = {}

    feature_names_full = list(model.feature_names)
    feature_names_no_consts = list(model.feature_names_no_constants)

    expr = program_str_to_sympy(program_str)
    expr_named = substitute_feature_names(expr, feature_names_full, feature_names_no_consts, constants_map)
    return expr_named


def save_program_srepr(program_str: str, path: str) -> None:
    """Write sympy.srepr of the given program string to `path` (text file)."""
    s = program_to_srepr(program_str)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


if __name__ == "__main__":
    # Test with saved GP run program
    # run_name = "2026-01-23T16-22-16_DEBBirthSymbolicClassifier"
    # gp_run_dir = Path("results/runs") / run_name /
    # program_file = gp_run_dir / "model" / "best_program.txt"

    # Test with best GP run
    gp_run_dir = Path("results") / "models" / "DEBBirthGP"
    program_file = gp_run_dir / "model" / "best_program.txt"

    program_str = program_file.read_text(encoding="utf-8").strip()
    print("Original program:")
    print(program_str)

    # Load GP config
    cfg_filename = gp_run_dir / "train_gp_config.json"
    cfg = json.loads(cfg_filename.read_text(encoding="utf-8"))
    from ...data.schema import DatasetSpec

    data_spec = DatasetSpec(feature_set=cfg["data_spec"]["feature_set"])
    feature_names = data_spec.feature_cols
    constants_map = {c["name"]: c["value"] for c in cfg["gp"]["constants"]}

    # Simplify the program
    simplified = simplify_program(
        program_str,
        feature_names,
        feature_names_no_constants=feature_names,
        constants=constants_map,
    )
    print("\nSimplified program:")
    print(simplified)
