from pathlib import Path
import sympy as sp
import importlib.util
from types import SimpleNamespace

# Load the symbolic module directly from src to avoid package import issues in tests
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SYMBOLIC_PATH = SRC / "debbirth" / "models" / "gp" / "symbolic.py"
spec = importlib.util.spec_from_file_location("debbirth.models.gp.symbolic", str(SYMBOLIC_PATH))
symbolic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(symbolic)  # type: ignore

program_str_to_sympy = symbolic.program_str_to_sympy
substitute_feature_names = symbolic.substitute_feature_names
simplify_program = symbolic.simplify_program
program_to_srepr = symbolic.program_to_srepr
model_program_to_sympy_strings = symbolic.model_program_to_sympy_strings


def test_parse_and_substitute_binary_ops():
    expr = program_str_to_sympy("add(X0, mul(X1, X2))")
    named = substitute_feature_names(expr, ["a", "b", "c"], constants_map=None)
    simplified = sp.simplify(named)
    assert str(simplified) == "a + b*c"


def test_numeric_literal_and_simplify():
    out = simplify_program("mul(X0, 2)", ["x"], constants=None)
    # Use sympy to compare algebraically
    assert sp.simplify(sp.sympify(out) - sp.sympify("2*x")) == 0


def test_constants_substitution_via_model_helper():
    class DummyModel:
        _program = "add(X0, X1)"
        feature_names = ("a", "c1")
        feature_names_no_constants = ("a",)
        constants = (SimpleNamespace(name="c1", value=1.0),)

    res = model_program_to_sympy_strings(DummyModel)
    assert res is not None
    expr = sp.sympify(res["simplified"])
    expected = sp.sympify("a + 1")
    assert sp.simplify(expr - expected) == 0


def test_program_to_srepr_roundtrip():
    srepr = program_to_srepr("add(X0, X1)")
    assert "Symbol('__X0__')" in srepr and "Symbol('__X1__')" in srepr
