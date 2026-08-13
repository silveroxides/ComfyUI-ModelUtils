import ast
from pathlib import Path


NODE_DIR = Path(__file__).resolve().parents[1] / "nodes"


def test_every_schema_input_has_a_direct_static_tooltip():
    missing = []
    non_static = []

    for path in sorted(NODE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for call in ast.walk(tree):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "Input"
            ):
                continue

            name = (
                call.args[0].value
                if call.args and isinstance(call.args[0], ast.Constant)
                else "<dynamic input name>"
            )
            tooltip = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "tooltip"),
                None,
            )
            location = f"{path.name}:{call.lineno}:{name}"
            if tooltip is None:
                missing.append(location)
            elif not (
                isinstance(tooltip, ast.Constant)
                and isinstance(tooltip.value, str)
                and tooltip.value.strip()
            ):
                non_static.append(location)

    assert not missing, "Inputs without tooltips:\n" + "\n".join(missing)
    assert not non_static, "Tooltips must be direct non-empty string literals:\n" + "\n".join(non_static)
