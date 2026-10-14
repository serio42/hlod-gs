"""Data-only configuration IO, including older Namespace(...) files."""
import ast
import json
import os
from argparse import Namespace


def parse_config(text):
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        tree = ast.parse(text.strip(), mode="eval").body
        if not (isinstance(tree, ast.Call) and isinstance(tree.func, ast.Name)
                and tree.func.id == "Namespace" and not tree.args):
            raise ValueError("Expected a JSON object or a Namespace of literal values")
        values = {}
        for keyword in tree.keywords:
            if keyword.arg is None or keyword.arg in values:
                raise ValueError("Invalid configuration keyword")
            values[keyword.arg] = ast.literal_eval(keyword.value)
    if not isinstance(values, dict) or not all(isinstance(key, str) for key in values):
        raise ValueError("Configuration must be an object with string keys")
    return Namespace(**values)


def load_config(model_path):
    for filename in ("config.json", "cfg_args"):
        path = os.path.join(model_path, filename)
        if os.path.isfile(path):
            with open(path) as stream:
                return parse_config(stream.read())
    raise FileNotFoundError(f"No config.json or cfg_args found in {model_path}")


def save_config(model_path, values):
    values = vars(values) if not isinstance(values, dict) else values
    values = {key: value for key, value in values.items() if not key.startswith("_")}
    with open(os.path.join(model_path, "config.json"), "w") as stream:
        json.dump(values, stream, indent=2, allow_nan=False)
        stream.write("\n")
    # Keep the old filename for downstream tools, without evaluating it ourselves.
    with open(os.path.join(model_path, "cfg_args"), "w") as stream:
        stream.write(str(Namespace(**values)))
