import json
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from .process_utils import run_subprocess

NODE_EXECUTABLE = ["/usr/bin/node", "--es-module-specifier-resolution=node", "main.js"]
COML_DIRECTORY = Path(__file__).parent.parent.parent / "coml"
LOG_IDENTIFIER = "<|coml_nodejs|>\n"

class FunctionDescription(TypedDict):
    name: str
    description: str
    parameters: Dict[str, Any]


def suggest_machine_learning_module(
    existing_modules: List[dict],
    target_role: str,
    target_schema: str
) -> List[dict]:
    return execute_coml_nodejs(
        "suggestMachineLearningModule",
        existing_modules,
        target_role,
        target_schema
    )


def get_function_description() -> FunctionDescription:
    return execute_coml_nodejs("getFunctionDescription")


def serialize_argument(arg: Any) -> str:
    if isinstance(arg, (dict, list)):
        return json.dumps(arg)
    return str(arg)


def execute_coml_nodejs(*args: Any) -> Any:
    serialized_args = [serialize_argument(a) for a in args]
    returncode, stdout, stderr = run_subprocess(
        NODE_EXECUTABLE + serialized_args,
        working_directory=COML_DIRECTORY.resolve(),
        timeout=120
    )
    if returncode != 0:
        raise RuntimeError(f"Node.js process exited with code {returncode}.")
    stdout_decoded = stdout.decode()
    index = stdout_decoded.find(LOG_IDENTIFIER)
    if index == -1:
        raise RuntimeError(f"Node.js process did not print the log identifier. Please re-examine the output.")
    try:
        result = stdout_decoded[index + len(LOG_IDENTIFIER):]
        return json.loads(result)
    except json.JSONDecodeError:
        raise RuntimeError(f"Node.js process did not print a valid JSON string. Please re-examine the output.")
