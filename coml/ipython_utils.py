from __future__ import annotations

import base64
import json
import re
from typing import Any

from IPython.core.display import Javascript
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display


def is_jupyter_lab_environ() -> bool:
    # https://stackoverflow.com/q/57173235/6837658
    import psutil

    parent = psutil.Process().parent()
    return "jupyter-lab" in parent.name()


def insert_cell_below(code: str, metadata: Any = None) -> None:
    if is_jupyter_lab_environ():
        try:
            input(
                json.dumps(
                    {"command": "insert_cell_below", "code": code, "metadata": metadata}
                )
            )
        except EOFError:
            # e.g., invoked from a widget callback. It will run in the log console.
            from ipylab import JupyterFrontEnd

            app = JupyterFrontEnd()
            app.commands.execute("coml:insert_cell_below", {"code": code, "metadata": metadata})  # type: ignore
    else:
        encoded_code = base64.b64encode(code.encode()).decode()
        encoded_metadata = base64.b64encode(json.dumps(metadata).encode()).decode()
        display(
            Javascript(
                f"""
            const cell = IPython.notebook.insert_cell_below('code');
            cell.set_text(atob("{encoded_code}"));
            cell.metadata.coml = JSON.parse(atob("{encoded_metadata}"));
            cell.focus_cell();
            cell.focus_editor();
        """
            )
        )


def run_code_in_next_cell(python_code: str, metadata: Any = None) -> None:
    if is_jupyter_lab_environ():
        try:
            input(
                json.dumps(
                    {
                        "command": "insert_and_execute_cell_below",
                        "code": python_code,
                        "metadata": metadata,
                    }
                )
            )
        except EOFError:
            # e.g., invoked from a widget callback
            from ipylab import JupyterFrontEnd

            app = JupyterFrontEnd()
            app.commands.execute("coml:insert_and_execute_cell_below", {"code": python_code, "metadata": metadata})  # type: ignore
    else:
        encoded_code = base64.b64encode(python_code.encode()).decode()
        encoded_metadata = base64.b64encode(json.dumps(metadata).encode()).decode()
        display(
            Javascript(
                f"""
            const cell = IPython.notebook.insert_cell_below('code');
            cell.set_text(atob("{encoded_code}"));
            cell.metadata.coml = JSON.parse(atob("{encoded_metadata}"));
            cell.focus_cell();
            cell.execute();
        """
            )
        )


def update_running_cell_metadata(metadata: Any) -> None:
    if is_jupyter_lab_environ():
        input(
            json.dumps(
                {"command": "update_running_cell_metadata", "metadata": metadata}
            )
        )
    else:
        encoded_metadata = base64.b64encode(json.dumps(metadata).encode()).decode()
        display(
            Javascript(
                """
            const cell = comlGetCurrentCell();
            cell.metadata.coml = Object.assign(cell.metadata.coml || {}, JSON.parse(atob(\""""
                + encoded_metadata
                + """\")));
        """
            )
        )


def get_ipython_history(ipython: InteractiveShell) -> list[str]:
    codes = []
    for code in ipython.user_ns["In"]:
        if not code:
            continue
        if code.startswith("get_ipython().run_cell_magic('runit',"):
            # Whitelist
            code_match = re.match(
                r"get_ipython\(\).run_cell_magic\('runit', '', (.*)\)", code
            )
            if code_match is not None:
                code = eval(code_match.group(1))
        if code.startswith("get_ipython().run"):
            continue
        codes.append(code)
    return codes


def get_running_cell() -> dict[str, Any] | None:
    """See `get_last_cell` for the output format."""
    return json.loads(input(json.dumps({"command": "running_cell"})))


def get_last_cell() -> dict[str, Any] | None:
    """The implementation is in nbclassic_init.js. This is a *hacked* RPC channel.

    Example output:

    {
        "metadata":{
            "coml":{
                "variables":{},
                "codes":[],
                "request":"Import the dataset from this [address](https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/master/data/weekly.csv).\nAssign it to a variable called flow",
                "answer":"```python\nimport pandas as pd\n\nurl = 'https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/master/data/weekly.csv'\nflow = pd.read_csv(url)\n```"
            },
            "trusted":true
        },
        "cell_type":"code",
        "source":"import pandas as pd\n\nurl = 'https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/master/data/weekly.csv'\nflow = pd.read_csv(url)",
        "execution_count":3,
        "outputs":[
            {
                "output_type":"stream",
                "text":"123\n",
                "name":"stdout"
            },
            {
                "output_type":"execute_result",
                "execution_count":4,
                "data":{
                    "text/plain":"456"
                }
            },
            {
                "output_type":"error",
                "ename":"NameError",
                "evalue":"123",
                "traceback":[
                    "-------",
                    "NameError Traceback (most recent call last)"
                ]
            }
        ]
    }
    """
    return json.loads(input(json.dumps({"command": "last_cell"})))


def parse_cell_outputs(outputs: list[dict]) -> tuple[str | None, str | None]:
    error = output = None
    for cell_out in outputs:
        if cell_out["output_type"] == "error":
            if "traceback" in cell_out:
                traceback = "\n".join(cell_out["traceback"])
                # Remove color characters
                error = re.sub(r"\x1b(\[.*?[@-~]|\].*?(\x07|\x1b\\))", "", traceback)
            elif "ename" in cell_out:
                error = "Error name: " + cell_out["ename"] + "\n"
                if "evalue" in cell_out:
                    error += "Error value: " + cell_out["evalue"] + "\n"
                break
        elif cell_out["output_type"] == "stream":
            if output is None:
                output = ""
            output += cell_out["text"]
        elif cell_out["output_type"] == "execute_result" and cell_out["data"]:
            if output is None:
                output = ""
            if "text/plain" in cell_out["data"]:
                output += cell_out["data"]["text/plain"] + "\n"
            else:
                output += list(cell_out["data"].values())[0] + "\n"
    return error, output
