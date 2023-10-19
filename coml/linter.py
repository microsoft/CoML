import json
import tempfile
from io import StringIO
from typing import Literal, Tuple

from pylint.lint import Run as PylintRun
from pylint.reporters import JSONReporter

LinterResult = Literal["error", "warning", "info", "ok"]


def lint(previous_code: str, new_code: str) -> Tuple[LinterResult, str]:
    # https://stackoverflow.com/q/75507725/6837658
    pylint_options = [
        "--disable=C0103",  # Invalid name
        "--disable=C0114",  # Missing module docstring
        "--disable=C0304",  # Final new line missing
    ]
    previous_lines = previous_code.count("\n") + 1
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(previous_code + "\n" + new_code)
        f.flush()
        f.seek(0)

        reporter_buffer = StringIO()
        results = PylintRun(
            [f.name] + pylint_options,
            reporter=JSONReporter(reporter_buffer),
            do_exit=False,
        )
        # Score is here.
        # score = results.linter.stats.global_note
        file_results = json.loads(reporter_buffer.getvalue())
        file_results = [e for e in file_results if e["line"] > previous_lines]

        details = []
        for error in file_results:
            line = f"{error['line'] - previous_lines}:{error['column']}: {error['message-id']}: {error['message']}"
            details.append(line)
        details_joined = "\n".join(details)

        if any(e["type"] in ("fatal", "error") for e in file_results):
            return "error", details_joined
        elif any(e["type"] == "warning" for e in file_results):
            return "warning", details_joined
        elif file_results:
            return "info", details_joined
        else:
            return "ok", "No issues found."
