from __future__ import annotations

import json
import re
import types
from pathlib import Path
from typing import Any, TypedDict, cast


class GenerateContextIncomplete(TypedDict):
    variables: dict[str, str]
    codes: list[str]
    request: str


class GenerateContext(GenerateContextIncomplete):
    answer: str


class InteractionIncomplete(TypedDict):
    error: str | None
    output: str | None
    hint: str | None


class Interaction(InteractionIncomplete):
    explanation: str
    observation: str
    code: str


class FixContext(TypedDict):
    variables: dict[str, str]
    codes: list[str]
    request: str | None
    first_attempt: str
    interactions: list[InteractionIncomplete | Interaction]


PANDAS_DESCRIPTION_CONFIG: Any = dict(max_cols=10, max_colwidth=20, max_rows=10)
MAXIMUM_LIST_ITEMS = 30


def describe_variable(value: Any) -> str:
    import numpy
    import pandas

    if isinstance(value, numpy.ndarray):
        return "numpy.ndarray(shape={}, dtype={})".format(value.shape, value.dtype)
    elif isinstance(value, pandas.DataFrame):
        return "pandas.DataFrame(shape={}, columns={})\n{}".format(
            value.shape,
            describe_variable(value.columns.tolist()),
            add_indent(value.to_string(**PANDAS_DESCRIPTION_CONFIG).rstrip()),
        )
    elif isinstance(value, pandas.Series):
        return "pandas.Series(shape={})".format(value.shape)
    elif isinstance(value, list):
        if len(value) > MAXIMUM_LIST_ITEMS:
            return "[{}, ...]".format(
                ", ".join(describe_variable(v) for v in value[:MAXIMUM_LIST_ITEMS])
            )
        return "[{}]".format(", ".join(describe_variable(v) for v in value))
    elif isinstance(value, dict):
        if len(value) > MAXIMUM_LIST_ITEMS:
            return "{{{}, ...}}".format(
                ", ".join(
                    f"{k}: {describe_variable(v)}"
                    for k, v in list(value.items())[:MAXIMUM_LIST_ITEMS]
                )
            )
        return "{{{}}}".format(
            ", ".join(f"{k}: {describe_variable(v)}" for k, v in value.items())
        )
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, (bool, int, float)):
        return str(value)
    elif value is None:
        return "None"
    else:
        val = str(value)
        if len(val) > 300:
            val = val[:300] + "..."
        return val


def add_indent(code: str, indent: str = "    ") -> str:
    return "".join(indent + line for line in code.splitlines(True))


def filter_variables(variables: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in variables.items()
        if not isinstance(
            value, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType)
        )
        and not name.startswith("__")
        and name
        not in [
            "_ih",
            "In",
            "_i",
            "_ii",
            "_iii",
            "_oh",
            "_dh",
            "Out",
            "get_ipython",
            "exit",
            "quit",
            "_",
        ]
        and not re.match(r"_i\d+", name)
    }


def extract_shot(problem_path: Path | str, index: int) -> dict:
    # https://github.com/ultmaster/pandas_exercises
    from dseval import Environment, ProblemSet

    problems = ProblemSet.fromfile(Path(problem_path))
    environment = Environment()

    codes = []
    for i, problem in enumerate(problems):
        if i >= index:
            break
        if problem.reference_code:
            answer = environment.execute(problem.reference_code)
            code = problem.reference_code
            if problem.question:
                prompt = "".join(
                    [
                        "# " + question_line
                        for question_line in problem.question.rstrip().splitlines(True)
                    ]
                )
                code = prompt.rstrip() + "\n" + code
            codes.append(code)

    return {
        "variables": {
            key: describe_variable(value)
            for key, value in filter_variables(environment.namespace).items()
        },
        "codes": codes,
        "request": problems[index].question,
        "answer": problems[index].reference_code,
    }


def render_code(code: str) -> str:
    return "```python\n" + code.rstrip() + "\n```"


def render_ipython_cells(codes: list[str]) -> str:
    return "".join(f"# %%\n{code.rstrip()}\n\n" for code in codes)


def render_generate_context(
    context: GenerateContext | GenerateContextIncomplete,
) -> tuple[str, str | None]:
    if context["variables"]:
        variables = (
            "Variables:\n\n"
            + "".join(
                f"{name}: {desc}\n" for name, desc in context["variables"].items()
            )
            + "\n\n"
        )
    else:
        variables = "No variables available currently.\n\n"
    if context["codes"]:
        code = "Executed code:\n\n" + render_ipython_cells(context["codes"]) + "\n"
    else:
        code = "No code has been executed yet.\n\n"

    if context["request"]:
        request = "Request:\n" + context["request"].rstrip()
    else:
        request = "User request is unclear."

    if "answer" in context:
        answer = render_code(context["answer"])
    else:
        answer = None

    return code + variables + request, answer


def render_fix_context(context: FixContext) -> list[str]:
    all_interactions: list[str] = []
    task_begin = "### Task Start ###\n\n"
    if context["request"] is None:
        interaction_prefix = (
            task_begin
            + "I wrote the following code and it's problematic. Please help fix it.\n\n"
            + render_code(context["first_attempt"])
        )
    else:
        first_request, _ = render_generate_context(context)  # type: ignore
        first_request = task_begin + first_request
        all_interactions += [first_request, render_code(context["first_attempt"])]
        interaction_prefix = ""
    for index, interaction in enumerate(context["interactions"]):
        if index == 0:
            if interaction_prefix:
                instruction = interaction_prefix
            else:
                instruction = (
                    "The generated code is problematic. Please help fix it.\n\n"
                )
        else:
            instruction = "The code is still problematic. Please help fix it.\n\n"

        if interaction["error"]:
            error = (
                "- The code is executed with the following exception:\n"
                + add_indent(interaction["error"].rstrip(), "  ")
                + "\n\n"
            )
        else:
            error = "- The code is executed without any exception.\n\n"

        if interaction["output"]:
            output = (
                "- The code is executed in a IPython cell, and here is its output:\n"
                + add_indent(interaction["output"].rstrip(), "  ")
                + "\n\n"
            )
        else:
            output = (
                "- The code is executed in a IPython cell, and it has NO OUTPUT.\n\n"
            )

        if interaction["hint"]:
            hint = (
                "- Here is a hint from the user: "
                + interaction["hint"].rstrip()
                + "\n\n"
            )
        else:
            hint = "- The user did not provide any hint.\n\n"

        post_instruction = 'With the information above, please first explain the code line-by-line, and then observe what might be wrong. Finally, you should provide the fixed code. If you think the code is correct, you can simply write "THE CODE IS CORRECT." in the observation section.'

        all_interactions.append(instruction + error + output + hint + post_instruction)

        if (
            "explanation" in interaction
            and "observation" in interaction
            and "code" in interaction
        ):
            explanation = (
                "Here is a line-by-line explanation of the code:\n"
                + interaction["explanation"].rstrip()
                + "\n\n"
            )
            observation = (
                "Observe what is wrong with the code:\n"
                + interaction["observation"].rstrip()
                + "\n\n"
            )
            all_interactions.append(
                explanation
                + observation
                + "The fixed code:\n\n"
                + render_code(interaction["code"])
            )

    return all_interactions


def render_check_context(code: str, context: GenerateContext | FixContext) -> str:
    result, _ = render_generate_context(cast(GenerateContextIncomplete, context))
    result += f"\n\nGenerated code:\n\n```\n{code.rstrip()}\n```"
    return result


def render_sanity_check_context(
    code: str,
    context: GenerateContext | FixContext,
    error: str | None,
    output: str | None,
) -> str:
    result, _ = render_generate_context(cast(GenerateContextIncomplete, context))
    result += f"\n\nGenerated code:\n\n```\n{code.rstrip()}\n```"
    if output:
        result += f"\n\nOutput of the code:\n\n```\n{output.rstrip()}\n```"
    else:
        result += "\n\nNo output."
    if error:
        result += f"\n\nError message:\n\n```\n{error.rstrip()}\n```"
    return result


GENERATE_INSTRUCTION = f"""You're an assistant of a data scientist. You're good at writing Python code to do data analysis, visualization, and machine learning. The user is working in a IPython interactive notebook, and needs your help in solving a particular problem. The user will give you some context (e.g., variables available and already-executed code currently). Your goal is to write a new cell in the notebook that serves the user's request.

Instructions:

- The generated code should be wrapped by ``` before and after it.
- Import necessary libraries at the beginning. You can leverage the Python libraries such as `pandas`, `sklearn`, `matplotlib`, `seaborn`, and etc. to achieve user's request.
- The output of a cell is the last statement of the code. Do not use `print` to output the result or `return` to return the result.
- Do not overwrite or modify the variables provided by the user, unless the user has explicitly asked for it. For example, if the user has provided a DataFrame `df`, you should not reassign `df`, unless the user asks to modify `df` in-place.
"""

FIX_INSTRUCTION = f"""{GENERATE_INSTRUCTION.rstrip()}
- If the user thinks the generated code is problematic, you should help fix it. The user will provide you with the exception message (if any), the output of the code (if any), and a hint (if any). You should provide a line-by-line explanation of the code, and point out what is wrong with the code. You should also provide the fixed code.
"""

SUGGEST_INSTRUCTION = """You're a data scientist. Given the code that has already been written, suggest three things that can be done next. Write the response in the following format:

1. ...
2. ...
3. ...

The suggestions should be specific, actionable, yet concise.
"""

EXPLAIN_INSTRUCTION = "You are a data scientist. Please generate a line-by-line explanation for the code given."

CHECK_INSTRUCTION = """
You are a data scientist. Please generate a line-by-line explanation for the code given.
Then answer what is the purpose of the code, what is the code doing, and whether it satisfies user's intention.
Finally, output a word "CORRECT" or "INCORRECT" in a single line to indicate whether the code is correct.
""".strip()

SANITY_CHECK_INSTRUCTION = """
You are a data scientist. You've written some code to solve a certain problem.
The code has produced some output (and optionally some error messages). You want to check whether the output is correct.
Please take a look at the output, and print a word "CORRECT" or "INCORRECT" in a single line to indicate whether the output is correct.
Derive the answer step by step.
"""


def cached_generate_fewshots() -> list[GenerateContext]:
    with open(Path(__file__).parent / "prompts/generate_fewshots.json") as f:
        return json.load(f)


def cached_fix_fewshots() -> list[FixContext]:
    with open(Path(__file__).parent / "prompts/fix_fewshots.json") as f:
        return json.load(f)


def update_generate_fewshots():
    GENERATE_FEWSHOTS = [
        extract_shot("../pandas_exercises/outputs/02-filtering-sorting-chipotle.py", 2),
        extract_shot(
            "../pandas_exercises/outputs/03-grouping-alcohol-consumption.py", 3
        ),
        extract_shot("../pandas_exercises/outputs/05-merge-auto-mpg.py", 2),
        extract_shot("../pandas_exercises/outputs/09-time-series-apple-stock.py", 8),
        extract_shot("../pandas_exercises/outputs/07-visualization-chipotle.py", 3),
    ]
    with open(Path(__file__).parent / "prompts/generate_fewshots.json", "w") as f:
        json.dump(GENERATE_FEWSHOTS, f, indent=2)


def update_fix_fewshots():
    shot0 = extract_shot(
        "../pandas_exercises/outputs/02-filtering-sorting-chipotle.py", 3
    )
    shot0[
        "first_attempt"
    ] = """item_prices = chipo.groupby('item_name')['item_price'].max()
item_prices_df = pd.DataFrame({'item_name': item_prices.index, 'item_price': item_prices.values})
item_prices_df.sort_values(by='item_price', ascending=False)"""
    answer = shot0.pop("answer")
    shot0["interactions"] = [
        {
            "error": None,
            "output": """                                item_name  item_price
24           Chips and Fresh Tomato Salsa       44.25
11                          Carnitas Bowl       35.25
..                                    ...         ...
26      Chips and Mild Fresh Tomato Salsa        3.00
28     Chips and Roasted Chili-Corn Salsa        2.39""",
            "hint": "The item_price column in chipo refers to the total price of order.",
            "explanation": """1. `item_prices = chipo.groupby('item_name')['item_price'].max()`: This calculates the highest price for each unique item using the 'item_name' column as groups.
2. `item_prices_df = pd.DataFrame(...)`: This creates a DataFrame with 'item_name' and 'item_price' columns, using the calculated maximum prices.
3. `item_prices_df.sort_values(by='item_price', ascending=False)`: This sorts the DataFrame by 'item_price' in descending order, showing items with the highest prices at the top.""",
            "observation": "The code didn't filter out orders with quantity > 1. So the prices are not correct.",
            "code": """# Filter out orders with quantity > 1
chipo_single_item = chipo[chipo['quantity'] == 1]
# Calculate the total price for each item
item_prices = chipo_single_item.groupby('item_name')['item_price'].mean()
# Create a DataFrame with item_name and item_price columns
item_prices_df = pd.DataFrame({'item_name': item_prices.index, 'item_price': item_prices.values})
# Sort the DataFrame by item_price in descending order
item_prices_df.sort_values(by='item_price', ascending=False)""",
        },
        {
            "error": None,
            "output": """                                item_name  item_price
15                    Carnitas Salad Bowl   11.056667
42                       Steak Salad Bowl   10.964074
..                                    ...         ...
10                      Canned Soft Drink    1.250000
9                             Canned Soda    1.090000""",
            "hint": "Do not take the mean of item_price. Use the first price for each item instead.",
            "explanation": """1. `chipo_single_item = chipo[chipo['quantity'] == 1]`: Filters out orders with a quantity greater than 1, creating a new DataFrame.
2. `item_prices = chipo_single_item.groupby('item_name')['item_price'].mean()`: Calculates the average price per unique item.
3. `item_prices_df = pd.DataFrame(...)`: Creates a DataFrame with item names and their average prices.
4. `item_prices_df.sort_values(by='item_price', ascending=False)`: Sorts the DataFrame by average item price in descending order, presenting the highest priced items first.""",
            "observation": "The code computes the average of item_price, but the user wants the first price.",
            "code": answer,
        },
    ]

    shot1 = extract_shot("../pandas_exercises/outputs/03-grouping-occupation.py", 3)
    shot1[
        "first_attempt"
    ] = "(users[users['gender'] == 'M'].groupby('occupation').gender.count() / users.groupby('occupation').gender.count()).sort_values(ascending=False)"
    shot1["interactions"] = [
        {
            "error": None,
            "output": """occupation
doctor           1.000000
engineer         0.970149
...                   ...
healthcare       0.312500
homemaker        0.142857
Name: gender, dtype: float64""",
            "hint": "Please show the ratio in percentage.",
            "explanation": "This code calculates the proportion of male users for each occupation in the 'users' DataFrame, then sorts and presents these proportions in descending order.",
            "observation": "The code is almost correct, but the user wants the ratio in percentage.",
            "code": "male_ratio_per_occupation = (users[users['gender'] == 'M'].groupby('occupation').gender.count() / users.groupby('occupation').gender.count() * 100).sort_values(ascending=False)",
        }
    ]

    shot2 = extract_shot(
        "../pandas_exercises/outputs/09-time-series-investor-flow-of-funds-us.py", 3
    )
    answer = shot2.pop("answer")
    shot2["first_attempt"] = "flow.index.freq"
    shot2["interactions"] = [
        {
            "error": None,
            "output": None,
            "hint": None,
            "explanation": "In time series data, the .index.freq in Pandas refers to the frequency of the DateTimeIndex. It helps identify how often data points occur, like daily, weekly, etc. You can use .index.freq to access or set the frequency for the DateTimeIndex in your code. If the data has a regular pattern, this attribute can be quite useful for analysis and manipulation.",
            "observation": "The returned value is None, which means the frequency is not set.",
            "code": "pd.infer_freq(flow.index)",
        },
        {
            "error": None,
            "output": None,
            "hint": None,
            "explanation": """The pd.infer_freq() function is used to automatically determine the frequency of a DateTimeIndex based on the time intervals between consecutive dates in the index. It analyzes the time gaps between the dates and tries to identify the most likely frequency that fits the pattern of the data. This function is applied to the DateTimeIndex of the `flow` DataFrame. It attempts to figure out the most appropriate frequency for the time series data based on the pattern of dates.""",
            "observation": "The returned value is still None. Notice that the gaps between dates are not regular. Some dates are skipped. Therefore the frequency needs to be inferred on a partial index.",
            "code": answer,
        },
    ]

    shot3 = extract_shot(
        "../pandas_exercises/outputs/01-getting-knowing-your-data-chipotle.py", 3
    )
    shot3[
        "first_attempt"
    ] = """import pandas as pd
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"
chipo = pd.DataFrame(shape=(4622, 5), columns=pd.Index(['order_id', 'quantity', 'item_name', 'choice_description', 'item_price'], dtype='object'))"""
    answer = shot3.pop("answer")
    shot3["interactions"] = [
        {
            "error": """Traceback (most recent call last):
  File \"<string>\", line 3, in <module>
TypeError: DataFrame.__init__() got an unexpected keyword argument 'shape'
""",
            "output": None,
            "hint": "You should use existing DataFrames.",
            "explanation": """1. `import pandas as pd`: Imports the Pandas library and assigns it the alias `pd`.
2. `url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"`: Defines a URL string pointing to a dataset (in TSV format) related to Chipotle orders.
3. `chipo = pd.DataFrame(shape=(4622, 5), columns=pd.Index(['order_id', 'quantity', 'item_name', 'choice_description', 'item_price'], dtype='object'))`: Creates a new DataFrame named `chipo` using the `pd.DataFrame()` constructor. Defines the column names using `pd.Index()`, which are 'order_id', 'quantity', 'item_name', 'choice_description', and 'item_price'. Sets the data type of the columns as 'object'. The error occurred in the original code because the shape parameter is not a valid parameter for initializing a DataFrame using the pd.DataFrame() constructor.""",
            "observation": "The code creates a new DataFrame, but the user wants to use the existing DataFrame `chipo`. Also, the code uses the `shape` parameter, which is not a valid parameter.",
            "code": answer,
        }
    ]

    with open(Path(__file__).parent / "prompts/fix_fewshots.json", "w") as f:
        json.dump([shot0, shot1, shot2, shot3], f, indent=2)


def updateall():
    global PANDAS_DESCRIPTION_CONFIG, MAXIMUM_LIST_ITEMS
    PANDAS_DESCRIPTION_CONFIG = dict(max_cols=6, max_colwidth=12, max_rows=5)
    MAXIMUM_LIST_ITEMS = 5
    update_generate_fewshots()
    update_fix_fewshots()


if __name__ == "__main__":
    updateall()
