from __future__ import annotations

import json
import random
import re
import sys
import warnings

import numpy as np
import pandas as pd
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .deconstruct import deconstruct

VERIFICATION_INSTRUCTION = """
You're an assistant of a data scientist.  You're very good at verifying visualizations to their visualization goals.
You will be given a visualization goal, a description of dataset and a description of a visualization.
Please first analyze what each column represents, data type(nominal, quantitative or temporal), and the approximate range of values based on the dataset description.
Then analyze the visualization goal in terms of chart type and visual encoding.
Finally check if the visualization meets the visualization goal and dataset description.

Instructions:

- Please analyze carefully first, and then summarize your judgment.
- You just need to check the information mentioned in the visualization goal or visualization description. If it doesn't exist then you don't need to check.
- Your OUTPUT MUST BE A VALID JSON LIST OF OBJECTS in the format:
    ```[
        {"aspect": "chart type", "rationale": "The visualization is a vertical stacked bar chart, which meets the bar chart in the visualization goal.", "answer": True,},
        {"aspect": "x channel", "rationale": "The x channel represents Sales which match visualization goal. However, Sales are quantitative in the dataset description but nominal in the visualization", "answer": False},
        {"aspect": "y channel", "rationale": "The y channel represents the maximum yield, which does not match the average yield described by the goal.", "answer": False}
        {"aspect": "title", "rationale": "The charts are titled Trends in Population Change, which coincides with the goal of the visualization.", "answer": True}
    ]
    ```
"""

VERIFICATION_EXAMPLES = [
    {
        "question": """
Visualization Goal: Bar chart of the total number faculty with each Sex in each rank.
Dataset Description: {"Faculty_dataset": "pandas.DataFrame(shape=(58, 8), columns=[\"FacID\", \"Lname\", \"Fname\", \"Rank\", \"Sex\", \"Phone\", \"Room\", \"Building\"])\n        FacID       Lname     Fname        Rank Sex  Phone  Room Building\n    0    1082    Giuliano      Mark  Instructor   M   2424   224      NEB\n    1    1121    Goodrich   Michael   Professor   M   3593   219      NEB\n    2    1148      Masson    Gerald   Professor   M   3402  224B      NEB\n    3    1172  Runolfsson   Thordur   AssocProf   M   3121   119   Barton\n    4    1177      Naiman    Daniel   Professor   M   3571   288  Krieger\n    ..    ...         ...       ...         ...  ..    ...   ...      ...\n    53   9811          Wu     Colin    AsstProf   M   2906   288  Krieger\n    54   9823        Pang  Jong-Shi   Professor   M   4366   288  Krieger\n    55   9824      Glaser    Robert  Instructor   M   4396   119   Barton\n    56   9826     Delcher    Arthur  Instructor   M   2956   329      NEB\n    57   9922        Hall    Leslie    AsstProf   F   7332   288  Krieger"\}
Chart Description: The chart type is vertical stacked bar chart. It's title is Total Number of Faculty by Sex and Rank. The channel x represents Rank whose ticks is ["F", "M"] and data type is nominal. The channel y represents Number of Faculty whose ticks is [0,10,20,30,40,50] and data type is quantitative. The channel fill represents Rank whose ticks is ["AssocProf", "AsstProf", "Instructor", "Professor"] and data type is nominal.
    """,
        "answer": """[
        {"aspect": "chart type", "rationale": "The visualization is a vertical stacked bar chart, which meets the bar chart in the visualization goal.", "answer": True},
        {"aspect": "x channel", "rationale": "The x channel represents Rank which match visualization goal. But, "F", "M" (in the visualization), etc. look more like \"Sex\" than \"Rank\" according to the dataset description.", "answer": False},
        {"aspect": "y channel", "rationale": "The y channel represents Number of Faculty, which match the total number faculty in the visualization goal. The data type it represents is quantitative, which is the same as the quantitative type implied by number of faculty.", "answer": True},
        {"aspect": "fill channel", "answer": True, "rationale": "The fill channel represents Rank, which match rank in the visualization goal. Its data type is nominal, consistent with what was inferred from the dataset description.", "answer": True},
        {"aspect": "title", "rationale": "The charts are titled Total Number of Faculty by Sex and Rank, which coincides with the goal of the visualization.", "answer": True}
    ]""",
    }
]

ANALYSIS_ORDER_INSTRUCTION = f"""You're an assistant of a data scientist. You're good at data analysis and visualization. The user will give you a sentence describing his visualization goal. Your task is to identify whether the sentence has explicitly mentioned sorting or ordering requirements.

Instructions:

- Please note that finding the maximum or minimum value is not the same as sorting!
- Please note that rank may just be a noun and does not indicate ordering.
- If not, your answer should be None. If yes, your answer should be such a JSON {json.dumps({"channel ": "", "order": ""})} to describe which channel 'x' or 'y' needs to be ordered, and in what order ( 'ascending','descending' or custom sort order).
- Your answer should be wrapped by ``` before and after it.
"""

ANALYSIS_ORDER_EXAMPLES = [
    {
        "question": "Show all the ranks and the number of male and female faculty for each rank in a bar chart.",
        "answer": "None",
    },
    {
        "question": "Show the production budget of the movie that has the highest Worldwide Gross in each major genre using a horizontal bar chart.",
        "answer": "None",
    },
    {
        "question": "A bar chart showing trends in subway traffic over the seven days of the week. x-axis represents day names in terms of three letter initials (e.g., 'mon'), sorted from monday to sunday.",
        "answer": json.dumps(
            {"channel": "x", "order": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]}
        ),
    },
    {
        "question": "Use a line graph to show the change in sales over twelve months. x-axis indicates the month (e.g., Jan, Feb), sorted from January to December.",
        "answer": json.dumps(
            {
                "channel": "x",
                "order": [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ],
            }
        ),
    },
    {
        "question": "Show the number of documents in different starting date and bin starting date by weekday interval with a bar chart, and rank y axis in desc order please.",
        "answer": json.dumps({"channel": "y", "order": "descending"}),
    },
    {
        "question": "Show the amount for all the payments processed with Visa distributing the payment date with a bar chart, I want to order in ascending by the amount of payment please.",
        "answer": json.dumps({"channel": "y", "order": "ascending"}),
    },
]


def parse_answer(response: str) -> str:
    match = re.search(r"```.*```", response)
    if match is not None:
        answer = match.group(0)[3:-3]
    else:
        # Give up. Return whole response.
        warnings.warn("Unable to parse the nl query from response.")
        answer = response
    return answer


def chart_summary(info: dict):
    summarization = ""
    if "chart" in info:
        summarization += f"""The chart type is {info['chart']}. """
    if "title" in info:
        summarization += f"""It's title is {info['title']}. """

    encoding = info["encoding"]
    for channel in encoding.keys():
        if "title" in encoding[channel]:
            summarization += f"""The channel {channel} represents {encoding[channel]['title']} whose """
        else:
            summarization += f"""Channel {channel}\'s"""
        if encoding[channel]["type"] == "quantitative":
            summarization += f"""domain is {encoding[channel]['scale']['domain']}"""
        else:
            if "ticks" in encoding[channel]["scale"]:
                summarization += f"""ticks is {encoding[channel]['scale']['ticks']}"""
            else:
                summarization += f"""domain is {encoding[channel]['scale']['domain']}"""
        summarization += f""" and data type is {encoding[channel]['type']}."""

    return summarization


def answer_question(
    question: str, request: str, previous_code: str, variable_descriptions: dict, agent
):
    prerequisites = f"I'm working on \"{request}\". Note that you don't need to draw the visualization, just complete the transformation of the data. "
    # generate code to verify value
    generating_context = agent.generate_code(
        prerequisites
        + question
        + " Please store your finding in a variable called 'finding'.",
        variable_descriptions,
        [previous_code],
    )
    code = generating_context["answer"]
    final_code = previous_code + "\n" + code
    # do not show figure
    final_code = final_code.replace("plt.show()", "")
    global_env = {"finding": None}
    try:
        exec(final_code, global_env)
        return global_env["finding"]
    except Exception:
        return None


PRECISION = 0.0001


def batch_check(
    data: list[dict],
    request: str,
    chart_info: dict,
    previous_code: str,
    variable_descriptions: dict,
    agent,
) -> list[dict]:
    # Analyze the channel
    ground_truth = chart_info["data"]
    encoding = chart_info["encoding"]
    quantitative_channels = []
    other_channels = []
    for channel in chart_info["encoding"].keys():
        if encoding[channel]["type"] == "quantitative":
            quantitative_channels.append(channel)
        else:
            other_channels.append(channel)

    # Choose a better way to query the data
    if len(quantitative_channels) == 1:
        # generate query, while only one quantitative channel
        query = f"Then find {len(data)} values, "
        items = []
        for datum in data:
            item = f'one for the {encoding[quantitative_channels[0]]["title"]} when '
            conditions = [
                f'the {encoding[channel]["title"]} is {datum[encoding[channel]["field"]]}'
                for channel in other_channels
            ]
            item += " and ".join(conditions)
            items.append(item)
        query += ", ".join(items)
        query += "."
        value = {
            "vis": [
                datum[encoding[quantitative_channels[0]]["field"]] for datum in data
            ],
            "data": answer_question(
                query, request, previous_code, variable_descriptions, agent
            ),
        }

        if value["data"] is None or len(value["data"]) != len(value["vis"]):
            return []
        # compare data
        verifications = []
        for index in range(len(value["data"])):
            try:
                data_value = float(value["data"][index])
                vis_value = value["vis"][index]
                answer = bool(abs((data_value - vis_value) / data_value) < PRECISION)
                verification = {
                    "aspect": "data point",
                    "answer": answer,
                    "rationale": items[index].replace("one for the", "The")
                    + f" is {data_value}.",
                }
                if answer is False:
                    verification[
                        "rationale"
                    ] += f" But in the visualization, the value is {vis_value}."
                verifications.append(verification)
            except Exception:
                pass
        return verifications
    elif len(quantitative_channels) == 2:
        # generate query, while only two quantitative channel
        query = f"Then find {len(data)} lists of values, "
        items = []
        value = {
            "vis": [],
        }
        verifications = []
        for datum in data:
            data_value = datum[encoding[quantitative_channels[0]]["field"]]
            item = f'a list of {encoding[quantitative_channels[1]]["title"]} when the {encoding[quantitative_channels[0]]["title"]} is between {data_value * (1-PRECISION)} and {data_value * (1+PRECISION)}'
            filtered_data = ground_truth
            if len(other_channels) > 0:
                item += " when "
                conditions = [
                    f'the {encoding[channel]["title"]} is {datum[encoding[channel]["field"]]}'
                    for channel in other_channels
                ]
                item += " and ".join(conditions)
                filtered_data = [
                    d
                    for d in filtered_data
                    if all(
                        [
                            d[encoding[channel]["field"]]
                            == datum[encoding[channel]["field"]]
                            for channel in other_channels
                        ]
                    )
                ]
            items.append(item)
            value["vis"].append(
                [
                    d[encoding[quantitative_channels[1]]["field"]]
                    for d in filtered_data
                    if d[encoding[quantitative_channels[0]]["field"]]
                    > data_value * (1 - PRECISION)
                    and d[encoding[quantitative_channels[0]]["field"]]
                    < data_value * (1 + PRECISION)
                ]
            )
            verifications.append(
                {
                    "aspect": "data point",
                    "rationale": item.replace("a list of", "All").replace(
                        f"between {data_value * (1-PRECISION)} and {data_value * (1+PRECISION)}",
                        f"around {data_value}",
                    ),
                }
            )

        query += ", ".join(items)
        query += "."

        value["data"] = answer_question(
            query, request, previous_code, variable_descriptions, agent
        )

        if value["data"] is None or len(value["data"]) != len(value["vis"]):
            return []
        # compare data
        for index in range(len(value["data"])):
            try:
                data_value = value["data"][index]
                # filter nan
                data_value = [d for d in data_value if not pd.isnull(d)]
                data_value = [float(d) for d in data_value]
                vis_value = value["vis"][index]
                verifications[index][
                    "rationale"
                ] += f' are {", ".join([str(d) for d in data_value])}.'
                answer = True
                if len(data_value) == len(vis_value):
                    data_value.sort()
                    vis_value.sort()
                    for index1 in range(len(data_value)):
                        if (
                            abs(
                                (data_value[index1] - vis_value[index1])
                                / data_value[index1]
                            )
                            > PRECISION
                        ):
                            answer = False
                            break
                else:
                    answer = False

                verifications[index]["answer"] = answer
                if answer is False:
                    verifications[index][
                        "rationale"
                    ] += f' But in the visualization, those are {", ".join([str(d) for d in vis_value])}.'
            except Exception:
                verifications[index] = None
        verifications = [verification for verification in verifications if verification]
        return verifications
    return []


def spot_check(
    datum: dict,
    request: str,
    chart_info: dict,
    previous_code: str,
    variable_descriptions: dict,
    agent,
):
    # Analyze the channel
    ground_truth = chart_info["data"]
    encoding = chart_info["encoding"]
    quantitative_channels = []
    other_channels = []
    for channel in chart_info["encoding"].keys():
        if encoding[channel]["type"] == "quantitative":
            quantitative_channels.append(channel)
        else:
            other_channels.append(channel)

    # Choose a better way to query the data
    if len(quantitative_channels) == 1:
        # generate query, while only one quantitative channel
        query = f'Then find the {encoding[quantitative_channels[0]]["title"]} when '
        conditions = [
            f'the {encoding[channel]["title"]} is {datum[encoding[channel]["field"]]}'
            for channel in other_channels
        ]
        query += " and ".join(conditions)
        query += "."
        value = {
            "vis": datum[encoding[quantitative_channels[0]]["field"]],
            "data": answer_question(
                query, request, previous_code, variable_descriptions, agent
            ),
        }
        if value["data"] is None:
            return None
        # parse data
        try:
            value["data"] = list(value["data"])[0]
        except Exception:
            value["data"] = value["data"]
        try:
            value["data"] = float(value["data"])
        except Exception:
            return None

        result = {
            "aspect": "data point",
            "answer": (
                bool(abs((value["data"] - value["vis"]) / value["data"]) < PRECISION)
            ),
            "rationale": query.replace("Then find the", "The")[0:-1]
            + f' is {value["data"]}.',
        }
        if result["answer"] is False:
            result[
                "rationale"
            ] += f' But in the visualization, the value is {value["vis"]}.'
        return result
    elif len(quantitative_channels) == 2:
        # generate query, while only two quantitative channel
        data_value = datum[encoding[quantitative_channels[0]]["field"]]
        query = f'Then find all {encoding[quantitative_channels[1]]["title"]} when the {encoding[quantitative_channels[0]]["title"]} is between {data_value * (1-PRECISION)} and {data_value * (1+PRECISION)}'
        filtered_data = ground_truth
        if len(other_channels) > 0:
            query += " when "
            conditions = [
                f'the {encoding[channel]["title"]} is {datum[encoding[channel]["field"]]}'
                for channel in other_channels
            ]
            query += " and ".join(conditions)
            filtered_data = [
                d
                for d in filtered_data
                if all(
                    [
                        d[encoding[channel]["field"]]
                        == datum[encoding[channel]["field"]]
                        for channel in other_channels
                    ]
                )
            ]
        query += "."
        value = {
            "vis": [
                d[encoding[quantitative_channels[1]]["field"]]
                for d in filtered_data
                if d[encoding[quantitative_channels[0]]["field"]]
                > data_value * (1 - PRECISION)
                and d[encoding[quantitative_channels[0]]["field"]]
                < data_value * (1 + PRECISION)
            ],
            "data": answer_question(
                query, request, previous_code, variable_descriptions, agent
            ),
        }

        if value["data"] is None:
            return None
        # filter nan
        value["data"] = [d for d in value["data"] if not pd.isnull(d)]
        try:
            value["data"] = [float(d) for d in value["data"]]
        except Exception:
            return None
        result = {
            "aspect": "data point",
            "answer": True,
            "rationale": query.replace("Then find all", "All").replace(
                f"between {data_value * (1-PRECISION)} and {data_value * (1+PRECISION)}",
                f"around {data_value}",
            )[0:-1]
            + f' are {", ".join([str(d) for d in value["data"]])}.',
        }
        if len(value["data"]) == len(value["vis"]):
            value["vis"].sort()
            value["data"].sort()
            for index in range(len(value["data"])):
                if (
                    abs(
                        (value["data"][index] - value["vis"][index])
                        / value["data"][index]
                    )
                    > PRECISION
                ):
                    result["answer"] = False
                    break
        else:
            result["answer"] = False

        if result["answer"] is False:
            result[
                "rationale"
            ] += f' But in the visualization, those are {", ".join([str(d) for d in value["vis"]])}.'
        return result

    return None


def get_order(llm: BaseChatModel, request: str):
    """Analysis the order from the given message."""
    messages: list[BaseMessage] = [
        SystemMessage(content=ANALYSIS_ORDER_INSTRUCTION),
    ]

    for example in ANALYSIS_ORDER_EXAMPLES:
        messages.append(HumanMessage(content=example["question"]))
        messages.append(AIMessage(content=f'```{example["answer"]}```'))

    messages.append(HumanMessage(content=request))

    response = llm(messages)
    try:
        answer = parse_answer(response.content)
        if answer == "None":
            return None
        else:
            return json.loads(answer)
    except Exception:
        return None


def check_order(order: dict, chart_info: dict):
    result = {"aspect": "order", "answer": True}
    encoding = chart_info["encoding"]
    data = chart_info["data"]
    if chart_info["mark"] == "arc":
        # by default, the arc is sorted by the start angle of the arc
        # matplotlib: counterclockwise, starting from the x-axis
        vis_order = encoding["fill"]["scale"]["domain"]
        if order["channel"] == "y":
            order["channel"] = "theta"
            sorted_data = sorted(
                chart_info["data"],
                key=lambda x: x["field_theta"],
                reverse=(order["order"] == "descending"),
            )
            request_order = [d["field_fill"] for d in sorted_data]
            # theta
        elif order["channel"] == "x":
            order["channel"] = "fill"
            # fill
            if order["order"] == "ascending":
                request_order = sorted(vis_order)
            elif order["order"] == "descending":
                request_order = sorted(vis_order, reverse=True)
            else:  # custom order
                request_order = order["order"]
        if not (np.array(vis_order) == np.array(request_order)).all():
            result["answer"] = False
    elif chart_info["mark"] in ["bar", "line"]:
        # bar, line
        order_channel = order["channel"]
        other_channel = "y" if order_channel == "x" else "x"

        order_channel_scale = encoding[order_channel]["scale"]
        other_channel_scale = encoding[other_channel]["scale"]
        if (
            encoding[order_channel]["type"] == "nominal"
            or encoding[order_channel]["type"] == "temporal"
        ):
            arr = []
            for index in range(len(order_channel_scale["domain"])):
                arr.append(
                    tuple(
                        [
                            order_channel_scale["domain"][index],
                            order_channel_scale["range"][index],
                        ]
                    )
                )

            if order["order"] == "ascending":
                arr.sort(key=lambda x: x[0], reverse=True)
            elif order["order"] == "descending":
                arr.sort(key=lambda x: x[0], reverse=False)
            else:  # custom order
                sort_order = {}
                for index in range(len(order["order"])):
                    sort_order[order["order"][index]] = index
                arr.sort(key=lambda x: sort_order[x[0]], reverse=True)

            is_sorted = all([arr[i][1] > arr[i + 1][1] for i in range(len(arr) - 1)])
            if not is_sorted:
                result["answer"] = False
        # 'quantitative'
        else:
            # sort other channel
            values_other = []
            for index in range(len(other_channel_scale["domain"])):
                values_other.append(
                    tuple(
                        [
                            other_channel_scale["domain"][index],
                            other_channel_scale["range"][index],
                        ]
                    )
                )
            values_other.sort(key=lambda x: x[1])
            values_other = [item[0] for item in values_other]

            # cumulative
            values_order = []
            for value in values_other:
                data_filter = [
                    d["field_" + order_channel]
                    for d in data
                    if d["field_" + other_channel] == value
                ]
                values_order.append(sum(data_filter))

            if order["order"] == "ascending":
                is_sorted = all(
                    [
                        values_order[i] <= values_order[i + 1]
                        for i in range(len(values_order) - 1)
                    ]
                )
            elif order["order"] == "descending":
                is_sorted = all(
                    [
                        values_order[i] >= values_order[i + 1]
                        for i in range(len(values_order) - 1)
                    ]
                )

            if not is_sorted:
                result["answer"] = False

    result["rationale"] = f"{order['channel']} is sorted in {order['order']} order."
    if result["answer"] is False:
        result["rationale"] = result["rationale"].replace("is sorted", "is not sorted")

    return result


NUM_SAMPLE = 3


class VisVerifier:
    def __init__(self, llm: BaseChatModel, agent):
        self.verifications = []
        self.llm = llm
        # CoMLAgent
        self.agent = agent

    def _add_verification(self, verification):
        self.verifications.append(verification)
        # display
        answer = ""
        if verification["answer"] is True:
            answer = "✅"
        elif verification["answer"] is False:
            answer = "❌"
        elif verification["answer"] is None:
            answer = "❔"
        aspect = verification["aspect"].capitalize()
        rationale = verification["rationale"]
        print(answer + " " + aspect + ": " + rationale)

    def verify(
        self,
        request: str,
        previous_code: str,
        svg_string: str,
        variable_descriptions: dict,
        source: str,
    ):
        self.verifications = []
        understand_fail_result = {
            "answer": None,
            "aspect": "Visualization understanding",
            "rationale": "Cannot understand the visualization.",
        }
        if source == "seaborn":
            # seaborn is based on matplotlib
            source = "matplotlib"
        try:
            # STEP 1: deconstruct svg
            chart_info = deconstruct(svg_string, source)
            if (chart_info is None) or ("data" not in chart_info):
                self._add_verification(understand_fail_result)
            else:
                # STEP2: check chart type, data encoding and title
                self.verify_chart_info(request, chart_info, variable_descriptions)
                pass_verify = all(
                    [
                        verification["answer"] is True
                        for verification in self.verifications
                    ]
                )
                if pass_verify:
                    # STEP3: check visualization data
                    self.verify_data(
                        request, previous_code, chart_info, variable_descriptions
                    )
        except Exception:
            self._add_verification(understand_fail_result)
        return self.verifications

    def verify_chart_info(
        self, request: str, chart_info: dict, variable_descriptions: dict
    ):
        try:
            chart = chart_summary(chart_info)
            messages = [
                SystemMessage(content=VERIFICATION_INSTRUCTION),
            ]
            for example in VERIFICATION_EXAMPLES:
                messages.append(HumanMessage(content=example["question"]))
                messages.append(AIMessage(content=example["answer"]))
            messages.append(
                HumanMessage(
                    content=f"Visualization Goal:{request}\nDataset Description:{json.dumps(variable_descriptions)}\nChart Description:{chart}"
                )
            )
            response = self.llm(messages)
            verifications = eval(response.content)

            for verification in verifications:
                self._add_verification(verification)

            return verifications
        except Exception:
            warnings.warn(str(sys.exc_info()))
        return []

    def verify_data(
        self,
        request: str,
        previous_code: str,
        chart_info: dict,
        variable_descriptions: dict,
    ):
        verifications = []
        try:
            # STEP 1: Spot-Check
            data = chart_info["data"]
            encoding = chart_info["encoding"]
            # check label
            for channel in encoding.keys():
                if "title" not in encoding[channel]:
                    verification = {
                        "aspect": channel + " label",
                        "answer": None,
                        "rationale": "Channel "
                        + channel
                        + " is not labeled, so accurate understanding of the data on the graph is difficult.",
                    }
                    self._add_verification(verification)
                    verifications.append(verification)
            if len(verifications) > 0:
                return verifications

            # random pick NUM_SAMPLE data points
            indexes = range(len(data))
            sampled_indexes = random.sample(indexes, NUM_SAMPLE)
            sampled_data = [data[i] for i in sampled_indexes]
            # Try batch checking first because it's more time efficient。
            verifications += batch_check(
                sampled_data,
                request,
                chart_info,
                previous_code,
                variable_descriptions,
                self.agent,
            )
            for verification in verifications:
                self._add_verification(verification)
            # if batch checking failed
            if len(verifications) == 0:
                for index in sampled_indexes:
                    datum = data[index]
                    verification = spot_check(
                        datum,
                        request,
                        chart_info,
                        previous_code,
                        variable_descriptions,
                        self.agent,
                    )
                    if verification:
                        self._add_verification(verification)
                        verifications.append(verification)
                        if verification["answer"] is not True:
                            break

            pass_verify = all(
                [verification["answer"] is True for verification in verifications]
            )
            if not pass_verify:
                return verifications
            # STEP2: check order
            order = get_order(self.llm, request)
            if order is not None:
                verification = self.check_order(order, chart_info)
                self._add_verification(verification)
                verifications.append(verification)
        except Exception:
            warnings.warn(str(sys.exc_info()))
        return verifications

    def check_order(self, order: dict, chart_info: dict):
        return check_order(order, chart_info)
