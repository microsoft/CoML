from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import orjson
from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from peewee import fn

from mlcopilot.constants import *
from mlcopilot.constants import TOKEN_COMPLETION_LIMIT, TOKEN_LIMIT
from mlcopilot.experience import gen_experience
from mlcopilot.knowledge import get_knowledge
from mlcopilot.orm import Knowledge, Solution, Space, Task, database_proxy
from mlcopilot.space import import_space, print_space
from mlcopilot.utils import (
    clean_input,
    escape,
    get_llm,
    get_token_count_func,
    parse_configs,
    set_llms,
)

zero_shot_prompt = """Recommend best configurations to train a model for a new task. Format strictly follows this template: ```Configuration 1: {{parameter_1_name}} is {{parameter_1_value}}. {{parameter_2_name}} is {{parameter_2_value}}...{{parameter_n_name}} is {{parameter_n_value}}.
Configuration 2: {{parameter_1_name}} is {{parameter_1_value}}. {{parameter_2_name}} is {{parameter_2_value}}...{{parameter_n_name}} is {{parameter_n_value}}.
Configuration 3: {{parameter_1_name}} is {{parameter_1_value}}. {{parameter_2_name}} is {{parameter_2_value}}...{{parameter_n_name}} is {{parameter_n_value}}.
```
"""


def suggest_interactive() -> None:
    """
    Suggest configurations interactively.

    Returns
    -------
    None
    """
    while True:
        print_space()

        select_space_id = clean_input(f"Please select a space(input space ID): ")
        space = import_space(select_space_id)
        if space is None:
            print(f"Space '{select_space_id}' not exist.")
            continue

        print(
            f"You selected space:\n (Space ID = {select_space_id}){space.desc}" "\n\n"
        )
        task_desc = clean_input("Your description for new task: ").strip(".") + "."
        suggest_configs, knowledge = suggest(space, task_desc)
        if knowledge:
            print(f"\n* Rationale: ")
            print(knowledge)
        print(f"\n* Recommended configurations: ")
        if isinstance(suggest_configs, str):
            print(suggest_configs)
        else:
            for i, suggest_config in enumerate(suggest_configs):
                print(f"Suggested configuration {i+1}. {suggest_config}")
        # press any key to continue, press 'q' to quit
        if clean_input('Press any key to continue, press "q" to quit: ') == "q":
            break


def suggest(space: Space, task_desc: str) -> Tuple[Any, Union[str, None]]:
    """
    Suggest configurations for a new task.

    Parameters
    ----------
    space: Space
        The space to suggest configurations.
    task_desc: str
        The description of the new task.

    Returns
    -------
    Tuple[Any, Union[str, None]]
        A tuple of suggested configurations and rationale.
    """
    task_desc = f"""Task: {task_desc}"""

    retrieved_tasks, examples = gen_experience(space, task_desc)
    knowledge = get_knowledge(space) or (
        get_knowledge(space, retrieved_tasks[0]) if len(retrieved_tasks) else None
    )

    llm = get_llm("suggest")()
    quantile_infos = orjson.loads(space.quantile_info) if space.quantile_info else None

    if space.space_id in ("model", "hp"):
        prompt = _get_prompt_in_huggingface(space, task_desc, examples, knowledge)
    else:
        if len(examples):
            prompt = _get_prompt_with_few_shot(space, task_desc, examples, knowledge)
        else:
            prompt = _get_prompt_with_zero_shot(space, task_desc)

    response = llm(prompt)
    if quantile_infos:
        suggest_configs = parse_configs(
            response,
            TOP_K,
            inverse_bin_map,
            quantile_infos,
        )
    else:
        suggest_configs = response
    return suggest_configs, knowledge


def _get_prompt_with_few_shot(
    space: Space, task_desc: str, examples: List[Dict[str, Any]], knowledge: str
) -> str:
    """
    Generate prompt with few-shot examples.

    Parameters
    ----------
    space: Space
        The space to suggest configurations.
    task_desc: str
        The description of the new task.
    examples
        A list of examples.
    knowledge: str
        The knowledge for the space.

    Returns
    -------
    str
        The prompt with few-shot examples.
    """
    prefix_token = get_token_count_func()(
        "Here are some tasks along with best hyper-parameter configurations to train a model on them.\n"
    )
    if knowledge:
        suffix_token = get_token_count_func()(
            "\nGuidelines:{knowledge}\n\n\nBased on the examples and guidelines above, recommend {TOP_K} hyper-parameter configurations for a new classification dataset.\n\n{output}".format(
                knowledge=knowledge,
                TOP_K=str(TOP_K),
                output=(f"{task_desc}\n\n"),
            )
        )
    else:
        suffix_token = get_token_count_func()(
            "\n\n\nBased on the examples above, recommend {TOP_K} hyper-parameter configurations for a new classification dataset.\n\n{output}".format(
                TOP_K=str(TOP_K),
                output=(f"{task_desc}\n\n"),
            )
        )

    example_prompt = PromptTemplate(
        input_variables=["input"],
        template="{input}",
    )

    example_selector = LengthBasedExampleSelector(
        examples=[{"input": example} for example in examples],
        example_prompt=example_prompt,
        max_length=TOKEN_LIMIT
        - prefix_token
        - suffix_token
        - TOKEN_COMPLETION_LIMIT
        - 20,  # 20 is the additional length for ensuring the completion.
        get_text_length=get_token_count_func(),
    )

    if knowledge:
        dynamic_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix="Here are some tasks along with best hyper-parameter configurations to train a model on them.\n",
            suffix="\nGuidelines:{knowledge}\n\n\nBased on the examples and guidelines above, recommend {TOP_K} hyper-parameter configurations for a new classification dataset.\n\n{output}",
            input_variables=["knowledge", "TOP_K", "output"],
        )
        prompt = dynamic_prompt.format(
            knowledge=knowledge,
            TOP_K=str(TOP_K),
            output=(f"{task_desc}\n\n"),
        )
    else:
        dynamic_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix="Here are some tasks along with best hyper-parameter configurations to train a model on them.\n",
            suffix="\n\n\nBased on the examples above, recommend {TOP_K} hyper-parameter configurations for a new classification dataset.\n\n{output}",
            input_variables=["TOP_K", "output"],
        )
        prompt = dynamic_prompt.format(
            TOP_K=str(TOP_K),
            output=(f"{task_desc}\n\n"),
        )
    return prompt


def _get_prompt_with_zero_shot(space: Space, task_desc: str) -> str:
    """
    Generate prompt with zero-shot examples.

    Parameters
    ----------
    space: Space
        The space to suggest configurations.
    task_desc: str
        The description of the new task.

    Returns
    -------
    str
        The prompt with zero-shot examples.
    """
    promptTemplate = PromptTemplate.from_template(
        "Recommend best configurations to train a model for a new task.\n\n{space_desc}{zero_shot_prompt}\n{task_desc}\n\n"
    )
    prompt = promptTemplate.format(
        space_desc=space.desc,
        zero_shot_prompt=zero_shot_prompt,
        task_desc=task_desc,
    )
    return prompt


def _get_prompt_in_huggingface(
    space: Space, task_desc: str, examples: List[Dict[str, Any]], knowledge: str
) -> str:
    """
    Generate prompt with few-shot examples.

    Parameters
    ----------
    space: Space
        The space to suggest configurations.
    task_desc: str
        The description of the new task.
    examples
        A list of examples.

    Returns
    -------
    str
        The prompt with few-shot examples.
    """
    prefix_token = get_token_count_func()(
        "Here are some tasks along with best configurations on them.\n"
    )
    suffix_token = get_token_count_func()(
        "\nGuidelines:{knowledge}\n\n\nBased on the examples and guidelines above, provide some suggestions for this new task.\n\n{output}".format(
            knowledge=knowledge,
            output=(f"{task_desc}\n\n"),
        )
    )

    example_prompt = PromptTemplate(
        input_variables=["input"],
        template="{input}",
    )

    example_selector = LengthBasedExampleSelector(
        examples=[{"input": escape(example)} for example in examples],
        example_prompt=example_prompt,
        max_length=TOKEN_LIMIT
        - prefix_token
        - suffix_token
        - TOKEN_COMPLETION_LIMIT
        - 20,  # 20 is the additional length for ensuring the completion.
        get_text_length=get_token_count_func(),
    )

    dynamic_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Here are some tasks along with best configurations on them.\n",
        suffix="\nGuidelines:{knowledge}\n\n\nBased on the examples and guidelines above, provide some suggestions for this new task.\n\n{output}",
        input_variables=["knowledge", "output"],
    )
    prompt = dynamic_prompt.format(
        knowledge=knowledge,
        output=(f"{task_desc}\n\n"),
    )
    return prompt
