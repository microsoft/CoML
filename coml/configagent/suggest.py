from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import orjson
from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
from peewee import fn

from .constants import *
from .experience import gen_experience
from .knowledge import get_knowledge
from .orm import Knowledge, Solution, Space, Task, database_proxy
from .space import import_space, print_space
from .utils import (
    clean_input,
    escape,
    get_llm,
    get_token_count_func,
    parse_configs,
    set_llms,
)


def print_suggested_configs(configurations: Any, knowledge: str | None) -> None:
    if knowledge:
        print(f"\n* Rationale: ")
        print(knowledge)
    print(f"\n* Recommended configurations: ")
    if isinstance(configurations, str):
        print(configurations)
    else:
        for i, suggest_config in enumerate(configurations):
            print(f"Suggested configuration {i+1}. {suggest_config}")


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
        print_suggested_configs(suggest_configs, knowledge)
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
    knowledge = (
        get_knowledge(space, retrieved_tasks[0])
        if len(retrieved_tasks)
        else get_knowledge(space)
    )

    llm = get_llm("suggest")()
    quantile_infos = orjson.loads(space.quantile_info) if space.quantile_info else None

    prompt_addition_info = {
        "new_task_desc": task_desc,
        "knowledge": knowledge,
        "TOP_K": str(TOP_K),
        "space_desc": space.desc,
    }

    prefix = format_prompt(space.prefix, prompt_addition_info)
    suffix = format_prompt(space.suffix, prompt_addition_info)

    token_count_func = get_token_count_func()
    prefix_token = token_count_func(prefix)
    suffix_token = token_count_func(suffix)

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
        - RELAX_TOKEN,
        get_text_length=token_count_func,
    )

    dynamic_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=[],
    )
    prompt = dynamic_prompt.format()

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


def format_prompt(prompt: str, prompt_optional_info: Dict[str, str]):
    fs = [f for f in PROMPT_FORMATS if re.search("(?<!{)\{" + f + "\}(?!})", prompt)]
    for f in fs:
        prompt = re.sub("(?<!{)\{" + f + "\}(?!})", prompt_optional_info[f], prompt)
    return prompt
