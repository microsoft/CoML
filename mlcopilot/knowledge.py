import random
from typing import Any, Callable, Dict, List, Optional

import orjson
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

from mlcopilot.constants import *
from mlcopilot.constants import TOKEN_COMPLETION_LIMIT, TOKEN_LIMIT
from mlcopilot.experience import gen_experience
from mlcopilot.orm import Knowledge, Solution, Space, Task, database_proxy
from mlcopilot.surrogate_utils import evaluate_configs
from mlcopilot.utils import get_llm, get_token_count_func, parse_configs

prefix_sep = "__DUMM_SEP__"


def gen_knowledge_candidate(examples: List[str]) -> str:
    """
    Generate knowledge candidate from examples.

    Parameters
    ----------
    examples: list of strings
        The list of examples.

    Returns
    -------
    str
        The generated knowledge candidate.
    """
    prefix_token = get_token_count_func()(
        "Here are some tasks along with best hyper-parameter configurations to train a model on them.\n"
    )
    suffix_token = get_token_count_func()(
        "\nQ: From the examples above, what patterns can we observe about the relationship between dataset characteristics and the best hyper-parameter configurations? (Answer MUST be concise, critical, point-by-point, line-by-line, and brief. Only include relevant observations without unnecessary elaboration.)\n\nA: 1."
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

    dynamic_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Here are some tasks along with best hyper-parameter configurations to train a model on them.\n",
        suffix="\nQ: From the examples above, what patterns can we observe about the relationship between dataset characteristics and the best hyper-parameter configurations? (Answer MUST be concise, critical, point-by-point, line-by-line, and brief. Only include relevant observations without unnecessary elaboration.)\n\nA: 1.",
        input_variables=[],
    )
    llm = get_llm("knowledge")()
    knowledge = "\n1." + llm(dynamic_prompt.format())
    return knowledge


def suggest_with_knowledge(
    examples: List[str],
    knowledge: str,
    valid_example: str,
    quantile_infos: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    """
    Suggest configurations with knowledge.

    Parameters
    ----------
    examples
        The list of examples.
    knowledge: str
        The knowledge.
    valid_example: str
        The valid example.
    quantile_infos
        The meta train info for stats. Used to convert the text to config value.

    Returns
    -------
    List[Dict[str, Any]]
        The list of suggested configurations.
    """
    prefix_token = get_token_count_func()(
        "Here are some tasks along with best hyper-parameter configurations to train a model on them.\n"
    )
    suffix_token = get_token_count_func()(
        "\nGuidelines:{knowledge}\n\n\nBased on the examples and guidelines above, recommend {TOP_K} hyper-parameter configurations for a new classification dataset.\n\n{output}".format(
            knowledge=knowledge,
            TOP_K=str(TOP_K),
            output=(
                valid_example[: valid_example.index("\nConfiguration 1:")] + "\n\n"
            ),
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

    dynamic_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Here are some tasks along with best hyper-parameter configurations to train a model on them.\n",
        suffix="\nGuidelines:{knowledge}\n\n\nBased on the examples and guidelines above, recommend {TOP_K} hyper-parameter configurations for a new classification dataset.\n\n{output}",
        input_variables=["knowledge", "TOP_K", "output"],
    )

    llm = get_llm("suggest")()

    response = llm(
        dynamic_prompt.format(
            knowledge=knowledge,
            TOP_K=str(TOP_K),
            output=(
                valid_example[: valid_example.index("\nConfiguration 1:")] + "\n\n"
            ),
        )
    )
    suggest_configs = parse_configs(
        response,
        TOP_K,
        inverse_bin_map,
        quantile_infos,
    )
    return suggest_configs


def post_validation(
    space: Space, surrogate_fn: Callable, config_names: List[str]
) -> str:
    """
    Post validation to generate knowledge.

    Parameters
    ----------
    space: Space
        The space.
    surrogate_fn: Callable
        The surrogate function.
    config_names: list of str
        The list of configuration names.

    Returns
    -------
    str
        The generated knowledge.
    """
    knowledge = get_knowledge(space.space_id)
    if knowledge is not None:
        print("Knowledge already exists.")
        return knowledge
    quantile_infos = orjson.loads(space.quantile_info)
    retrieved_tasks, examples = gen_experience(space)
    best_score = float("-inf")
    knowledge = None
    for _ in range(3):
        random.shuffle(examples)
        knowledge_candidate = gen_knowledge_candidate(examples)
        score = 0
        for val_step in range(3):
            random.shuffle(examples)
            assert len(examples) > 1, "Not enough examples in the memory."
            valid_example = examples[-1]
            train_example = examples[:-1]

            try:
                suggest_configs = suggest_with_knowledge(
                    train_example,
                    knowledge_candidate,
                    valid_example,
                    quantile_infos,
                )
                _score = evaluate_configs(
                    surrogate_fn, suggest_configs, config_names
                ).mean()
            except:
                _score = float("-inf")
            score += _score
        if best_score < score:
            best_score = score
            knowledge = knowledge_candidate
    assert knowledge is not None, "Knowledge is not generated."

    return knowledge


def get_knowledge(space: Space, task=None):
    try:
        knowledge = Knowledge.get(
            Knowledge.space_id == space.space_id, Knowledge.task == task
        ).knowledge
        return knowledge
    except:
        return None
