import re

import pandas as pd
import pytest

from coml.knowledge import (
    gen_knowledge_candidate,
    post_validation,
    split_knowledge,
    suggest_with_knowledge,
)
from coml.space import create_space, delete_space
from coml.surrogate_utils import process_history_df, train_surrogate
from coml.utils import set_llms

from .llm import MockEmbeddingModel, MockKnowledgeLLM, MockSuggestLLM

examples = [
    "Task: .\n"
    "Configuration 1: cost is large. gamma is small. kernel is radial. degree is very small.\n"
    "Configuration 2: cost is very large. gamma is small. kernel is radial. degree is very small.\n"
    "Configuration 3: cost is medium. gamma is medium. kernel is radial. degree is very small."
]


def test_gen_knowledge_candidate():
    set_llms(knowledge_model=MockKnowledgeLLM)
    knowledge = gen_knowledge_candidate(examples)
    assert knowledge == "\n1.This is a mock knowledge."
    return knowledge


def test_eval_knowledge():
    knowledge_candidate = test_gen_knowledge_candidate()
    quantile_info = {
        "cost": [
            0.04495582729578018,
            2.966309070587158,
            8.799302101135254,
            53.32119369506836,
            223.55349731445312,
        ],
        "gamma": [
            0.006136692129075527,
            0.03009503148496151,
            0.1399584263563156,
            1.139962911605835,
            2.557131290435791,
        ],
        "degree": [2.0, 2.0, 3.0, 3.0, 3.0],
    }
    set_llms(suggest_model=MockSuggestLLM)
    suggest_configs = suggest_with_knowledge(
        examples, knowledge_candidate, examples[0], quantile_info
    )
    assert len(suggest_configs) == 3
    return suggest_configs


def test_post_validation():
    history_path = "assets/example_history.csv"
    space_id = "__test_space__"
    delete_space(space_id)
    space = create_space(
        space_id=space_id,
        history=history_path,
        task_desc="assets/example_descriptions.json",
        no_knowledge=True,
    )
    history_df = pd.read_csv(history_path)
    history_df_processed, config_names = process_history_df(history_df)
    surrogate = train_surrogate(history_df_processed)
    set_llms(suggest_model=MockSuggestLLM, knowledge_model=MockKnowledgeLLM)
    space_id = "__test_space__"
    knowledges = post_validation(space, surrogate, config_names)
    delete_space(space_id)
    assert set(knowledges) == set(split_knowledge(f"1. {MockKnowledgeLLM()('')}"))
    return knowledges
