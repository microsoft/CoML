import pandas as pd
import pytest

from coml.space import (
    create_space,
    database_proxy,
    delete_space,
    drop_tables,
    gen_space_description,
    list_available_spaces,
    print_space,
)
from coml.utils import set_llms

from .llm import MockEmbeddingModel, MockKnowledgeLLM, MockSuggestLLM


def test_gen_space_description():
    space_desc = gen_space_description(
        pd.read_csv("assets/example_history.csv"),
    )
    assert space_desc == (
        "Space has 4 configurable hyper-parameters, i.e., 'cost', 'gamma', 'kernel', 'degree'.\n"
    )
    return space_desc


def test_delete_space():
    space_id = "__test_space__"
    delete_space(space_id)
    available_spaces = list_available_spaces()
    assert space_id not in (space.space_id for space in available_spaces)


def test_create_space():
    space_id = "__test_space__"
    delete_space(space_id)
    available_spaces = list_available_spaces()
    set_llms(knowledge_model=MockKnowledgeLLM)
    create_space(
        space_id=space_id,
        history="assets/example_history.csv",
        task_desc="assets/example_descriptions.json",
        no_knowledge=True,
    )
    assert len(list_available_spaces()) == 1 + len(available_spaces)
    assert space_id in (space.space_id for space in list_available_spaces())
    delete_space(space_id)
