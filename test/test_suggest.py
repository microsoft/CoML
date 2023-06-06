import orjson
import pytest

from mlcopilot.constants import TOP_K, inverse_bin_map
from mlcopilot.orm import Knowledge
from mlcopilot.space import create_space, delete_space
from mlcopilot.suggest import _get_prompt_with_zero_shot, suggest
from mlcopilot.utils import parse_configs, set_llms

from .llm import MockEmbeddingModel, MockKnowledgeLLM, MockSuggestLLM


def _create_space():
    set_llms(knowledge_model=MockKnowledgeLLM, embedding_model=MockEmbeddingModel)
    space_id = "__test_space__"
    space = create_space(
        space_id=space_id,
        history="assets/example_history.csv",
        task_desc="assets/example_descriptions.json",
        no_knowledge=True,
    )
    return space


def test_suggest_with_few_shot_no_knowledge():
    space = _create_space()
    set_llms(suggest_model=MockSuggestLLM, embedding_model=MockEmbeddingModel)
    suggest_configs, knowledge = suggest(space, "")
    delete_space(space.space_id)
    assert len(suggest_configs) == 3
    assert knowledge is None


def test_suggest_with_few_shot_with_knowledge():
    space = _create_space()
    knowledge_ = f"\n1.{MockKnowledgeLLM()('')}"
    try:
        Knowledge.create(space_id=space.space_id, knowledge=knowledge_)
    except:
        pass
    set_llms(suggest_model=MockSuggestLLM, embedding_model=MockEmbeddingModel)
    suggest_configs, knowledge = suggest(space, "")
    delete_space(space.space_id)
    assert len(suggest_configs) == 3
    assert knowledge == knowledge_


def test_suggest_with_zero_shot():
    space = _create_space()
    set_llms(suggest_model=MockSuggestLLM, embedding_model=MockEmbeddingModel)
    task_desc = "Task: New task description."
    llm = MockSuggestLLM()
    quantile_infos = orjson.loads(space.quantile_info)
    prompt = _get_prompt_with_zero_shot(space, task_desc)

    response = llm(prompt)
    suggest_configs = parse_configs(
        response,
        TOP_K,
        inverse_bin_map,
        quantile_infos,
    )
    delete_space(space.space_id)
    assert len(suggest_configs) == 3