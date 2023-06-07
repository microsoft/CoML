import numpy as np
import pandas as pd
import pytest
from peewee import fn

from mlcopilot.constants import TOP_K
from mlcopilot.experience import (
    _ingest_solution,
    _ingest_space,
    _ingest_task,
    canonicalize_config,
    gen_experience,
    gen_experience_per_task,
    get_quantile_stat,
    ingest_experience,
)
from mlcopilot.orm import Solution, Task
from mlcopilot.utils import set_llms

from .llm import MockEmbeddingModel, MockKnowledgeLLM

history_df = pd.read_csv("assets/example_history.csv")


def test_get_quantile_stat():
    experience_df = history_df[
        history_df[["TASK_ID", "SCORE"]]
        .groupby("TASK_ID")
        .rank(method="first", ascending=False)["SCORE"]
        <= TOP_K
    ]
    quantile_info = get_quantile_stat(experience_df)
    assert quantile_info == {
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
    return quantile_info


def test_ingest_space():
    space_id = "__test_space__"
    space_desc = "This is space description"
    quantile_info = test_get_quantile_stat()
    space = _ingest_space(space_id, space_desc, quantile_info)
    assert space is not None
    return space, quantile_info


def test_ingest_task():
    set_llms(embedding_model=MockEmbeddingModel)
    task_desc = {k: "." for k in history_df["TASK_ID"].unique()}
    _ingest_task(history_df, task_desc)


def test_canonicalize_config():
    config_term = {
        "cost": 62.72359795339274,
        "gamma": 0.0051534831825209,
        "kernel": "radial",
        "degree": np.nan,
    }
    _, quantile_info = test_ingest_space()
    demo = canonicalize_config(config_term, quantile_info)
    assert (
        demo
        == "cost is large. gamma is very small. kernel is radial. degree is very small."
    )
    return demo


def test_ingest_solution():
    space, _ = test_ingest_space()
    _ingest_solution(history_df, space)


def test_ingest_experience():
    space_desc = "This is space description"
    space_id = "__test_space__"
    space = ingest_experience(history_df, None, space_desc, space_id)
    assert space is not None
    return space


def test_gen_experience_per_task():
    space, _ = test_ingest_space()
    test_ingest_task()
    task = Task.get(Task.task_id == "43")
    experience_per_task = gen_experience_per_task(space, task)
    assert isinstance(experience_per_task, str)
    return experience_per_task


def test_gen_experience():
    task_desc = "test task description"
    space, _ = test_ingest_space()

    test_ingest_task()

    tasks_select = (
        Task.select()
        .join(Solution)
        .where(Solution.space == space)
        .distinct()
        .order_by(fn.cosine_similarity(task_desc, Task.embedding).desc())
    )  # TODO SQL groupby
    examples_gt = [gen_experience_per_task(space, task) for task in tasks_select]
    examples = gen_experience(space, task_desc)
    assert examples == examples_gt
