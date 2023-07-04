import numpy as np
import pandas as pd
import pytest
from peewee import JOIN, fn

from mlcopilot.constants import MLCOPILOT_DB_BACKEND, TOP_K
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
from mlcopilot.space import create_tables, drop_tables
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
    space = ingest_experience(
        history_df,
        {task_id: task_id for task_id in history_df.TASK_ID.unique().astype(str)},
        space_desc,
        space_id,
    )
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
    set_llms(embedding_model=MockEmbeddingModel)
    drop_tables()
    create_tables()
    task_desc = "test task description"
    space = test_ingest_experience()
    order_key = Task.embedding.cosine_distance(task_desc)
    subquery = (
        Task.select(Task.task_id)
        .join(Solution)
        .where(Solution.space == space)
        .distinct()
    )

    tasks_select = (
        Task.select()
        .join(subquery, JOIN.LEFT_OUTER, on=(Task.task_id == subquery.c.task_id))
        .order_by(order_key)
    )
    examples_gt = [gen_experience_per_task(space, task) for task in tasks_select]

    retrieved_tasks, examples = gen_experience(space, task_desc)

    assert all(examples[i][:10] == examples_gt[i][:10] for i in range(len(examples)))
