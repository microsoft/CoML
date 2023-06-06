from test.llm import MockEmbeddingModel

import pytest
from peewee import fn

from mlcopilot.orm import Knowledge, Solution, Space, Task, database_proxy, import_db
from mlcopilot.space import create_space, delete_space, list_available_spaces
from mlcopilot.utils import set_llms


def test_import_db():
    space_id = "__test_space__"
    delete_space(space_id)
    import_db("assets/mlcopilot.db")
    for table in [Space, Task, Solution, Knowledge]:
        assert table.select().count() > 0
    database_proxy.close()
    for space in list_available_spaces():
        delete_space(space.space_id)


def test_cosine_similarity():
    set_llms(embedding_model=MockEmbeddingModel)
    space_id = "__test_space__"
    delete_space(space_id)
    space = create_space(
        space_id=space_id,
        history="assets/example_history.csv",
        task_desc="assets/example_descriptions.json",
        no_knowledge=True,
    )
    unordered_tasks = list(
        Task.select().join(Solution).where(Solution.space == space).distinct()
    )
    task = unordered_tasks[1]
    assert (
        task.desc
        == list(
            Task.select()
            .join(Solution)
            .where(Solution.space == space)
            .distinct()
            .order_by(fn.cosine_similarity(task.desc, Task.embedding).desc())
        )[0].desc
    )
