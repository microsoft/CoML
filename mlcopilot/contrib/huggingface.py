import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from test.llm import MockEmbeddingModel
from typing import List, Optional

import numpy as np
import orjson
import pandas as pd

from mlcopilot.experience import SAVE_OPTIONS, canonicalize_task
from mlcopilot.knowledge import get_knowledge
from mlcopilot.orm import Knowledge, Solution, Space, Task, database_proxy
from mlcopilot.utils import get_llm, set_llms


def gen_space_description(
    history_df: pd.DataFrame, space_desc: Optional[str] = None
) -> str:
    """
    Generate space description from history_df and space_desc.

    Parameters
    ----------
    history_df: pandas.DataFrame
        The history of configurations.
    space_desc: str | None
        The path to the space description.

    Returns
    -------
    str
        The generated space description.
    """
    history_df = history_df.rename(
        columns=lambda x: x[7:] if x.startswith("CONFIG_") else None
    )
    del history_df[None]
    history_df.columns
    descriptions = """Space has {} configurable hyper-parameters, i.e., {}.\n""".format(
        len(history_df.columns),
        ", ".join(["'{}'".format(x) for x in history_df.columns]),
    )
    return descriptions + (space_desc if space_desc is not None else "")


def create_tables():
    with database_proxy:
        database_proxy.create_tables([Task, Solution, Space, Knowledge])


def drop_tables():
    with database_proxy:
        database_proxy.drop_tables([Task, Solution, Space, Knowledge])


def gen_data_json(jsons):
    ret = {k: {} for k in jsons}
    for task in jsons:
        l = jsons[task]
        datas = Counter()
        for v in l:
            datas.update(v["datasets"].keys())
        # find the count >=5 datasets
        for dataset, count in datas.items():
            if count >= 3:
                ret[task][dataset] = []
        for v in l:
            for x in v["datasets"]:
                if x in ret[task]:
                    ret[task][x].append(v)
    return {k: v for k, v in ret.items() if len(v) > 0}


def create_model_space(
    space_id: str,
    history: str,
    task_desc: Optional[str] = None,
    space_desc: Optional[str] = None,
    knowledge: Optional[str] = None,
) -> Space:
    """
    Create a space from history csv file and task description json file.

    Parameters
    ----------
    space_id: str
        The ID of the space to identify the space.
    history: str
        The path to the history of configurations. A csv file, format see `mlcopilot::experience::ingest_experience`.
    task_desc: str
        The JSON path to the task description. A json file, format see `mlcopilot::experience::ingest_experience`.
    space_desc: str
        The text path to the space description. Optional.
    no_knowledge: bool
        Whether to generate knowledge from history.

    Returns
    -------
    None

    Examples
    --------
    >>> create_space(
    ...     space_id="example_space",
    ...     history="assets/example_history.csv",
    ...     task_desc="assets/example_task_description.json",
    ...     space_desc="assets/example_space_desc.txt",
    ... )
    """
    knowledge = pd.read_csv(knowledge, index_col=[0, 1])
    all_datas = []
    task_to_json = defaultdict(list)
    for file in glob.iglob(
        "assets/private/model_cards_1400/*.json"
    ):
        with open(file, "r") as f:
            data = json.load(f)
            all_datas.append(data)
        task_to_json[data["task"]].append(data)

    with open("assets/dataset_desc.json") as f:
        dataset_desc = json.load(f)

    task_dataset_to_json_ = gen_data_json(task_to_json)

    with open(task_desc) as f:
        tasks_from_huggingface = json.load(f)
    task_descs = {}
    task_dataset_to_json = defaultdict(dict)
    for task in task_dataset_to_json_:
        if not task_dataset_to_json_[task]:
            continue
        for dataset in task_dataset_to_json_[task]:
            if dataset not in dataset_desc:
                continue
            # resp = gen_knowledge_per_task_v3(task, dataset)
            task_dataset_to_json[task][dataset] = task_dataset_to_json_[task][dataset]
            task_descs[f"{task}_{dataset}"] = (
                f"The task is {task}. "
                f"Task description: {tasks_from_huggingface[task][0]}. "
                f"The dataset is {dataset}. "
                f"Dataset description: {dataset_desc[dataset]}. "
            )

    with database_proxy.atomic():
        try:
            space = Space.get(Space.space_id == space_id)
            print(f"Space {space_id} already exists, skip ingestion.")
            return space
        except:
            space = Space.create(
                space_id=space_id,
                desc=space_desc,
            )
            print("Ingested space into database.")
    # save db
    database_proxy.commit()

    embeddings = get_llm("embedding")()
    with database_proxy.atomic():
        for task_id in task_descs:
            try:
                Task.get(Task.task_id == task_id)
                print(f"Task {task_id} already exists, skip ingestion.")
                continue
            except Task.DoesNotExist:
                task_desc = canonicalize_task(task_descs[task_id])
                embedding = np.asarray(
                    embeddings.embed_query(task_desc), dtype=np.float32
                ).tobytes()
                Task.create(
                    task_id=task_id,
                    embedding=embedding,
                    desc=task_desc,
                    row_desc=task_descs[task_id],
                )
                print("Ingested task into database.")
    # save db
    database_proxy.commit()

    solutions = []

    for task_ in task_dataset_to_json:
        for dataset in task_dataset_to_json[task_]:
            task = f"{task_}_{dataset}"
            for j in task_dataset_to_json[task_][dataset]:
                row_config_dict = j["model"]
                row_config = orjson.dumps((row_config_dict), option=SAVE_OPTIONS)
                try:
                    Solution.get(
                        Solution.task == task,
                        Solution.space == space.space_id,
                        Solution.row_config == row_config,
                    )
                    print(
                        f"Solution '{row_config}' for space {space.space_id} and task {task} already exists, skip ingestion."
                    )
                    continue
                except Solution.DoesNotExist:
                    solutions.append(
                        {
                            "task": task,
                            "space": space,
                            "row_config": row_config,
                            "extra_metric": j["metrics"],
                            "demo": json.dumps(row_config_dict),
                        }
                    )
            if (
                get_knowledge(space, task) is None
                and (task_, dataset) in knowledge.index
            ):
                with database_proxy.atomic():
                    Knowledge.create(
                        space_id=space.space_id,
                        task=task,
                        knowledge=knowledge.loc[(task_, dataset)].values.item(),
                    )
    with database_proxy.atomic():
        Solution.insert_many(solutions).execute()
        print("Ingested solution into database.")
    # save db
    database_proxy.commit()
    return space


def create_hp_space(
    space_id: str,
    history: str,
    task_desc: Optional[str] = None,
    space_desc: Optional[str] = None,
    knowledge: Optional[str] = None,
) -> Space:
    """
    Create a space from history csv file and task description json file.

    Parameters
    ----------
    space_id: str
        The ID of the space to identify the space.
    history: str
        The path to the history of configurations. A csv file, format see `mlcopilot::experience::ingest_experience`.
    task_desc: str
        The JSON path to the task description. A json file, format see `mlcopilot::experience::ingest_experience`.
    space_desc: str
        The text path to the space description. Optional.
    no_knowledge: bool
        Whether to generate knowledge from history.

    Returns
    -------
    None

    Examples
    --------
    >>> create_space(
    ...     space_id="example_space",
    ...     history="assets/example_history.csv",
    ...     task_desc="assets/example_task_description.json",
    ...     space_desc="assets/example_space_desc.txt",
    ... )
    """
    knowledge = pd.read_csv(knowledge, index_col=[0, 1])
    all_datas = []
    task_to_json = defaultdict(list)
    for file in glob.iglob(
        "/home/v-leizhang3/project/huggingface/results_4_refine_data/*.json"
    ):
        with open(file, "r") as f:
            data = json.load(f)
            all_datas.append(data)
        task_to_json[data["task"]].append(data)

    with open("assets/dataset_desc.json") as f:
        dataset_desc = json.load(f)

    task_dataset_to_json_ = gen_data_json(task_to_json)

    with open(task_desc) as f:
        tasks_from_huggingface = json.load(f)
    task_descs = {}
    task_dataset_to_json = defaultdict(dict)
    for task in task_dataset_to_json_:
        if not task_dataset_to_json_[task]:
            continue
        for dataset in task_dataset_to_json_[task]:
            if dataset not in dataset_desc:
                continue
            # resp = gen_knowledge_per_task_v3(task, dataset)
            task_dataset_to_json[task][dataset] = task_dataset_to_json_[task][dataset]
            task_descs[f"{task}_{dataset}"] = (
                f"The task is {task}. "
                f"Task description: {tasks_from_huggingface[task][0]}. "
                f"The dataset is {dataset}. "
                f"Dataset description: {dataset_desc[dataset]}. "
            )

    with database_proxy.atomic():
        try:
            space = Space.get(Space.space_id == space_id)
            print(f"Space {space_id} already exists, skip ingestion.")
            return space
        except:
            space = Space.create(
                space_id=space_id,
                desc=space_desc,
            )
            print("Ingested space into database.")
    # save db
    database_proxy.commit()

    embeddings = get_llm("embedding")()
    with database_proxy.atomic():
        for task_id in task_descs:
            try:
                Task.get(Task.task_id == task_id)
                print(f"Task {task_id} already exists, skip ingestion.")
                continue
            except Task.DoesNotExist:
                task_desc = canonicalize_task(task_descs[task_id])
                embedding = np.asarray(
                    embeddings.embed_query(task_desc), dtype=np.float32
                ).tobytes()
                Task.create(
                    task_id=task_id,
                    embedding=embedding,
                    desc=task_desc,
                    row_desc=task_descs[task_id],
                )
                print("Ingested task into database.")
    # save db
    database_proxy.commit()

    solutions = []

    for task_ in task_dataset_to_json:
        for dataset in task_dataset_to_json[task_]:
            task = f"{task_}_{dataset}"
            for j in task_dataset_to_json[task_][dataset]:
                row_config_dict = {
                    "model": j["model"],
                    "hyperparameters": j["hyperparameters"],
                }
                row_config = orjson.dumps((row_config_dict), option=SAVE_OPTIONS)
                try:
                    Solution.get(
                        Solution.task == task,
                        Solution.space == space.space_id,
                        Solution.row_config == row_config,
                    )
                    print(
                        f"Solution '{row_config}' for space {space.space_id} and task {task} already exists, skip ingestion."
                    )
                    continue
                except Solution.DoesNotExist:
                    solutions.append(
                        {
                            "task": task,
                            "space": space,
                            "row_config": row_config,
                            "extra_metric": j["metrics"],
                            "demo": json.dumps(row_config_dict),
                        }
                    )
            if (
                get_knowledge(space, task) is None
                and (task_, dataset) in knowledge.index
            ):
                with database_proxy.atomic():
                    Knowledge.create(
                        space_id=space.space_id,
                        task=task,
                        knowledge=knowledge.loc[(task_, dataset)].values.item(),
                    )
    with database_proxy.atomic():
        Solution.insert_many(solutions).execute()
        print("Ingested solution into database.")
    # save db
    database_proxy.commit()
    return space


def list_available_spaces() -> List[Space]:
    """
    List all available spaces.

    Returns
    -------
    list
        A list of available spaces.
    """
    return list(Space.select())


def print_space() -> list:
    """
    Print all available spaces to the console.

    Returns
    -------
    list
        A list of available spaces.
    """
    print("Current space available: ")
    available_spaces = list_available_spaces()
    for i, space in enumerate(available_spaces):
        print(f"{i+1}. Design space: (Space ID = {space.space_id}){space.desc}" "\n\n")
    return available_spaces


def delete_space(space_id: str):
    """
    Delete a space.

    Parameters
    ----------
    space_id: str
        The ID of the space to delete.

    Returns
    -------
    None
    """
    space = import_space(space_id)
    if space is None:
        print(f"Space {space_id} does not exist.")
        return
    else:
        print(f"Deleting space {space_id}...")

    Solution.delete().where(Solution.space == space.space_id).execute()
    Knowledge.delete().where(Knowledge.space == space.space_id).execute()
    # delete task if no other space is using it
    for task in Task.select():
        if Solution.select().where(Solution.task == task.task_id).count() == 0:
            task.delete_instance()

    space.delete_instance()
    print(f"Space {space_id} deleted.")


def import_space(space_id: str) -> Space:
    """
    Import a space.

    Parameters
    ----------
    space_id: str
        The ID of the space to import.

    Returns
    -------
    Space
        The imported space.
    """
    try:
        space = Space.get(Space.space_id == space_id)
    except:
        space = None
    return space


if __name__ == "__main__":
    set_llms(embedding_model=MockEmbeddingModel)

    drop_tables()
    create_tables()
    create_model_space(
        space_id="model",
        history="assets/example_history.csv",
        task_desc="assets/tasks_from_huggingface.json",
        space_desc="",
        knowledge="assets/model_knowledge.csv",
    )

    create_hp_space(
        space_id="hp",
        history="assets/example_history.csv",
        task_desc="assets/tasks_from_huggingface.json",
        space_desc="",
        knowledge="assets/hp_knowledge.csv",
    )
