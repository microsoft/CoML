from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import langchain
import numpy as np
import orjson
import pandas as pd
from langchain.cache import InMemoryCache
from peewee import ModelSelect, fn

from .constants import *
from .orm import Knowledge, Solution, Space, Task, database_proxy
from .utils import format_config, get_llm

SAVE_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS


langchain.llm_cache = InMemoryCache()


def ingest_experience(
    history_df: pd.DataFrame,
    task_desc: Optional[Dict[str, str]],
    space_desc: str,
    space_id: str,
) -> Space:
    """Ingest experience from history dataframe.

    Parameters
    ----------
    history_df: pandas.DataFrame
        The history of configurations.
    task_desc
        The task descriptions.
    space_desc
        The space description.
    space_id
        The space id.

    Returns
    -------
    None

    Notes
    -----
    The history_df should be a dataframe with the following columns:
        - CONFIG_0
        - CONFIG_1
        - ...
        - CONFIG_N
        - METRIC
    The task_desc should be a dict with the following format:
        {
            "task_id_0": "task_description_0",
            "task_id_1": "task_description_1",
            ...
            "task_id_N": "task_description_N"
        }
    """
    history_df = history_df.drop_duplicates()
    history_df["TASK_ID"] = history_df["TASK_ID"].astype(str)
    if task_desc is None:
        task_desc = {k: "." for k in history_df["TASK_ID"].unique()}
    experience_df = history_df[
        history_df[["TASK_ID", "SCORE"]]
        .groupby("TASK_ID")
        .rank(method="first", ascending=False)["SCORE"]
        <= TOP_K
    ]
    quantile_info = get_quantile_stat(experience_df)

    space = _ingest_space(space_id, space_desc, quantile_info)

    _ingest_task(history_df, task_desc)

    _ingest_solution(history_df, space)

    print("Ingested experience into database.")
    # save db
    database_proxy.commit()
    return space


def _ingest_space(
    space_id: str, space_desc: str, quantile_info: Dict[str, List[float]]
) -> Space:
    with database_proxy.atomic():
        try:
            space = Space.get(Space.space_id == space_id)
            print(f"Space {space_id} already exists, skip ingestion.")
            return space
        except:
            space = Space.create(
                space_id=space_id,
                desc=space_desc,
                quantile_info=orjson.dumps(quantile_info, option=SAVE_OPTIONS),
            )
            print("Ingested space into database.")
    # save db
    database_proxy.commit()
    return space


def _ingest_task(history_df: pd.DataFrame, row_task_desc: Dict[str, str]) -> None:
    embeddings = get_llm("embedding")()
    with database_proxy.atomic():
        for task_id in history_df["TASK_ID"].unique():
            try:
                Task.get(Task.task_id == task_id)
                print(f"Task {task_id} already exists, skip ingestion.")
                continue
            except Task.DoesNotExist:
                task_desc = canonicalize_task(row_task_desc[task_id])
                embedding = np.asarray(
                    embeddings.embed_query(task_desc), dtype=np.float32
                ).tobytes()
                Task.create(
                    task_id=task_id,
                    embedding=embedding,
                    desc=task_desc,
                    row_desc=row_task_desc[task_id],
                )
                print("Ingested task into database.")
    # save db
    database_proxy.commit()


def _ingest_solution(history_df: pd.DataFrame, space: Space) -> None:
    with database_proxy.atomic():
        solutions = []

        for _, row in (
            history_df.groupby("TASK_ID")
            .apply(lambda x: x.sort_values("SCORE", ascending=False))
            .reset_index(drop=True)
            .iterrows()
        ):
            row_config_dict = {
                k[7:]: v for k, v in row.to_dict().items() if k.startswith("CONFIG_")
            }
            row_config = orjson.dumps((row_config_dict), option=SAVE_OPTIONS)
            try:
                Solution.get(
                    Solution.task == row["TASK_ID"],
                    Solution.space == space.space_id,
                    Solution.row_config == row_config,
                )
                print(
                    f"Solution '{row_config}' for space {space.space_id} and task {row['TASK_ID']} already exists, skip ingestion."
                )
                continue
            except Solution.DoesNotExist:
                solutions.append(
                    {
                        "task": row["TASK_ID"],
                        "space": space,
                        "metric": row["SCORE"],
                        "row_config": row_config,
                        "extra_metric": row.get("EXTRA_METRIC", ""),
                        "demo": canonicalize_config(
                            row_config_dict, orjson.loads(space.quantile_info)
                        ),
                    }
                )
        Solution.insert_many(solutions).execute()
        print("Ingested solution into database.")
    # save db
    database_proxy.commit()


def canonicalize_task(row_task_desc: str) -> str:
    """
    Canonicalize task to a string.

    Parameters
    ----------
    row_task_desc: str
        The task description.

    Returns
    -------
    str
        The canonicalized task description.
    """
    task_desc = f"Task: {row_task_desc}"
    return task_desc


def canonicalize_config(
    config_term: Dict[str, Any], quantile_info: Dict[str, List[float]]
) -> str:
    """
    Canonicalize configuration to a string.

    Parameters
    ----------
    config_term
        The configuration term.
    quantile_info
        The meta train info for stat. Record quantile information for each hyperparameter.

    Returns
    -------
    str
        The canonicalized configuration.
    """
    demo = format_config(config_term, quantile_info, bin_map=bin_map)
    return demo


def get_quantile_stat(experience_df: pd.DataFrame) -> Dict[str, List[float]]:
    """
    Get quantile stat from experience dataframe.

    Parameters
    ----------
    experience_df: pandas.DataFrame
        The experience dataframe.

    Returns
    -------
    Dict[str, List[float]]
        The quantile stat.
    """
    # map column name 'CONFIG_{}' to '{}', rest of columns are deleted
    meta_train_df_for_stat = experience_df.rename(
        columns=lambda x: x[7:] if x.startswith("CONFIG_") else None
    )
    del meta_train_df_for_stat[None]

    try:
        quantile_info = {
            col: np.quantile(
                meta_train_df_for_stat[col][~meta_train_df_for_stat[col].isna()],
                list(bin_map.keys()),
                method="nearest",
            )
            .astype(np.float32)
            .tolist()
            for col in meta_train_df_for_stat.select_dtypes(include=[np.number])
        }
    except:
        quantile_info = {
            col: np.quantile(
                meta_train_df_for_stat[col][~meta_train_df_for_stat[col].isna()],
                list(bin_map.keys()),
                interpolation="nearest",
            )
            .astype(np.float32)
            .tolist()
            for col in meta_train_df_for_stat.select_dtypes(include=[np.number])
        }

    return quantile_info


def gen_experience_per_task(space: Space, task: Task) -> str:
    """
    Generate experience content from space and task.

    Parameters
    ----------
    task_desc: str
        The task description.
    demos: str
        The demos.
        example:
        Configurations 1: ...
        Configurations 2: ...
        Configurations 3: ...

    Returns
    -------
    str
        The experience content.
    """
    demos = _gen_experience_demos(space, task)
    content = f"{task.desc}\n{demos}"
    return content


def _gen_experience_demos(space: Space, task: Task) -> str:
    """
    Generate experience demonstrations

    Parameters
    ----------
    space: Space
        The space.
    task: Task
        The task.

    Returns
    -------
    str
        The experience demonstrations per task.
    """
    solutions = (
        Solution.select()
        .where(Solution.task_id == task.task_id, Solution.space_id == space.space_id)
        .order_by(Solution.metric.desc())
        .limit(TOP_K)
    )
    demos = "\n".join(
        [
            f"Configuration {i+1}: {solution.demo}"
            for i, solution in enumerate(solutions)
        ]
    )
    return demos


def _get_best_relevant_solutions(space: Space, task_desc: str) -> ModelSelect:
    """
    Get the best relevant solution for a task.
    The relevance is measured by cosine similarity between task description embeddings, which affects the order of results.

    Parameters
    ----------
    space: Space
        The space.
    task_desc: str
        The task description.

    Returns
    -------
    ModelSelect
        The best relevant solution.
    """
    SolutionAlias = Solution.alias()
    order_key = Task.embedding.cosine_distance(task_desc)
    subquery = (
        SolutionAlias.select(
            SolutionAlias.demo,
            Task.task_id,
            Task.desc,
            Task.embedding,
            fn.ROW_NUMBER()
            .over(
                partition_by=[SolutionAlias.space, SolutionAlias.task],
                order_by=[SolutionAlias.metric.desc()],
            )
            .alias("rnk"),
        )
        .where(SolutionAlias.space == space)
        .join(Task, on=(SolutionAlias.task == Task.task_id))
        .order_by(order_key)
        .alias("subq")
    )
    query = (
        Solution.select(subquery.c.task_id, subquery.c.demo, subquery.c.desc)
        .from_(subquery)
        .where(subquery.c.rnk <= TOP_K)
    )
    return query


def _get_best_solutions(space: Space) -> ModelSelect:
    """
    Get the best solution for each task.

    Parameters
    ----------
    space: Space
        The space.

    Returns
    -------
    ModelSelect
        The best solution for each task.
    """
    SolutionAlias = Solution.alias()
    subquery = (
        SolutionAlias.select(
            SolutionAlias.demo,
            Task.task_id,
            Task.desc,
            Task.embedding,
            fn.ROW_NUMBER()
            .over(
                partition_by=[SolutionAlias.space, SolutionAlias.task],
                order_by=[SolutionAlias.metric.desc()],
            )
            .alias("rnk"),
        )
        .where(SolutionAlias.space == space)
        .join(Task, on=(SolutionAlias.task == Task.task_id))
        .alias("subq")
    )
    query = (
        Solution.select(subquery.c.task_id, subquery.c.demo, subquery.c.desc)
        .from_(subquery)
        .where(subquery.c.rnk <= TOP_K)
    )
    return query


def gen_experience(
    space: Space, task_desc: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Generate experience content from space and optional task description.

    Parameters
    ----------
    space: Space
        The space.
    task_desc
        The task description.

    Returns
    -------
    List[str]
        The experience content.
    """
    if task_desc is None:
        query = _get_best_solutions(space)
    else:
        query = _get_best_relevant_solutions(space, task_desc)
    examples = OrderedDict()

    for solution in query:
        if solution.task_id not in examples:
            examples[solution.task_id] = [solution.desc]
        examples[solution.task_id].append(
            f"Configuration {len(examples[solution.task_id])}: {solution.demo}"
        )
    return list(examples.keys()), ["\n".join(e) for e in examples.values()]
