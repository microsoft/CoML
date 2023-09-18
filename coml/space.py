import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .experience import ingest_experience
from .knowledge import get_knowledge
from .orm import Knowledge, Solution, Space, Task, database_proxy


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


def create_space(
    space_id: str,
    history: str,
    task_desc: Optional[str] = None,
    space_desc: Optional[str] = None,
    no_knowledge: bool = False,
) -> Space:
    """
    Create a space from history csv file and task description json file.

    Parameters
    ----------
    space_id: str
        The ID of the space to identify the space.
    history: str
        The path to the history of configurations. A csv file, format see `coml.experience.ingest_experience`.
    task_desc: str
        The JSON path to the task description. A json file, format see `coml.experience.ingest_experience`.
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
    history_df = pd.read_csv(history)
    if task_desc is not None:
        task_desc = json.loads(Path(task_desc).read_text())
    space_desc = Path(space_desc).read_text() if space_desc is not None else None
    space_desc = gen_space_description(history_df, space_desc)
    space = ingest_experience(history_df, task_desc, space_desc, space_id)

    if not no_knowledge and get_knowledge(space) == "":
        from .knowledge import post_validation
        from .surrogate_utils import process_history_df, train_surrogate

        history_df_processed, config_names = process_history_df(history_df)
        surrogate_fn = train_surrogate(history_df_processed)
        knowledges = post_validation(space, surrogate_fn, config_names)
        for knowledge in knowledges:
            Knowledge.create(space_id=space.space_id, knowledge=knowledge)
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


create_tables()
