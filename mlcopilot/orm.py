from typing import Any, Dict, List

import numpy as np
from peewee import (
    AutoField,
    BlobField,
    CompositeKey,
    FloatField,
    ForeignKeyField,
    Model,
    PrimaryKeyField,
    Proxy,
    TextField,
)
from playhouse.sqlite_ext import SqliteExtDatabase

from mlcopilot.constants import MLCOPILOT_DB_PATH
from mlcopilot.utils import get_llm

database_proxy = Proxy()


def init_db():
    database_proxy.initialize(SqliteExtDatabase(MLCOPILOT_DB_PATH))
    database_proxy.create_tables([Space, Task, Solution, Knowledge])

    @database_proxy.func()
    def cosine_similarity(text: str, task_emb: BlobField) -> float:
        emb = np.frombuffer(task_emb, dtype=np.float32)
        text_emb = np.asarray(
            get_llm("embedding")().embed_query(text), dtype=np.float32
        )
        return np.dot(emb, text_emb).item()


class BaseModel(Model):
    class Meta:
        database = database_proxy


class Space(BaseModel):
    space_id: str = TextField(primary_key=True)
    desc = TextField()
    quantile_info = BlobField()


class Task(BaseModel):
    task_id: str = TextField(primary_key=True)
    embedding = BlobField()
    desc = TextField()
    row_desc = TextField()


class Solution(BaseModel):
    demo = TextField()
    task = ForeignKeyField(Task, backref="solutions")
    space = ForeignKeyField(Space, backref="solutions")
    metric = FloatField()
    row_config = TextField()
    extra_metric = TextField()

    class Meta:
        primary_key = CompositeKey("space", "task", "row_config")
        database = database_proxy


class Knowledge(BaseModel):
    knowledge = TextField()
    space = ForeignKeyField(Space, backref="knowledge", primary_key=True)


def extract_tables() -> Dict[BaseModel, List[Dict[str, Any]]]:
    """
    Extracts the contents of the database into a dictionary.

    Returns:
    --------
    Dict[BaseModel, List[Dict[str, Any]]]
        A dictionary with the tables as keys and a list of records as values.
    """
    tables = {
        Space: [],
        Task: [],
        Solution: [],
        Knowledge: [],
    }
    for table in tables:
        for record in table.select():
            tables[table].append(record.__data__)
    return tables


def import_db(imported_db_path: str) -> None:
    """
    Import database from a given path to the default database path.

    Parameters
    ----------
    db_path: str
        The path to the database.

    Returns
    -------
    None
    """
    database_proxy.close()
    imported_db = SqliteExtDatabase(imported_db_path)
    database_proxy.initialize(imported_db)
    database_proxy.connect()
    database_proxy.create_tables([Space, Task, Solution, Knowledge])
    tables = extract_tables()
    database_proxy.close()
    init_db()
    for table, records in tables.items():
        # skip duplicate records
        with database_proxy.atomic():
            for record in records:
                try:
                    table.insert(record).execute()
                except:
                    pass


init_db()
