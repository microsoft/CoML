from typing import Any, Dict, List

import numpy as np
from peewee import (
    AutoField,
    BlobField,
    CompositeKey,
    DatabaseProxy,
    Expression,
    Field,
    FloatField,
    ForeignKeyField,
    Model,
    ModelBase,
    PrimaryKeyField,
    TextField,
    Value,
    fn,
)

try:
    from pgvector.psycopg2 import VectorAdapter, register_vector
    from pgvector.utils import from_db, to_db
except ImportError:
    from_db = to_db = None

from .constants import *
from .utils import get_llm


class ArrayField(BlobField):
    field_type = "BLOB"

    def db_value(self, value):
        if isinstance(value, np.ndarray):
            value = value.tobytes()
        return super().db_value(value)

    def python_value(self, value):
        return np.frombuffer(value, dtype=np.float32)

    def cosine_distance(self, text: str):
        return fn.cosine_similarity(self, text).desc()


class VectorField(Field):
    field_type = "VECTOR"

    def __init__(self, dim=None, *args, **kwargs):
        self.dim = dim
        super().__init__(*args, **kwargs)

    def get_modifiers(self) -> None:
        return [self.dim]

    def db_value(self, value):
        if isinstance(value, str):
            return value
        elif isinstance(value, bytes):
            value = np.frombuffer(value, dtype=np.float32)
        return to_db(value, self.dim)

    def python_value(self, value):
        return from_db(value)

    def cosine_distance(self, text: str):
        text_emb = np.asarray(
            get_llm("embedding")().embed_query(text), dtype=np.float32
        )
        return Expression(self, "<=>", Value(to_db(text_emb, self.dim), unpack=False))


database_proxy = DatabaseProxy()

if COML_DB_BACKEND == "sqlite":
    from peewee import SqliteDatabase

    init_db_func = lambda: SqliteDatabase(COML_DB_PATH)
elif COML_DB_BACKEND == "postgres":
    from peewee import PostgresqlDatabase

    init_db_func = lambda: PostgresqlDatabase(
        COML_DB_NAME,
        host=COML_DB_HOST,
        port=COML_DB_PORT,
        user=COML_DB_USER,
        password=COML_DB_PASSWORD,
    )
else:
    raise NotImplementedError(f"COML_DB_BACKEND {COML_DB_BACKEND} not supported.")


def init_db():
    database_proxy.initialize(init_db_func())
    conn = database_proxy.connection()
    if COML_DB_BACKEND == "postgres":
        register_vector(conn)
    database_proxy.create_tables([Space, Task, Solution, Knowledge])

    if COML_DB_BACKEND == "sqlite":
        _cache = {}

        @database_proxy.func()
        def cosine_similarity(task_emb: BlobField, text: str) -> float:
            emb = np.frombuffer(task_emb, dtype=np.float32)
            if text not in _cache:
                _cache[text] = np.asarray(
                    get_llm("embedding")().embed_query(text), dtype=np.float32
                )
            text_emb = _cache[text]
            return np.dot(emb, text_emb).item()


class BaseModel(Model):
    class Meta:
        database = database_proxy


class Space(BaseModel):
    space_id: str = TextField(primary_key=True)
    desc = TextField()
    quantile_info = BlobField(null=True)
    prefix = TextField(default=DEFAULT_PROMPT_PREFIX)
    suffix = TextField(default=DEFAULT_PROMPT_SUFFIX)


class Task(BaseModel):
    task_id: str = TextField(primary_key=True)
    embedding = ArrayField() if COML_DB_BACKEND == "sqlite" else VectorField(EMBED_DIM)
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
    space = ForeignKeyField(Space, backref="knowledge")
    task = ForeignKeyField(Task, backref="knowledge", null=True)


def import_db(tables: Dict[ModelBase, List[Dict[str, Any]]]) -> None:
    """
    Imports the contents of the database from a dictionary.

    Parameters:
    -----------
    tables: Dict[BaseModel, List[Dict[str, Any]]]
        A dictionary with the tables as keys and a list of records as values.

    Returns:
    --------
    None
    """
    for table, records in tables.items():
        # skip duplicate records
        with database_proxy.atomic():
            for record in records:
                try:
                    eval(table).insert(record).execute()
                except:
                    pass


init_db()
