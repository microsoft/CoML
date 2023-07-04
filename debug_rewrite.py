from test.llm import MockEmbeddingModel, MockKnowledgeLLM
from test.test_experience import test_ingest_experience

import numpy as np
import pandas as pd
# import pytest
from peewee import JOIN, fn

# from mlcopilot.constants import MLCOPILOT_DB_BACKEND, TOP_K
from mlcopilot.orm import Solution, Task
from mlcopilot.space import create_tables, drop_tables
from mlcopilot.utils import set_llms

set_llms(embedding_model=MockEmbeddingModel)
drop_tables()
create_tables()
task_desc = "test task description"
space = test_ingest_experience()
order_key = Task.embedding.cosine_distance(task_desc)
