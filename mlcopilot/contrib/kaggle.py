import json
from pathlib import Path

import numpy as np
import pandas as pd
import orjson

from mlcopilot.experience import SAVE_OPTIONS
from mlcopilot.knowledge import split_knowledge
from mlcopilot.orm import database_proxy, Space, Task, Solution, Knowledge
from mlcopilot.utils import get_llm


def ingest_kaggle_data(root_dir: Path = Path('assets/private')):
    solutions = pd.read_json(root_dir / 'kaggle_solution.jsonl', lines=True)

    embeddings = get_llm("embedding")()

    # Creating spaces
    space_descriptions = json.loads((root_dir / 'api_docs.json').read_text())
    with database_proxy.atomic():
        for api in solutions['api'].unique():
            try:
                Space.get(Space.space_id == api)
                print(f"Space {api} already exists, skip ingestion.")
            except Space.DoesNotExist:
                Space.create(
                    space_id=api,
                    desc=space_descriptions[api]["short"],  # TODO: find API description
                )
                print("Ingested space into database:", api)

    # Creating knowledge
    knowledge = json.loads((root_dir / 'kaggle_knowledge.json').read_text())
    for space_name, knowledge_text in knowledge.items():
        for kn in split_knowledge(knowledge_text):
            with database_proxy.atomic():
                try:
                    Knowledge.get(
                        Knowledge.knowledge == kn,
                        Knowledge.space == space_name
                    )
                except Knowledge.DoesNotExist:
                    Knowledge.create(
                        knowledge=kn,
                        space=space_name
                    )
                    print("Ingested knowledge into database:", space_name, kn)

    # Creating tasks and solutions
    for _, row in solutions.iterrows():
        with database_proxy.atomic():
            try:
                task_id = Task.get(desc=row['context']).task_id
                print(f"Task {row['context']} already exists, skip ingestion.")
            except Task.DoesNotExist:
                # Find available id
                for i in range(1000):
                    task_id = f"{row['source']}-{i + 1:03d}"
                    try:
                        Task.get(Task.task_id == task_id)
                    except Task.DoesNotExist:
                        break
                else:
                    raise ValueError("Cannot find available task id.")
                embedding = np.asarray(
                    embeddings.embed_query(row['context']), dtype=np.float32
                )
                Task.create(
                    task_id=task_id,
                    desc='Scenario: ' + row['context'],
                    row_desc=row['context'],
                    embedding=embedding.tobytes()
                )

                print("Ingested task into database:", task_id)

        with database_proxy.atomic():
            config_json = orjson.dumps(row['parameters'], option=SAVE_OPTIONS)
            try:
                Solution.get(
                    Solution.space == row['api'],
                    Solution.task == task_id,
                    Solution.row_config == config_json,
                )
                print(
                    f"Solution {row['api']}--{task_id}--{config_json} already exists, skip ingestion."
                )
            except Solution.DoesNotExist:
                Solution.create(
                    space=row['api'],
                    task=task_id,
                    row_config=config_json,
                    demo=config_json,
                    metric=0.,
                    extra_metric="",
                )
                print(
                    "Ingested solution into database:",
                    row['api'],
                    task_id,
                    config_json,
                )



if __name__ == "__main__":
    ingest_kaggle_data()
