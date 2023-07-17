"""
Schema:

Solution {
    id: str
    modules: Module[]
    metrics: float | Metric[] | undefined
    source: "hpob" | "huggingface" | "kaggle"
}

Knowledge {
    id: str
    contextScope: Module[] // AND
    subjectRole: str
    subjectSchema: str?  // point to a schema
    knowledge: str
}

Metric {
    dataset: Dataset
    metric: float
    extra: str
    split: str?
    protocol: str?
}

Module {
    role: "dataset" | "taskType" | "model" | "algorithm" | "verifiedAlgorithm" | "solutionSummary"
    purpose: str?
    module: Dataset | SolutionSummary | TaskType | Model | Algorithm | VerifiedAlgorithm | str
}

SolutionSummary {
    id: str?
    summary: str
}

Dataset {
    id: str?
    name: str
    description: str
}

TaskType {
    id: str?
    name: str
    description: str
}

Model {
    id: str?
    name: str
    description: str
}

Algorithm {
    id: str?
    config: dict
}

VerifiedAlgorithm {
    id: str?
    schema_id: str
    config: dict
}

Schema {
    id: str?
    description: str
    parameters: list[Parameter]
}

Parameter {
    name: str
    dtype: "int" | "float" | "str" | undefined
    categorical: bool
    choices: str[]?
    low: float?
    high: float?
    log_distributed: bool?
}
"""

import glob
import json
import random
import re
import numpy as np
import pandas as pd
from pathlib import Path
from mlcopilot.knowledge import split_knowledge
from mlcopilot.orm import Solution, Space, Task, Knowledge

from slugify import slugify

random.seed(42)
np.random.seed(42)

solutions_ = []
datasets_ = []
schemas_ = []
task_types_ = []
knowledges_ = []
algorithms_ = []

def hpob_solutions():
    import numpy as np
    from .hpob_preprocess import read_data, array_to_config

    # data = read_data()
    schemas = {}
    for space in Space.select():
        if space.space_id.isdigit():
            print(Solution.select().where(Solution.space_id == space.space_id).count())
            # print(space.__dict__)
            lines = list(space.desc.splitlines())
            description = lines[0].strip()[1:-1]
            parameters = {}
            for name, raw_schema in re.findall(r'\t\t\d+\. (.*?): (.*)', space.desc[:space.desc.find('Packeges')]):
                print(raw_schema)
                if raw_schema.startswith("choose from"):
                    schema = {
                        "name": name,
                        "dtype": "str",
                        "categorical": True,
                        "choices": raw_schema[len("choose from "):].split(", ")
                    }
                    if schema["choices"] in [["true", "false"], ["false", "true"]]:
                        schema["dtype"] = "bool"
                        schema["choices"] = [True, False]
                    print(schema)
                else:
                    match = re.match(r"(integer|real number); low = (.*); high = (.*); (log|linear) scale", raw_schema)
                    if match.group(1) == "integer":
                        dtype = "int"
                    else:
                        dtype = "float"
                    if dtype == "int":
                        low = int(match.group(2))
                        high = int(match.group(3))
                    else:
                        low = float(match.group(2))
                        high = float(match.group(3))
                    log_distributed = match.group(4) == "log"
                    schema = {
                        "name": name,
                        "dtype": dtype,
                        "categorical": False,
                        "low": low,
                        "high": high,
                        "logDistributed": log_distributed
                    }
                    if "; only when" in raw_schema:
                        condition = re.search(r'only when (.*) = (.*)', raw_schema)
                        schema["condition"] = [
                            {
                                "match": {
                                    condition.group(1): condition.group(2)
                                }
                            }
                        ]
                parameters[name] = schema
            packages = [t.replace('_', ' ') for t in re.findall(r'\t\t- (.*)', space.desc[space.desc.find('Packeges'):])]
            description += '. Environment: ' + ', '.join(packages) + '.'
            print(description)

            slug = "-".join(description.split()[1].split(".")[2:]) + "-" + space.space_id

            schemas[space.space_id] = {"id": slug, "description": description, "parameters": list(parameters.values())}

    contexts = {}

    for task in Task.select():
        if task.task_id.isdigit():
            dataset_desc = task.desc[len("Task:  "):].replace("dataset name is", "dataset is called")
            dataset_name = re.search(r'\"(.*)\"\.', dataset_desc).group(1).replace("_", " ")
            contexts[task.task_id] = {
                "id": "openml-" + dataset_name.replace(" ", "-").lower() + "-" + task.task_id,
                "name": dataset_name,
                "description": dataset_desc
            }
            print(contexts[task.task_id])

    hpob_data = read_data()
    solutions = []
    for space_id, space_data in hpob_data.items():
        for task_id, task_data in space_data.items():
            if task_id not in contexts:
                continue
            X, y = np.array(task_data["X"]), np.array(task_data["y"]).flatten()
            for idx in np.argsort(y + np.random.uniform(0, 1e-8))[::-1][:10]:  # TODO: change to 100
                config = array_to_config(X[idx], space_id)
                metric = y[idx]
                schema_id = schemas[space_id]["id"]
                context_id = contexts[task_id]["id"]
                solution_id = f"hpob-{schema_id}-{context_id}-{idx + 1:03d}"
                solutions.append({
                    "id": solution_id,
                    "modules": [
                        {
                            "role": "dataset",
                            "module": context_id
                        },
                        {
                            "role": "verifiedAlgorithm",
                            "module": {
                                # "id": solution_id,
                                "schema": schema_id,
                                "config": config
                            }
                        }
                    ],
                    "metrics": metric,
                    "context": context_id,
                    "schema": schema_id,
                    "source": "hpob"
                })

    print(len(solutions))
    schemas_.extend(schemas.values())
    datasets_.extend(contexts.values())
    solutions_.extend(solutions)

    knowledges = []
    for knowledge in Knowledge.select():
        if knowledge.space_id.isdigit():
            knowledges.append({
                "id": f"hpob-{schemas[knowledge.space_id]['id']}-{knowledge.id:04d}",
                "contextScope": [],
                "subjectRole": "verifiedAlgorithm",
                "subjectSchema": schemas[knowledge.space_id]['id'],
                "knowledge": knowledge.knowledge
            })
    knowledges_.extend(knowledges)


def huggingface_solutions():
    dataset_desc = json.loads(Path("assets/private/dataset_desc.json").read_text())
    datasets = {}
    existing_ids = set([d["id"] for d in datasets_])
    for dataset_name, dataset in dataset_desc.items():
        dataset_id = slugify(dataset_name)
        assert dataset_id not in existing_ids
        if dataset_id in [d["id"] for d in datasets.values()]:
            for i in range(2, 100):
                if dataset_id + "-" + str(i) not in existing_ids:
                    dataset_id = dataset_id + "-" + str(i)
                    break
        datasets[dataset_name] = {
            "id": dataset_id,
            "name": dataset_name,
            "description": dataset
        }

    assert len(set([d["id"] for d in datasets.values()])) == len(datasets)

    print(list(datasets.values())[:10])
    datasets_.extend(datasets.values())

    tasks_desc = json.loads(Path("assets/private/tasks_from_huggingface.json").read_text())
    tasks = {
        id: {
            "id": id,
            "name": id.replace("-", " ").capitalize(),
            "description": task[0]
        } for id, task in tasks_desc.items()
    }
    print(list(tasks.values()))
    task_types_.extend(tasks.values())

    def search_dataset_id(dataset_name):
        mappings = {
            "SNLI": "SNLI (Stanford Natural Language Inference)",
        }
        if dataset_name in mappings:
            dataset_name = mappings[dataset_name]
        if dataset_name.startswith("Common Voice"):
            dataset_name = "Common Voice"
        if dataset_name.startswith("JW300"):
            dataset_name = "JW300"
        return datasets[dataset_name]["id"]

    solutions = []
    wrong_count = 0
    for file in glob.glob("assets/private/model_cards_1400/*.json"):
        model_card = json.loads(Path(file).read_text())
        used_dataset = []
        wrong = False
        for dataset_name, purpose in model_card["datasets"].items():
            if not isinstance(purpose, str):
                print("Purpose is not a string:", purpose)
                wrong = True
                continue
            try:
                used_dataset.append({
                    "role": "dataset",
                    "purpose": purpose,
                    "module": search_dataset_id(dataset_name)
                })
            except KeyError:
                print("Dataset not found", dataset_name)
                wrong = True

        metrics = []
        for protocol in model_card["metrics"]:
            for dataset_name in model_card["metrics"][protocol]:
                try:
                    dataset_id = search_dataset_id(dataset_name)
                except KeyError:
                    print("Dataset not found", dataset_name)
                    wrong = True
                    continue

                metric = model_card["metrics"][protocol][dataset_name]
                if isinstance(metric, list) and len(metric) == 1:
                    metric = metric[0]

                def _transform_metric(metric):
                    if isinstance(metric, (float, int)):
                        return {"metric": float(metric)}
                    if isinstance(metric, str):
                        return {"metric": float(metric.split()[0]), "extra": metric}
                    raise ValueError(metric)

                if not isinstance(metric, list):
                    try:
                        metrics.append({
                            "dataset": dataset_id,
                            **_transform_metric(metric),
                            "protocol": protocol,
                        })
                    except ValueError:
                        print("Metric error", metric)
                        wrong = True
                else:
                    if len(metric) >= 2:
                        metric = metric[-2:]
                    assert len(metric) == 2
                    for i, m in enumerate(metric):
                        try:
                            split = {0: "val", 1: "test"}[i]
                            metrics.append({
                                "dataset": search_dataset_id(dataset_name),
                                **_transform_metric(m),
                                "split": split,
                                "protocol": protocol,
                            })
                        except ValueError:
                            print("Metric error", m)
                            wrong = True

        wrong_count += wrong

        solution = {
            "id": "huggingface-" + slugify(Path(file).stem),
            "modules": [
                {
                    "role": "model",
                    "module": model_card["model"],
                },
                {
                    "role": "algorithm",
                    "purpose": "Model training hyper-parameters.",
                    "module": {"config": model_card["hyperparameters"]}
                },
                {
                    "role": "taskType",
                    "module": model_card["task"]
                },
                {
                    "role": "solutionSummary",
                    "module": {
                        "summary": model_card["summary"]
                    }
                }
            ] + used_dataset,
            "metrics": metrics,
            "source": "huggingface"
        }
        solutions.append(solution)

    print(wrong_count)
    print(solutions[:10])
    solutions_.extend(solutions)

    knowledges = {}
    for knowledge in Knowledge.select():
        if knowledge.space_id in ("model", "hp"):
            task_name, dataset_name = knowledge.task_id.split("_", 1)
            did = search_dataset_id(dataset_name)
            for i in range(1000):
                kid = f"huggingface-{knowledge.space_id}-{task_name}-{did}-{i + 1:03d}"
                if kid not in knowledges:
                    break
            knowledges[kid] = {
                "id": kid,
                "contextScope": [
                    {"role": "taskType", "module": task_name},
                    {"role": "dataset", "module": did}
                ],
                "subjectRole": "model" if knowledge.space_id == "model" else "algorithm",
                "knowledge": knowledge.knowledge
            }

    knowledges_.extend(knowledges.values())


def kaggle_solutions():
    root_dir = Path("assets/private")

    # TODO: creating spaces

    # Creating tasks and solutions
    existing_ids = set()
    converted_solutions = []
    algorithms = {}
    solutions = pd.read_json(root_dir / 'kaggle_solution.jsonl', lines=True)
    for _, row in solutions.iterrows():
        for i in range(1000):
            sid = f"{row['source']}-{i + 1:03d}"
            if sid not in existing_ids:
                break
        existing_ids.add(sid)

        if row["api"] not in algorithms:
            algorithms[row["api"]] = {
                "id": slugify(row["api"]),
                "config": {"api": row["api"]}
            }

        converted_solutions.append({
            "id": sid,
            "modules": [
                {
                    "role": "algorithm",
                    "purpose": "API",
                    "module": algorithms[row["api"]]["id"]
                },
                {
                    "role": "algorithm",
                    "purpose": "API parameters",
                    "module": {"config": row['parameters']}
                },
                {
                    "role": "solutionSummary",
                    "module": {
                        "summary": row['context']
                    }
                }
            ],
            "source": "kaggle"
        })

    print(list(algorithms.values())[:10])
    print(converted_solutions[:10])
    algorithms_.extend(algorithms.values())
    solutions_.extend(converted_solutions)

    # Creating knowledge
    knowledges = {}
    knowledge = json.loads((root_dir / 'kaggle_knowledge.json').read_text())
    for space_name, knowledge_text in knowledge.items():
        for kn in split_knowledge(knowledge_text):
            for i in range(1000):
                kid = f"{algorithms[space_name]['id']}-{i + 1:03d}"
                if kid not in knowledges:
                    break
            knowledges[kid] = {
                "id": kid,
                "contextScope": [
                    {"role": "algorithm", "module": algorithms[space_name]['id']},
                ],
                "subjectRole": "algorithm",
                "knowledge": kn
            }
    knowledges_.extend(knowledges.values())


hpob_solutions()
huggingface_solutions()
kaggle_solutions()

Path("coml/app/public/data/solutions.json").write_text(json.dumps(solutions_))
Path("coml/app/public/data/algorithms.json").write_text(json.dumps(algorithms_))
Path("coml/app/public/data/knowledges.json").write_text(json.dumps(knowledges_))
Path("coml/app/public/data/datasets.json").write_text(json.dumps(datasets_))
Path("coml/app/public/data/schemas.json").write_text(json.dumps(schemas_))
Path("coml/app/public/data/taskTypes.json").write_text(json.dumps(task_types_))
