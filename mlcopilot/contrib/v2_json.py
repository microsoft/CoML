"""
Schema:

Solution {
    id: str
    config: dict
    modules: Module[]
    metrics: float | Metric[]
    source: "hpob" | "huggingface" | "kaggle"
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
    module: Module | str
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
    schema_id: str
    config: dict
}

Schema {
    id: str?
    description: str
    parameters: dict[str, Parameter]
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
import re
from pathlib import Path
from mlcopilot.orm import Solution, Space, Task

from slugify import slugify

solutions_ = []
datasets_ = []
schemas_ = []
task_types_ = []

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
                    log_distribution = match.group(4) == "log"
                    schema = {
                        "name": name,
                        "dtype": dtype,
                        "categorical": False,
                        "low": low,
                        "high": high,
                        "log_distribution": log_distribution
                    }
                parameters[name] = schema
            packages = [t.replace('_', ' ') for t in re.findall(r'\t\t- (.*)', space.desc[space.desc.find('Packeges'):])]
            description += '. Environment: ' + ', '.join(packages) + '.'
            print(description)

            slug = "-".join(description.split()[1].split(".")[2:]) + space.space_id

            schemas[space.space_id] = {"id": slug, "description": description, "parameters": parameters}

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
            for idx in np.argsort(y + np.random.uniform(0, 1e-8))[::-1][:100]:
                config = array_to_config(X[idx], space_id)
                metric = y[idx]
                schema_id = schemas[space_id]
                context_id = contexts[task_id]
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
                                "schema_id": schema_id,
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
    datasets_.extend(datasets)

    tasks_desc = json.loads(Path("assets/private/tasks_from_huggingface.json").read_text())
    tasks = {
        id: {
            "id": id,
            "name": id.replace("-", " ").capitalize(),
            "description": task[0]
        } for id, task in tasks_desc.items()
    }
    print(list(tasks.values()))
    task_types_.extend(tasks)

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
                    "module": model_card["hyperparameters"]
                },
                {
                    "role": "taskType",
                    "module": model_card["task"]
                },
                {
                    "role": "solutionSummary",
                    "module": model_card["summary"]
                }
            ] + used_dataset,
            "metrics": metrics,
            "source": "huggingface"
        }
        solutions.append(solution)

    print(wrong_count)
    print(solutions[:10])
    solutions_.extend(solutions)


# def kaggle_solutions():



huggingface_solutions()
