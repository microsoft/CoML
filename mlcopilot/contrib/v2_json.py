"""
Schema:

Solution {
    id: str
    config: dict
    metric: float
    context: (str | Context | ContextUsage)[]
    schema: str
    source: "hpob" | "huggingface" | "kaggle"
}

ContextUsage {
    uses: str
    purpose: str?
}

Context {
    id: str
    role: "dataset" | "taskType" | "model" | "arbitrary"
    name: str?
    description: str?
}

Schema {
    id: str
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

import json
import re
from mlcopilot.orm import Solution, Space, Task

solutions_ = []
contexts_ = []
schemas_ = []

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
            dataset_name = re.search(r'\"(.*)\"\.', dataset_desc).group(1).replace("_", "-").lower()
            contexts[task.task_id] = {
                "id": "openml-" + dataset_name + "-" + task.task_id,
                "role": "dataset",
                "description": dataset_desc
            }

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
                solutions.append({
                    "id": f"{schema_id}-{context_id}-{idx + 1:03d}",
                    "config": config,
                    "metric": metric,
                    "context": context_id,
                    "schema": schema_id,
                    "source": "hpob"
                })

    print(len(solutions))
    schemas_.extend(schemas.values())
    contexts_.extend(contexts.values())
    solutions_.extend(solutions)





hpob_solutions()
