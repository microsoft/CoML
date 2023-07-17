"""
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
    schema: str
    config: dict
}

Schema {
    id: str?
    description: str
    parameters: list[Parameter]
}

Parameter {
    name: str
    dtype: "int" | "float" | "str" | "bool" | undefined
    categorical: bool
    choices: str[]?
    low: float?
    high: float?
    logDistributed: bool?
    condition: list[Condition]
}

Condition {
    match: dict?
}
"""


import json
from pathlib import Path

solutions = json.loads(Path("coml/app/public/data/solutions.json").read_text())
algorithms = json.loads(Path("coml/app/public/data/algorithms.json").read_text())
knowledges = json.loads(Path("coml/app/public/data/knowledges.json").read_text())
datasets = json.loads(Path("coml/app/public/data/datasets.json").read_text())
schemas = json.loads(Path("coml/app/public/data/schemas.json").read_text())
task_types = json.loads(Path("coml/app/public/data/taskTypes.json").read_text())

def id_all_unique(data):
    ids = set()
    for d in data:
        assert d["id"] not in ids
        ids.add(d["id"])

id_all_unique(solutions)
id_all_unique(algorithms)
id_all_unique(knowledges)
id_all_unique(datasets)
id_all_unique(schemas)
id_all_unique(task_types)

def check_id_is_in(data, id):
    for d in data:
        if d["id"] == id:
            return
    assert False

def check_valid_dataset(dataset):
    assert isinstance(dataset["name"], str)
    assert isinstance(dataset["description"], str)

def check_valid_task_type(task_type):
    assert isinstance(task_type["name"], str)
    assert isinstance(task_type["description"], str)

def check_valid_model(model):
    assert isinstance(model["name"], str)
    assert isinstance(model["description"], str)

def check_valid_algorithm(algorithm):
    assert isinstance(algorithm["config"], dict)

def check_valid_solution_summary(solution_summary):
    assert isinstance(solution_summary["summary"], str)

def check_valid_schema(schema):
    assert isinstance(schema["description"], str)
    assert isinstance(schema["parameters"], list)
    assert len(set([parameter["name"] for parameter in schema["parameters"]])) == len(schema["parameters"])
    for parameter in schema["parameters"]:
        assert isinstance(parameter["name"], str)
        assert "dtype" not in parameter or parameter["dtype"] in ["int", "float", "str", "bool"]
        assert isinstance(parameter["categorical"], bool)
        if parameter["categorical"]:
            assert isinstance(parameter["choices"], list)
        else:
            assert parameter["dtype"] in ["int", "float"]
            if parameter["dtype"] == "int":
                assert isinstance(parameter["low"], int)
                assert isinstance(parameter["high"], int)
            else:
                assert isinstance(parameter["low"], float)
                assert isinstance(parameter["high"], float)
            assert isinstance(parameter["logDistributed"], bool)
        if "condition" in parameter:
            assert isinstance(parameter["condition"], list)
            for condition in parameter["condition"]:
                assert isinstance(condition["match"], dict)

def find_schema(schema_id):
    for schema in schemas:
        if schema["id"] == schema_id:
            return schema
    assert False

def verify_config(config, schema):
    verified_keys = set()
    for parameter in schema["parameters"]:
        if parameter["name"] not in config:
            assert "condition" in parameter and \
                config[list(parameter["condition"][0]["match"].keys())[0]] != list(parameter["condition"][0]["match"].values())[0]
            continue
        verified_keys.add(parameter["name"])
        if parameter["categorical"]:
            assert config[parameter["name"]] in parameter["choices"]
        else:
            assert parameter["low"] <= config[parameter["name"]] <= parameter["high"]
    assert len(verified_keys) == len(config)

for schema in schemas:
    check_valid_schema(schema)

print(len(solutions))
for solution in solutions:
    assert solution["source"] in ["kaggle", "huggingface", "hpob"]
    assert isinstance(solution["modules"], list)
    for module in solution["modules"]:
        assert isinstance(module, dict)
        assert module["role"] in ["dataset", "taskType", "model", "algorithm", "verifiedAlgorithm", "solutionSummary"]
        assert "purpose" not in module or isinstance(module["purpose"], str)
        if isinstance(module["module"], str):
            if module["role"] == "dataset":
                check_id_is_in(datasets, module["module"])
            elif module["role"] == "taskType":
                check_id_is_in(task_types, module["module"])
            elif module["role"] in ["algorithm", "verifiedAlgorithm"]:
                check_id_is_in(algorithms, module["module"])
            else:
                assert False
        elif module["role"] == "dataset":
            check_valid_dataset(module["module"])
        elif module["role"] == "taskType":
            check_valid_task_type(module["module"])
        elif module["role"] == "model":
            check_valid_model(module["module"])
        elif module["role"] == "algorithm":
            check_valid_algorithm(module["module"])
        elif module["role"] == "verifiedAlgorithm":
            check_valid_algorithm(module["module"])
            check_id_is_in(schemas, module["module"]["schema"])
            verify_config(module["module"]["config"], find_schema(module["module"]["schema"]))

for algorithm in algorithms:
    check_valid_algorithm(algorithm)
for dataset in datasets:
    check_valid_dataset(dataset)
for task_type in task_types:
    check_valid_task_type(task_type)
for knowledge in knowledges:
    assert isinstance(knowledge["knowledge"], str)
    for module in knowledge["contextScope"]:
        assert module["role"] in ["dataset", "taskType", "model", "algorithm", "verifiedAlgorithm", "solutionSummary"]
        assert "purpose" not in module or isinstance(module["purpose"], str)
        if isinstance(module["module"], str):
            if module["role"] == "dataset":
                check_id_is_in(datasets, module["module"])
            elif module["role"] == "taskType":
                check_id_is_in(task_types, module["module"])
            elif module["role"] in ["algorithm", "verifiedAlgorithm"]:
                check_id_is_in(algorithms, module["module"])
            else:
                assert False
    assert knowledge["subjectRole"] in ["dataset", "taskType", "model", "algorithm", "verifiedAlgorithm", "solutionSummary"], knowledge
    assert "subjectSchema" not in knowledge or isinstance(knowledge["subjectSchema"], str)
    if "subjectSchema" in knowledge and isinstance(knowledge["subjectSchema"], str):
        check_id_is_in(schemas, knowledge["subjectSchema"])
