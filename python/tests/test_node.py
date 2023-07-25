from coml.node import get_function_description, suggest_machine_learning_module
from coml.prompt import COML_EXAMPLES

def test_get_function_description():
    description = get_function_description()
    assert description["name"] == "suggestMachineLearningModule"
    assert description["description"]
    assert description["parameters"]

def test_suggest_machine_learning_module():
    module = suggest_machine_learning_module(
        [
            {
                "role": "dataset",
                "module": {
                    "name": "MNIST",
                    "description": "A dataset of handwritten digits",
                }
            }
        ],
        "verifiedAlgorithm",
        "rpart-preproc-4796"
    )
    assert len(module) == 3
    for m in module:
        assert m["role"] == "verifiedAlgorithm"
        assert m["module"]["schema"] == "rpart-preproc-4796"
        assert m["module"]["config"]

def test_coml_examples():
    for example in COML_EXAMPLES:
        assert example["goal"]
        assert example["data"]
        assert example["response"]
        if isinstance(example["response"], dict):
            module = suggest_machine_learning_module(
                example["response"]["existingModules"],
                example["response"]["targetRole"],
                example["response"].get("targetSchemaId")
            )
            assert len(module) == 3
            print(module)
