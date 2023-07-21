from coml.node import get_function_description, suggest_machine_learning_module

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
