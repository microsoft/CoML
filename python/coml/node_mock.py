# Mocked version of node.

from typing import List, Optional


def suggest_machine_learning_module(
    existing_modules: List[dict], target_role: str, target_schema: Optional[str]
) -> List[dict]:
    if target_role == "model":
        return [
            {"role": "model", "module": {"name": "Random Forest", "description": "Random Forest Classifier."}},
            {"role": "model", "module": {"name": "SVM", "description": "Support Vector Machine (SVM) Classifier."}},
            {"role": "model", "module": {"name": "Boosting", "description": "Gradient Boosting Classifier."}},
        ]
    else:
        raise NotImplementedError()


def get_function_description():
    # This might be a legacy set of description, only to mock the behavior of this function.
    return {
        "name": "suggestMachineLearningModule",
        "description": "Get recommendations of a machine learning module given existing modules on the pipeline. A machine learning pipeline consists of multiple modules playing different roles. For example, a pipeline can consist of a dataset, a task type, a model, and an algorithm. This function recommends a module of the target role given existing modules on the pipeline.",
        "parameters": {
            "type": "object",
            "properties": {
                "existingModules": {
                    "type": "array",
                    "description": "Existing modules on the pipeline. It can be a dataset, a selected ML model, a certain task type, an overview of the whole ML solution, or an algorithm configuration. Existing modules MUST BE NOT EMPTY.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "enum": [
                                    "dataset",
                                    "taskType",
                                    "model",
                                    "algorithm",
                                    "verifiedAlgorithm",
                                    "solutionSummary",
                                ],
                                "description": "The role of the module within the pipeline.\n- dataset: Data used for training or testing.\n- taskType: The type of the machine learning task, e.g., image classification.\n- model: A program that fits onto the training data and make predictions.\n- algorithm: Any ML component that can be expressed with a configuration, e.g., training hyper-parameters, data-preprocessing steps, etc.\n- verifiedAlgorithm: An algorithm that strictly follows a schema and thus directly runnable.\n- solutionSummary: An overview of the entire machine learning pipeline/solution.",
                            },
                            "purpose": {"type": "string", "description": "Why this module is used on the pipeline."},
                            "module": {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "string",
                                        "description": "ID of the module in the database. Do not use this field if you are not sure.",
                                    },
                                    "name": {
                                        "type": "string",
                                        "description": "Name of the dataset / model / task type. Only use this field when role is dataset/model/taskType.",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Description of the dataset / model / task type. Only use this field when role is dataset/model/taskType.",
                                    },
                                    "summary": {
                                        "type": "string",
                                        "description": "Summary of the solution. Only use this field when role is solutionSummary.",
                                    },
                                    "config": {
                                        "type": "object",
                                        "description": "Configuration of the algorithm. Only use this field when role is algorithm/verifiedAlgorithm.",
                                    },
                                    "schema": {
                                        "type": "string",
                                        "description": "Schema ID of the algorithm. Only use this field when role is verifiedAlgorithm.",
                                    },
                                },
                            },
                        },
                        "required": ["role", "module"],
                    },
                    "minItems": 1,
                },
                "targetRole": {
                    "type": "string",
                    "description": "The role of the module to be recommended.",
                    "enum": ["dataset", "taskType", "model", "algorithm", "verifiedAlgorithm", "solutionSummary"],
                },
                "targetSchemaId": {
                    "type": "string",
                    "description": "This field should be used together with targetRole = verifiedAlgorithm.The function will return an algorithm that is valid for the specified schema. Valid schema IDs and descriptions are:\n- rpart-preproc-4796: Learner mlr.classif.rpart.preproc from package(s) rpart. Environment: R 3.2.5, mlr 2.9, rpart 4.1.10.\n- svm-5527: Learner mlr.classif.svm from package(s) e1071. Environment: R 3.3.2, mlr 2.11, e1071 1.6.8.\n- rpart-5636: Learner mlr.classif.rpart from package(s) rpart. Environment: R 3.3.2, mlr 2.11, rpart 4.1.10.\n- rpart-5859: Learner mlr.classif.rpart from package(s) rpart. Environment: R 3.3.1, mlr 2.10, rpart 4.1.10.\n- glmnet-5860: Learner mlr.classif.glmnet from package(s) glmnet. Environment: R 3.3.1, mlr 2.10, glmnet 2.0.5.\n- svm-5891: Learner mlr.classif.svm from package(s) e1071. Environment: R 3.3.1, mlr 2.10, e1071 1.6.8.\n- xgboost-5906: Learner mlr.classif.xgboost from package(s) xgboost. Environment: R 3.3.1, mlr 2.10, xgboost 0.6.4.\n- ranger-5965: Learner mlr.classif.ranger from package(s) ranger. Environment: R 3.3.1, mlr 2.11, ranger 0.6.0.\n- glmnet-5970: Learner mlr.classif.glmnet from package(s) glmnet. Environment: R 3.3.1, mlr 2.11, glmnet 2.0.5.\n- xgboost-5971: Learner mlr.classif.xgboost from package(s) xgboost. Environment: R 3.3.1, mlr 2.11, xgboost 0.6.4.\n- glmnet-6766: Learner mlr.classif.glmnet from package(s) glmnet. Environment: R 3.3.2, mlr 2.11, glmnet 2.0.10.\n- xgboost-6767: Learner mlr.classif.xgboost from package(s) xgboost. Environment: R 3.3.2, mlr 2.11, xgboost 0.6.4.\n- ranger-6794: Learner mlr.classif.ranger from package(s) ranger. Environment: R 3.3.2, mlr 2.11, ranger 0.8.0.\n- ranger-7607: Learner mlr.classif.ranger from package(s) ranger. Environment: R 3.3.3, mlr 2.12, ranger 0.8.0.\n- ranger-7609: Learner mlr.classif.ranger from package(s) ranger. Environment: R 3.3.3, mlr 2.12, ranger 0.8.0.\n- ranger-5889: Learner mlr.classif.ranger from package(s) ranger. Environment: R 3.3.1, mlr 2.10, ranger 0.6.0.",
                },
            },
            "required": ["existingModules", "targetRole"],
        },
    }
