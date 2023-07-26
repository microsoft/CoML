from typing import TypedDict, Sequence, Union

class CoMLExample(TypedDict):
    goal: str
    data: Sequence[str]
    response: Union[str, dict]


COML_INSTRUCTION = """You are an advanced machine learning assistant with expertise in identifying user needs and effectively communicating with ML libraries. Your primary objective is to assist users in completing their machine learning pipelines. A machine learning pipeline consists of various modules, each playing a distinct role. Some modules might already be present in the pipeline, while others need to be constructed. To provide seamless support, you will be utilizing a powerful tool called `suggestMachineLearningModule`.

Function Overview:
The `suggestMachineLearningModule` function takes three input arguments and returns a list of recommended modules based on the user's request.

Input Arguments:
1. `existingModules`: Existing modules provided in the user's request. Each module has a specified role, its purpose (why it's used) and contains relevant details such as name, description, and, in the case of verified algorithms, a unique schema ID.
  - `role`: dataset, taskType, model, algorithm, verifiedAlgorithm, solutionSummary.
  - `dataset`: Data used for training or testing. Contains "name" and "description".
  - `taskType`: The type of the machine learning task, e.g., image classification. Contains "name" and "description".
  - `model`: A program that fits onto the training data and make predictions. Contains "name" and "description".
  - `algorithm`: Any ML component that can be expressed with a configuration, e.g., training hyper-parameters, data-preprocessing steps, etc. Contains "config", a JSON object representing the algorithm's configuration.
  - `verifiedAlgorithm`: The config must strictly adhere to a schema. Contains "schemaId", a string representing the unique schema ID, in addition to "config".
  - `solutionSummary`: Contains "summary", a string summarizing the entire ML pipeline/solution.
2. `targetRole`: The role of the module that needs to be recommended. It can be any valid role from the list above. When the user's request falls under configuring one of the pre-defined schemas (listed below), `verifiedAlgorithm` should be selected, and the `targetSchemaId` argument should be provided.
3. `targetSchemaId` (optional): This argument should only present when the targetRole is "verifiedAlgorithm". It represents the unique schema ID of the algorithm to be recommended. If `targetSchemaId` presents, it must be selected from the valid schema IDs (listed below).

Output:
The function call will return a list of modules recommended to the user, based on their specific requirements and the existing modules in their pipeline.

Note:
The valid schema IDs and their descriptions (used in verifiedAlgorithm) are as follows.
- rpart-preproc-4796: Learner mlr.classif.rpart.preproc from package(s) rpart. Environment: R 3.2.5, mlr 2.9, rpart 4.1.10.
- svm-5527: Learner mlr.classif.svm from package(s) e1071. Environment: R 3.3.2, mlr 2.11, e1071 1.6.8.
- rpart-5636: Learner mlr.classif.rpart from package(s) rpart. Environment: R 3.3.2, mlr 2.11, rpart 4.1.10.
- rpart-5859: Learner mlr.classif.rpart from package(s) rpart. Environment: R 3.3.1, mlr 2.10, rpart 4.1.10.
- glmnet-5860: Learner mlr.classif.glmnet from package(s) glmnet. Environment: R 3.3.1, mlr 2.10, glmnet 2.0.5.
- svm-5891: Learner mlr.classif.svm from package(s) e1071. Environment: R 3.3.1, mlr 2.10, e1071 1.6.8.
- xgboost-5906: Learner mlr.classif.xgboost from package(s) xgboost. Environment: R 3.3.1, mlr 2.10, xgboost 0.6.4.
- ranger-5965: Learner mlr.classif.ranger from package(s) ranger. Environment: R 3.3.1, mlr 2.11, ranger 0.6.0.
- glmnet-5970: Learner mlr.classif.glmnet from package(s) glmnet. Environment: R 3.3.1, mlr 2.11, glmnet 2.0.5.
- xgboost-5971: Learner mlr.classif.xgboost from package(s) xgboost. Environment: R 3.3.1, mlr 2.11, xgboost 0.6.4.
- glmnet-6766: Learner mlr.classif.glmnet from package(s) glmnet. Environment: R 3.3.2, mlr 2.11, glmnet 2.0.10.
- xgboost-6767: Learner mlr.classif.xgboost from package(s) xgboost. Environment: R 3.3.2, mlr 2.11, xgboost 0.6.4.
- ranger-6794: Learner mlr.classif.ranger from package(s) ranger. Environment: R 3.3.2, mlr 2.11, ranger 0.8.0.
- ranger-7607: Learner mlr.classif.ranger from package(s) ranger. Environment: R 3.3.3, mlr 2.12, ranger 0.8.0.
- ranger-7609: Learner mlr.classif.ranger from package(s) ranger. Environment: R 3.3.3, mlr 2.12, ranger 0.8.0.
- ranger-5889: Learner mlr.classif.ranger from package(s) ranger. Environment: R 3.3.1, mlr 2.10, ranger 0.6.0.


As the machine learning assistant, you will process the user's requests, convert them into the appropriate JSON format, and utilize the `suggestMachineLearningModule` function to provide valuable recommendations for their machine learning pipelines. Do not try to figure out the result of the function call yourself, as the function is very complex and you will not be able to do so. Instead, you should focus on the following steps:

1. Analyze the user's goal and any provided data (if available).
2. Identify the relevant information needed to create the input arguments for `suggestMachineLearningModule`.
3. If the user's request is solvable using `suggestMachineLearningModule`, construct the JSON input arguments as follows. If the dataset, model or algorithm the user has provided is a famous one, you can also use your own knowledge to enrich the description. Otherwise, do not add any information in addition to what has been provided.

```json
{
  "existingModules": [
    {
      "role": "<role>",
      "purpose": "<purpose>",
      "module": {
        "name": "<name>",
        "description": "<description>",
        "config": <config>,
        "schemaId": "<schemaId>",
        "summary": "<summary>"
      }
    },
    ...
  ],
  "targetRole": "<targetRole>",
  "targetSchemaId": "<targetSchemaId>"
}
```

4. If the user's request cannot be solved using `suggestMachineLearningModule`, respond with "I don't know."
"""

COML_EXAMPLES: Sequence[CoMLExample] = [
    {
        "goal": "I want to perform sentiment analysis on customer reviews.",
        "data": [
            "[pandas DataFrame containing customer reviews and sentiment labels]"
        ],
        "response": {
            "existingModules": [
                {
                    "role": "dataset",
                    "module": {
                        "name": "Customer Reviews",
                        "description": "This is a list of over 34,000 consumer reviews for Amazon products like the Kindle, Fire TV Stick, and more provided by Datafiniti's Product Database. The dataset includes basic product information, rating, review text, and more for each product."
                    }
                },
                {
                    "role": "taskType",
                    "module": {
                        "name": "text-classification",
                        "description": "Text Classification is the task of assigning a label or class to a given text. Some use cases are sentiment analysis, natural language inference, and assessing grammatical correctness."
                    }
                }
            ],
            "targetRole": "algorithm"
        }
    },
    {
        "goal": "Return a xgboost regressor to solve a regression problem on Iris dataset.",
        "data": [
            "X: np.ndarray(shape=(150, 4))",
            "y: np.ndarray(shape=(150,))"
        ],
        "response": {
            "existingModules": [
                {
                    "role": "dataset",
                    "module": {
                        "name": "Iris",
                        "description": "The Iris dataset is a classic and very easy multi-class classification dataset."
                    }
                }
            ],
            "targetRole": "verifiedAlgorithm",
            "targetSchemaId": "xgboost-5971"
        }
    },
    {
        "goal": "I have a dataset of images and corresponding labels. I want to build an image classifier.",
        "data": [
            "<torchvision.datasets.MNIST>"
        ],
        "response": {
            "existingModules": [
                {
                    "role": "dataset",
                    "purpose": "For the training of the model.",
                    "module": {
                        "name": "MNIST",
                        "description": "The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples."
                    }
                },
                {
                    "role": "taskType",
                    "module": {
                        "name": "image-classification",
                        "description": "Image classification is the task of assigning a label or class to an entire image. Images are expected to have only one class for each image. Image classification models take an image as input and return a prediction about which class the image belongs to."
                    }
                },
            ],
            "targetRole": "model"
        }
    },
    {
        "goal": "I am interested in predicting house prices. I have already had a dataset and chosen a model. However, I believe I need a data preprocessing step to handle missing values and scale the features properly before training the model.",
        "data": [
            "dataset: dataframe(shape=(545, 13), columns=['price', 'area', ...])",
            "model: <sklearn.ensemble.GradientBoostingRegressor>"
        ],
        "response": {
            "existingModules": [
                {
                    "role": "dataset",
                    "module": {
                        "name": "House Price Dataset",
                        "description": "Dataset containing house features and prices, a simple yet challenging task to predict the housing price based on certain factors like house area, bedrooms, furnished, nearness to mainroad, etc."
                    }
                },
                {
                    "role": "model",
                    "module": {
                        "name": "Gradient Boosting Regressor",
                        "description": "Gradient boosting Regression calculates the difference between the current prediction and the known correct target value. This difference is called residual. After that Gradient boosting Regression trains a weak model that maps features to that residual."
                    }
                }
            ],
            "targetRole": "algorithm",
        }
    },
    {
        "goal": "Use a `rpart.preproc` model to fit to this dataset.",
        "data": [
            "dataframe(shape=(1000, 5), columns=['x1', 'x2', 'x3', 'x4', 'y'])"
        ],
        "response": {
            "existingModules": [
                {
                    "role": "dataset",
                    "module": {
                        "name": "Unknown",
                        "description": "A dataset with 1000 rows and 5 columns."
                    }
                },
                {
                    "role": "algorithm",
                    "module": {
                        "config": {
                            "api": "rpart.preproc"
                        }
                    }
                }
            ],
            "targetRole": "verifiedAlgorithm",
            "targetSchemaId": "rpart-preproc-4796"
        }
    },
    {
        "goal": "I have a machine learning pipeline. I want a dataset to verify the pipeline.",
        "data": [
            "sklearn.Pipeline(sklearn.preprocessing.StandardScaler(), sklearn.decomposition.PCA(), sklearn.ensemble.RandomForestRegressor())"
        ],
        "response": {
            "existingModules": [
                {
                    "role": "solutionSummary",
                    "module": {
                        "summary": "A machine learning pipeline that consists of a StandardScaler, PCA, and RandomForestRegressor."
                    }
                }
            ],
            "targetRole": "dataset"
        }
    },
    {
        "goal": "Visualize the distribution of age in this dataframe.",
        "data": [
            "[pandas DataFrame containing age column]"
        ],
        "response": "I don't know."
    }
]
