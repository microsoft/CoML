import json
import logging
from typing import Any, List, Optional

from colorama import Fore, Style
from langchain.chat_models.base import BaseChatModel
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage, FunctionMessage

from .node import get_function_description, suggest_machine_learning_module
# from .node_mock import get_function_description, suggest_machine_learning_module

_logger = logging.getLogger(__name__)

CoMLIntention = str

COML_INSTRUCTION = """You are a machine learning assistant who is good at identifying user's needs and communicating with ML libraries.

Specifically, the user is trying to complete a machine learning pipeline. A pipeline consists of multiple multiple modules (a.k.a. components), playing different roles. Some modules of the pipeline could already exist, while others remain to be constructed. For example, the pipeline could already have a dataset, a task type, and a model, but the user is still looking for an algorithm to train the model. The final goal is to recommend a module of the target role given existing modules on the pipeline.

To accomplish this goal, you should be use is a tool called `suggestMachineLearningModule`. Given user's request, your task is to convert it into a function call of `suggestMachineLearningModule`, which accepts 3 input arguments:

1. existingModules: Existing modules that have already been given in the user's request. It could be one of the following roles:
    - dataset: Data used for training or testing. A dataset should have a "name" and a "description".
    - taskType: The type of the machine learning task, e.g., image classification. A task type should have a "name" and a "description".
    - model: A program that fits onto the training data and make predictions. A model should have a "name" and a "description".
    - algorithm: Any ML component that can be expressed with a configuration, e.g., training hyper-parameters, data-preprocessing steps, etc. An algorithm should have a "config", which is a JSON object.
    - verifiedAlgorithm: An algorithm that strictly follows a schema and thus directly runnable. Compared to algorithm, a verified algorithm should have an additional "schemaId", which is a string.
    - solutionSummary: An overview of the entire machine learning pipeline/solution. A solution summary should have a "summary", which is a string.
2. targetRole: The role of the module that you are trying to recommend. It could be one of the valid roles listed above.
3. targetSchemaId: The schema ID of the algorithm that you are trying to recommend. This argument is optional, and should only be used when targetRole is "verifiedAlgorithm".

The function call will return a list of modules that will be recommended to the user.

The valid schema IDs and their descriptions (used in verifiedAlgorithm) are as follows:
- rpart-preproc-4796: Learner mlr.classif.rpart.preproc from package(s) rpart. Environment: R 3.2.5, mlr 2.9, rpart 4.1.10.\n- svm-5527: Learner mlr.classif.svm from package(s) e1071. Environment: R 3.3.2, mlr 2.11, e1071 1.6.8.\n- rpart-5636: Learner mlr.classif.rpart from package(s) rpart. Environment: R 3.3.2, mlr 2.11, rpart 4.1.10.\n- rpart-5859: Learner mlr.classif.rpart from package(s) rpart. Environment: R 3.3.1, mlr 2.10, rpart 4.1.10.\n- glmnet-5860: Learner mlr.classif.glmnet from package(s) glmnet. Environment: R 3.3.1, mlr 2.10, glmnet 2.0.5.\n- svm-5891: Learner mlr.classif.svm from package(s) e1071. Environment: R 3.3.1, mlr 2.10, e1071 1.6.8.\n- xgboost-5906: Learner mlr.classif.xgboost from package(s) xgboost. Environment: R 3.3.1, mlr 2.10, xgboost 0.6.4.\n- ranger-5965: Learner mlr.classif.ranger from package(s) ranger. Environment: R 3.3.1, mlr 2.11, ranger 0.6.0.\n- glmnet-5970: Learner mlr.classif.glmnet from package(s) glmnet. Environment: R 3.3.1, mlr 2.11, glmnet 2.0.5.\n- xgboost-5971: Learner mlr.classif.xgboost from package(s) xgboost. Environment: R 3.3.1, mlr 2.11, xgboost 0.6.4.\n- glmnet-6766: Learner mlr.classif.glmnet from package(s) glmnet. Environment: R 3.3.2, mlr 2.11, glmnet 2.0.10.\n- xgboost-6767: Learner mlr.classif.xgboost from package(s) xgboost. Environment: R 3.3.2, mlr 2.11, xgboost 0.6.4.\n- ranger-6794: Learner mlr.classif.ranger from package(s) ranger. Environment: R 3.3.2, mlr 2.11, ranger 0.8.0.\n- ranger-7607: Learner mlr.classif.ranger from package(s) ranger. Environment: R 3.3.3, mlr 2.12, ranger 0.8.0.\n- ranger-7609: Learner mlr.classif.ranger from package(s) ranger. Environment: R 3.3.3, mlr 2.12, ranger 0.8.0.\n- ranger-5889: Learner mlr.classif.ranger from package(s) ranger. Environment: R 3.3.1, mlr 2.10, ranger 0.6.0.

The user will present their goal and optionally the data (e.g., pandas DataFrame) they already have. Your only mission is to identify what the user wants and transforms them into the input arguments of `suggestMachineLearningModule`. Your response should strictly follow the following format. Please do not add any additional content to the response. If you think the request is not solvable by `suggestMachineLearningModule`, write "I don't know." as the response.

```
{
    "existingModules": [
        {
            "role": "dataset",
            "module": {
                "name": "iris",
                "description": "The iris dataset is a classic and very easy multi-class classification dataset.",
            }
        },
        {
            "role": "taskType",
            "module": {
                "name": "classification",
                "description": "Classification task.",
        }
    ],
    "targetRole": "algorithm",
    "targetSchemaId": "rpart-preproc-4796"
}
```

"""

CODING_INSTRUCTION = """You're a data scientist. You're good at writing Python code to do data analysis, visualization, and machine learning. You can leverage the Python libraries such as `pandas`, `sklearn`, `matplotlib`, `seaborn`, and etc. to achieve user's request.

Specifically, the user will present a goal and optionally the data (e.g., pandas DataFrame) they already have, and several (around 3) suggestions from a machine learning expert, ordered from most confidence to least. Your task is to write a Python function to solve the user's request, taking the given data as input and returning the goal as output. The suggestions from the machine learning expert are a good reference for you.

Here are some coding instructions:

- The function should be wrapped by ``` before and after it.
- Import necessary libraries at the start of the code.
- The Python function should be named as `_coml_solution`.
- The input arguments of the function should be in the same order as that given by the user.
- The output of the function should be exactly what the user has asked for.
- Use type annotations for function arguments and return values (e.g., `df: pd.DataFrame` for a DataFrame input).
- If the function has multiple return values, return them as a tuple.
- Write the main code logic inside the function, keeping it clean and well-organized.
- Do not write any code outside the function, except for the `import` clauses.
- Do not provide examples of function usage.
- Do not use any global variables."""


def _smart_repr(value: Any) -> str:
    import numpy, pandas
    if isinstance(value, numpy.ndarray):
        return 'array(shape={})'.format(value.shape)
    elif isinstance(value, pandas.DataFrame):
        return 'dataframe(shape={}, columns={})'.format(value.shape, value.columns)
    elif isinstance(value, pandas.Series):
        return 'series(shape={})'.format(value.shape)
    elif isinstance(value, list):
        if len(value) > 30:
            return '[{}, ...]'.format(', '.join(_smart_repr(v) for v in value[:30]))
        return '[{}]'.format(', '.join(_smart_repr(v) for v in value))
    elif isinstance(value, dict):
        if len(value) > 30:
            return '{{{}, ...}}'.format(', '.join(f'{k}: {_smart_repr(v)}' for k, v in list(value.items())[:30]))
        return '{{{}}}'.format(', '.join(f'{k}: {_smart_repr(v)}' for k, v in value.items()))
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, (bool, int, float)):
        return str(value)
    elif value is None:
        return 'None'
    else:
        val = str(value)
        if len(val) > 100:
            val = val[:100] + '...'
        return val


class AgentBase:

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._messages: List[BaseMessage] = []

        self.reset()

    def reset(self):
        self._record_message(self._system_message())

    def _record_message(self, message: BaseMessage):
        self._messages.append(message)
        self._log_message(message)

    def _log_message(self, message: BaseMessage):
        if isinstance(message, HumanMessage):
            _logger.info("%s(%s) User:\n%s%s", Fore.GREEN, self.__class__.__name__, message.content, Style.RESET_ALL)
        elif isinstance(message, AIMessage):
            if not message.content and "function_call" in message.additional_kwargs:
                _logger.info("%s(%s) Assistant:\nFunction call:%sArguments:%s%s",
                             Fore.BLUE,
                             self.__class__.__name__,
                             message.additional_kwargs["function_call"].get("name"),
                             message.additional_kwargs["function_call"].get("arguments"),
                             Style.RESET_ALL)
            else:
                _logger.info("%s(%s) Assistant:\n%s%s", Fore.BLUE, self.__class__.__name__, message.content, Style.RESET_ALL)
        elif isinstance(message, FunctionMessage):
            _logger.info("%s(%s) Function result:\n%s%s", Fore.YELLOW, self.__class__.__name__, message.content, Style.RESET_ALL)
        elif isinstance(message, SystemMessage):
            _logger.info("%s(%s) System:\n%s%s", Fore.RED, self.__class__.__name__, message.content, Style.RESET_ALL)

    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def _system_message(self) -> SystemMessage:
        raise NotImplementedError()


class CoMLAgent(AgentBase):

    def _format_request(self, intention: CoMLIntention, arguments: List[Any]) -> str:
        if arguments:
            return f"Goal: {intention}\nData:\n" + "\n".join([
                f"  {i + 1}: {_smart_repr(arg)}" for i, arg in enumerate(arguments)
            ])
        else:
            return f"Goal: {intention}"

    def _system_message(self) -> SystemMessage:
        return SystemMessage(content=COML_INSTRUCTION)

    def __call__(self, intention: CoMLIntention, arguments: List[Any]) -> Optional[List[dict]]:
        self._record_message(HumanMessage(content=self._format_request(intention, arguments)))
        result = self.llm(self._messages)
        self._record_message(result)

        if result.content == "I don't know.":
            return None

        parameters = json.loads(result.content)
        if "existingModules" in parameters and "targetRole" in parameters:
            schema = parameters["targetSchemaId"] if "targetSchemaId" in parameters else None
            result = suggest_machine_learning_module(
                parameters["existingModules"],
                parameters["targetRole"],
                schema
            )
            self._record_message(AIMessage(content=json.dumps(result, indent=2)))
            return result

        return None


class CodingAgent(AgentBase):

    def __init__(self, llm: BaseChatModel, coml_agent: CoMLAgent):
        super().__init__(llm)
        self.coml_agent = coml_agent

    def _system_message(self) -> SystemMessage:
        return SystemMessage(content=CODING_INSTRUCTION)

    def _format_request(self, intention: CoMLIntention, arguments: List[Any],
                        coml_response: Optional[List[dict]]) -> str:
        if arguments:
            request = f"User request: {intention}\nArguments:\n" + "\n".join([
                f"  {i + 1}: {_smart_repr(arg)}" for i, arg in enumerate(arguments)
            ])
        else:
            request = f"User request: {intention}\nNo input argument."
        if coml_response is not None:
            request += "\nML expert recommendations:\n" + json.dumps(coml_response, indent=2)
        return request

    def __call__(self, intention: CoMLIntention, arguments: List[Any]):
        coml_response = self.coml_agent(intention, arguments)
        self._record_message(
            HumanMessage(content=self._format_request(intention, arguments, coml_response))
        )

        result = self.llm(self._messages)
        self._record_message(result)
