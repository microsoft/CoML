import json
import logging
from typing import Any, List, Optional, Union, Dict

from colorama import Fore, Style
from langchain.chat_models.base import BaseChatModel
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage, FunctionMessage

from .node import get_function_description, suggest_machine_learning_module
# from .node_mock import get_function_description, suggest_machine_learning_module
from .prompt import COML_INSTRUCTION, COML_EXAMPLES

_logger = logging.getLogger(__name__)

CoMLIntention = str

CODING_INSTRUCTION = """You're a data scientist. You're good at writing Python code to do data analysis, visualization, and machine learning. You can leverage the Python libraries such as `pandas`, `sklearn`, `matplotlib`, `seaborn`, and etc. to achieve user's request.

Specifically, the user will present a goal and optionally the data (e.g., pandas DataFrame) they already have, and several (around 3) suggestions from a machine learning expert. Your task is to write a Python function to solve the user's request, taking the given data as input and returning the goal as output. The suggestions from the machine learning expert are ordered from the most confident to least. They only serve as a reference and you don't have to use them all.

Some extra coding instructions:

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

    def _record_message(self, message: Union[List[BaseMessage], BaseMessage]):
        if isinstance(message, list):
            for m in message:
                self._record_message(m)
        else:
            self._messages.append(message)
            self._log_message(message)

    def _log_message(self, message: BaseMessage):
        if isinstance(message, HumanMessage):
            _logger.info("%s(%s) User:\n%s%s", Fore.GREEN, self.__class__.__name__, message.content, Style.RESET_ALL)
        elif isinstance(message, AIMessage):
            if not message.content and "function_call" in message.additional_kwargs:
                _logger.info("%s(%s) Assistant:\nFunction call: %s\nArguments: %s%s",
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

    def _system_message(self) -> List[BaseMessage]:
        raise NotImplementedError()


class CoMLAgent(AgentBase):

    def _format_request(self, intention: CoMLIntention, *arguments: Any, **kwargs: Any) -> str:
        if arguments or kwargs:
            return f"Goal: {intention}\nData:\n" + "\n".join([
                f"  - {_smart_repr(arg)}" for arg in arguments
            ] + [
                f"  - {k}: {_smart_repr(v)}" for k, v in kwargs.items()
            ])
        else:
            return f"Goal: {intention}"

    def _system_message(self) -> List[BaseMessage]:
        messages: List[BaseMessage] = [SystemMessage(content=COML_INSTRUCTION)]
        for example in COML_EXAMPLES:
            messages.append(HumanMessage(
                content=self._format_request(
                    example["goal"], *example["data"]
                )
            ))
            messages.append(AIMessage(content=json.dumps(example["response"], indent=2)))
        return messages

    def __call__(self, intention: CoMLIntention, *args: Any, **kwargs: Any) -> Optional[List[dict]]:
        self._messages = []
        self._record_message(self._system_message())

        self._record_message(HumanMessage(content=self._format_request(intention, *args, **kwargs)))
        result = self.llm(self._messages)
        self._record_message(result)
        if result.content == "I don't know.":
            return None

        arguments = json.loads(result.content)
        if "existingModules" in arguments and "targetRole" in arguments:
            schema = arguments["targetSchemaId"] if "targetSchemaId" in arguments else None
            result = suggest_machine_learning_module(
                arguments["existingModules"],
                arguments["targetRole"],
                schema)
            self._log_message(
                FunctionMessage(
                    name="suggestMachineLearningModule",
                    content=json.dumps(result, indent=2)
                )
            )
            return result

        return None


class CodingAgent(AgentBase):

    def __init__(self, llm: BaseChatModel, coml_agent: CoMLAgent):
        super().__init__(llm)
        self.coml_agent = coml_agent

    def _system_message(self) -> List[BaseMessage]:
        return [
            SystemMessage(content=CODING_INSTRUCTION)
        ]

    def _format_request(
        self, intention: CoMLIntention,
        positional_arguments: List[Any],
        keyword_arguments: Dict[str, Any],
        coml_response: Optional[List[dict]]
    ) -> str:
        request = f"User request: {intention}"
        if positional_arguments:
            request += "\nPositional arguments:\n" + "\n".join([
                f"  {i}: {_smart_repr(arg)}" for i, arg in enumerate(positional_arguments)
            ])
        if keyword_arguments:
            request += "\nKeyword arguments:\n" + "\n".join([
                f"  {k}: {_smart_repr(v)}" for k, v in keyword_arguments.items()
            ])
        if coml_response is not None:
            request += "\nML expert recommendations:\n" + json.dumps(coml_response, indent=2)
        return request

    def __call__(self, intention: CoMLIntention, *args: Any, **kwargs: Any):
        if not self._messages:
            self._record_message(self._system_message())

        coml_response = self.coml_agent(intention, *args, **kwargs)
        self._record_message(
            HumanMessage(content=self._format_request(intention, list(args), kwargs, coml_response))
        )

        result = self.llm(self._messages)
        self._record_message(result)
