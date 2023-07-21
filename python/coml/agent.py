import json
import logging
from typing import Any, List

from colorama import Fore, Style
from langchain.chat_models.base import BaseChatModel
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage, FunctionMessage

from .node import get_function_description, suggest_machine_learning_module

_logger = logging.getLogger(__name__)

CoMLIntention = str

INSTRUCTION = "You're a data scientist. " \
    "You're good at writing Python code to do data analysis, visualization, and machine learning. " \
    "You can leverage the Python libraries such as `pandas`, `sklearn`, `matplotlib`, `seaborn`, and etc. to achieve user's request. " \
    "For every question and request, please write a Python function to solve it.\n\n" \
    "Here are some instructions on how to write the Python code:\n\n" \
    """- The function should be wrapped by ``` before and after it.
- Import necessary libraries at the start of the code.
- The Python function should be named as `_coml_solution`.
- The input arguments of the function should be in the same order as that given by the user.
- The output of the function should be exactly what the user has asked for.
- Use type annotations for function arguments and return values (e.g., `df: pd.DataFrame` for a DataFrame input).
- If the function has multiple return values, return them as a tuple.
- Write the main code logic inside the function, keeping it clean and well-organized.
- Do not write any code outside the function, except for the `import` clauses.
- Do not provide examples of function usage.
- Do not use any global variables.""" \
    "\n\nTo make the code better satisfy the user's request, you will be given a tool called `suggestMachineLearningModule`. " \
    "To use this tool, your should first identify the datasets, models, task types, and other existing components specified by the users, if provided. " \
    "Additionally, you should determine the specific type of module that users are interested in."


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


class CoMLAgent:

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self._messages: List[BaseMessage] = []

        self.reset()

    def reset(self):
        self._messages = [SystemMessage(content=INSTRUCTION)]

    def _format_request(self, intention: CoMLIntention, arguments: List[Any]) -> str:
        if arguments:
            return f"User request: {intention}\nArguments:\n" + "\n".join([
                f"  {i + 1}: {_smart_repr(arg)}" for i, arg in enumerate(arguments)
            ])
        else:
            return f"User request: {intention}\nNo input argument."

    def _record_message(self, message: BaseMessage):
        self._messages.append(message)
        self._log_message(message)

    def __call__(self, intention: CoMLIntention, arguments: List[Any]) -> Any:
        request_string = self._format_request(intention, arguments)
        self._record_message(HumanMessage(content=request_string))
        self._log_message(self._messages[-1])
        self._trigger_llm_call()

    def _trigger_llm_call(self):
        result = self.llm(
            self._messages,
            functions=[get_function_description()],
            function_call="none" if isinstance(self._messages[-1], FunctionMessage) else "auto"    
        )
        self._log_message(result)

        self._messages.append(result)
        if ("function_call" in result.additional_kwargs and
                result.additional_kwargs["function_call"].get("name") == "suggestMachineLearningModule" and
                result.additional_kwargs["function_call"].get("arguments")):
            args = json.loads(result.additional_kwargs["function_call"]["arguments"])
            if "existingModules" in args and "targetRole" in args:
                schema = args["targetSchemaId"] if "targetSchemaId" in args else None
                result = suggest_machine_learning_module(args["existingModules"], args["targetRole"], schema)
                self._log_message(
                    FunctionMessage(
                        name="suggestMachineLearningModule",
                        content=json.dumps(result, indent=2)
                    )
                )
                self._log_message(
                    HumanMessage(
                        content="Please write a Python function to solve user's request. "
                            "You can use the suggested module above as a reference."
                    )
                )
                self._trigger_llm_call()

    def _log_message(self, message: BaseMessage):
        if isinstance(message, HumanMessage):
            _logger.info("%sUser: %s%s", Fore.GREEN, message.content, Style.RESET_ALL)
        elif isinstance(message, AIMessage):
            if not message.content and "function_call" in message.additional_kwargs:
                _logger.info("%sAssistant:\nFunction call:%sArguments:%s%s",
                             Fore.BLUE,
                             message.additional_kwargs["function_call"].get("name"),
                             message.additional_kwargs["function_call"].get("arguments"),
                             Style.RESET_ALL)
            else:
                _logger.info("%sAssistant: %s%s", Fore.BLUE, message.content, Style.RESET_ALL)
        elif isinstance(message, FunctionMessage):
            _logger.info("%sML expert: %s%s", Fore.YELLOW, message.content, Style.RESET_ALL)
