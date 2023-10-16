import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

from .constants import *

LLM_MODELS = {
    "suggest": lambda: OpenAI(model_name="text-davinci-003", temperature=0),
    "knowledge": lambda: OpenAI(model_name="text-davinci-003"),
    "embedding": lambda: OpenAIEmbeddings(),
}
_TOKEN_COUNT_FUNC = None


def clean_input(prompt: str = ""):
    try:
        return input(prompt)
    except KeyboardInterrupt:
        print("You interrupted CoML")
        print("Quitting...")
        exit(0)


pattern_0 = re.compile("Configuration(?: \d)*: (.*)\.\n")


def parse_configs(
    response: str,
    TOP_k: int,
    inverse_bin_map: Dict = {},
    quantile_info: Optional[Dict] = None,
) -> List[Dict]:
    """
    Parse the response from the LLM API and return the suggested configurations.

    Parameters
    ----------
    response : str
        The response from the LLM API.
    TOP_k : int
        The number of suggested configurations to return.
    inverse_bin_map : Dict
        The inverse bin map from the Space object.
    quantile_info : Optional[Dict]
        The meta train info for statistics from the Space object.

    Returns
    -------
    suggest_configs : List[Dict]
        The suggested configurations.
    """
    suggest_configs = []
    groups = re.findall(pattern_0, response + "\n")
    for t in groups[:TOP_k]:
        kvs = t.split(". ")
        config = {}
        for kv in kvs:
            _k, v = kv.strip().split(" is ")
            if v in inverse_bin_map:
                config_col = list(quantile_info[_k])
                value = config_col[q_num.index(inverse_bin_map[v])]
            elif v in ("True", "False"):
                value = eval(v)
            else:
                value = v
            config[_k] = value

        suggest_configs.append(config)
    return suggest_configs


def format_config(
    config: Dict[str, Any],
    quantile_info: Optional[Dict[str, List[float]]] = None,
    bin_map: Dict[float, str] = {},
) -> str:
    """
    Format the configuration to a string which can be input to the LLM API.

    Parameters
    ----------
    config
        The configuration to be formatted.
    quantile_info
        The meta train info for statistics from the Space object.
    bin_map
        The bin map is to map the bin value to the string.

    Returns
    -------
    config_str : str
        The formatted configuration.
    """
    result = []
    for k, v in config.items():
        _k = k
        if v is None:
            continue
        elif v in ["TRUE", "FALSE"]:
            result.append(f"{_k} is {v.lower().capitalize()}")
        elif v is True or v is False:
            v = str(v)
            result.append(f"{_k} is {v.lower().capitalize()}")
        elif isinstance(v, str):
            result.append(f"{_k} is {v}")
        elif isinstance(v, (float, int)):
            assert quantile_info is not None, "quantile_info is None"
            config_col = list(quantile_info[k])
            anchor = min(config_col, key=lambda x: abs(x - v))
            value = bin_map[q_num[config_col.index(anchor)]]
            result.append(f"{_k} is {value}")

        else:
            assert False, f"{v}"
    return ". ".join(result) + "."


@lru_cache(maxsize=1000)
def _token_count(text):
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except KeyError:
        print("Warning: model not found. Using gpt-2 encoding.")
        encoding = tiktoken.get_encoding("gpt2")
    return len(encoding.encode(text))


def token_count(texts):
    if isinstance(texts, str):
        return _token_count(texts)
    l = 0
    for text in texts:
        l += _token_count(text)
    return l


def set_token_count_func(func):
    global _TOKEN_COUNT_FUNC
    _TOKEN_COUNT_FUNC = func


def get_token_count_func():
    global _TOKEN_COUNT_FUNC
    return _TOKEN_COUNT_FUNC


def get_llm(model_type: str):
    return LLM_MODELS[model_type]


def set_llms(
    suggest_model: Optional[Callable] = None,
    knowledge_model: Optional[Callable] = None,
    embedding_model: Optional[Callable] = None,
):
    global LLM_MODELS
    if suggest_model is not None:
        LLM_MODELS["suggest"] = suggest_model
    if knowledge_model is not None:
        LLM_MODELS["knowledge"] = knowledge_model
    if embedding_model is not None:
        LLM_MODELS["embedding"] = embedding_model


def escape(text: str) -> str:
    return re.sub("(?<!{)\{(.*?)\}(?!})", r"{{\1}}", text)


set_token_count_func(token_count)
