import os
from pathlib import Path

__all__ = [
    "COML_DB_PATH",
    "TOP_K",
    "EMBED_DIM",
    "bin_map",
    "inverse_bin_map",
    "q_num",
    "COML_DB_BACKEND",
    "COML_DB_NAME",
    "COML_DB_HOST",
    "COML_DB_PORT",
    "COML_DB_USER",
    "COML_DB_PASSWORD",
    "PROMPT_FORMATS",
    "DEFAULT_PROMPT_PREFIX",
    "DEFAULT_PROMPT_SUFFIX",
    "TOKEN_LIMIT",
    "TOKEN_COMPLETION_LIMIT",
    "RELAX_TOKEN",
]

TOP_K = 3
EMBED_DIM = 1536
TOKEN_LIMIT = 4096
TOKEN_COMPLETION_LIMIT = 800
RELAX_TOKEN = 500  # RELAX_TOKEN is the number of tokens to void token limit

COML_DB_BACKEND = os.environ.get("COML_DB_BACKEND", "sqlite")

COML_DB_PATH = Path(
    os.environ.get("COML_DB_PATH", Path.home() / ".coml" / "coml.db")
).expanduser()

COML_DB_NAME = os.environ.get("COML_DB_NAME", "coml")
COML_DB_HOST = os.environ.get("COML_DB_HOST", "localhost")
COML_DB_PORT = os.environ.get("COML_DB_PORT", 5432)
COML_DB_USER = os.environ.get("COML_DB_USER", "postgres")
COML_DB_PASSWORD = os.environ.get("COML_DB_PASSWORD", "")

bin_map = {
    0.1: "very small",
    0.3: "small",
    0.5: "medium",
    0.7: "large",
    0.9: "very large",
}

inverse_bin_map = {v: k for k, v in bin_map.items()}
inverse_bin_map.update(
    {
        "very low": 0.1,
        "low": 0.3,
        "high": 0.7,
        "very high": 0.9,
        "extremely large": 0.9,
        "extremely small": 0.1,
        "extra large": 0.9,
        "extra small": 0.1,
        "very medium": 0.5,
        "very small": 0.1,
        "small": 0.3,
        "large": 0.7,
        "very large": 0.9,
    }
)

q_num = sorted(list(bin_map.keys()))

PROMPT_FORMATS = {
    "TOP_K",
    "knowledge",
    "space_desc",
    "new_task_desc",
}

DEFAULT_PROMPT_PREFIX = """{space_desc}\nRecommend best configurations to train a model for a new task. Format strictly follows this template: ```Configuration 1: {{parameter_1_name}} is {{parameter_1_value}}. {{parameter_2_name}} is {{parameter_2_value}}...{{parameter_n_name}} is {{parameter_n_value}}.
Configuration 2: {{parameter_1_name}} is {{parameter_1_value}}. {{parameter_2_name}} is {{parameter_2_value}}...{{parameter_n_name}} is {{parameter_n_value}}.
Configuration 3: {{parameter_1_name}} is {{parameter_1_value}}. {{parameter_2_name}} is {{parameter_2_value}}...{{parameter_n_name}} is {{parameter_n_value}}.
```\nHere are some tasks along with best hyper-parameter configurations to train a model on them.\n"""

DEFAULT_PROMPT_SUFFIX = """\nGuidelines:{knowledge}\n\n\nBased on the examples(if provided) and guidelines(if provided) above, recommend {TOP_K} hyper-parameter configurations for a new classification dataset.\n\n{new_task_desc}"""
