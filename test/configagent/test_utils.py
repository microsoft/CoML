import pytest

from coml.configagent.utils import format_config, parse_configs


def test_parser_configs():
    configs = parse_configs(
        "Configuration 1: learning rate is 0.01. batch size is 32.\n"
        "Configuration 2: learning rate is 0.02. batch size is 64.\n"
        "Configuration 3: learning rate is 0.03. batch size is 128.\n",
        3,
    )
    assert configs == [
        {"learning rate": "0.01", "batch size": "32"},
        {"learning rate": "0.02", "batch size": "64"},
        {"learning rate": "0.03", "batch size": "128"},
    ]


def test_format_config():
    config = format_config(
        {"learning rate": "0.01", "batch size": "32"},
    )
    assert config == "learning rate is 0.01. batch size is 32."
