import numpy as np
import pandas as pd
import pytest

from mlcopilot.surrogate_utils import (
    evaluate_configs,
    process_history_df,
    train_surrogate,
)


def test_process_history_df():
    history_df = pd.read_csv("assets/example_history.csv")
    history_df_processed, config_names = process_history_df(history_df)
    assert history_df_processed.shape == (340, 7)
    assert set(config_names) == {
        "cost",
        "gamma",
        "degree",
        "kernel__DUMM_SEP__linear",
        "kernel__DUMM_SEP__polynomial",
        "kernel__DUMM_SEP__radial",
    }
    return history_df_processed, config_names


def test_train_surrogate():
    history_df_processed, config_names = test_process_history_df()
    surrogate = train_surrogate(history_df_processed)
    assert surrogate.predict(np.random.rand(10, 6)).shape == (10,)
    return surrogate, config_names


def test_evaluate_configs():
    surrogate, config_names = test_train_surrogate()
    configs = [
        {"cost": 0.1, "gamma": 10, "kernel": "linear"},
        {"cost": 10, "gamma": 0.1, "degree": 5, "kernel": "polynomial"},
        {"cost": 5, "gamma": 2, "kernel": "radial"},
    ]
    scores = evaluate_configs(surrogate, configs, config_names)
    assert scores.shape == (3,)
    return scores
