from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

prefix_sep = "__DUMM_SEP__"


def process_history_df(history_df: pd.DataFrame):
    """
    Process the history dataframe to be used for training the surrogate model.

    Parameters
    ----------
    history_df: pd.DataFrame
        The history dataframe, format see `coml.experience.ingest_experience`.

    Returns
    -------
    history_df_: pd.DataFrame
        The processed history dataframe.
    """
    history_df_ = history_df
    score = history_df_["SCORE"].astype(float)
    history_df_ = history_df_.rename(
        columns=lambda x: x[7:] if x.startswith("CONFIG_") else None
    )
    del history_df_[None]
    history_df_["SCORE"] = score
    history_df_ = history_df_[history_df_["SCORE"].notna()]
    # history_df_ = history_df_.fillna(0)
    history_df_ = pd.get_dummies(history_df_, prefix_sep=prefix_sep)

    config_names = list(history_df_.drop(columns=["SCORE"]).columns)
    return history_df_, config_names


def train_surrogate(history_df_processed: pd.DataFrame) -> Callable:
    """
    Train a surrogate model based on the history of the user.

    Parameters
    ----------
    history_df_processed: pd.DataFrame
        The processed history dataframe.

    Returns
    -------
    surrogate: Callable
        The surrogate model.
    """
    train_df, test_df = train_test_split(
        history_df_processed, test_size=0.2, random_state=42
    )
    # normalize the score
    train_df["SCORE"] = (train_df["SCORE"] - train_df["SCORE"].min()) / (
        train_df["SCORE"].max() - train_df["SCORE"].min() + 1.0e-4
    )
    train_X = train_df.drop(columns=["SCORE"])
    train_y = train_df["SCORE"]
    test_X = test_df.drop(columns=["SCORE"])
    test_y = test_df["SCORE"]
    surrogate = Pipeline(
        [
            ("scaler", RobustScaler()),
            ("xgb", xgb.XGBRegressor(objective="reg:squarederror")),
        ]
    )
    surrogate.fit(train_X, train_y)
    preds = surrogate.predict(test_X)
    return surrogate


def evaluate_configs(
    surrogate_fn: Callable, configs: List[Dict[str, Any]], config_names: List[str]
) -> np.ndarray:
    """
    Evaluate a list of configs using the surrogate model.

    Parameters
    ----------
    surrogate_fn: Callable
        The surrogate model.
    configs
        The list of configs to be evaluated.
    config_names: List[str]
        The list of config names.

    Returns
    -------
    scores: np.ndarray
        The scores of the configs. It should be 1-dimensional.
    """
    df = pd.DataFrame(columns=config_names)
    dumm_cols = set([n[: n.index(prefix_sep)] for n in config_names if prefix_sep in n])
    for config in configs:
        # add one NaN row
        idx = len(df)
        df.loc[idx] = np.nan
        for k, v in config.items():
            if k in dumm_cols:
                new_k = k + prefix_sep + v
                assert new_k in config_names, "Unknown config name: {}".format(new_k)
                df.loc[idx, new_k] = 1
                for _k in config_names:
                    if _k != new_k and _k.startswith(k + prefix_sep):
                        df.loc[idx, _k] = 0
            else:
                assert k in config_names, "Unknown config name: {}".format(k)
                df.loc[idx, k] = v
    return surrogate_fn.predict(df).ravel()
