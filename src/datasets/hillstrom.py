from copy import copy
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklift.datasets import fetch_hillstrom

CAT_COVAR_HILLSTROM = [
    # "recency",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
]
NUM_COVAR_HILLSTROM = ["history"]


def load_fetch_hillstrom_data(
    conf_var: Optional[str] = None,
    support_var: str = "zip_code",
    split_data: bool = True,
    support_feature_values: Optional[Union[list, tuple]] = None,
    proportion_full_support: float = 0.5,
    seed: int = 42,
    target_col: str = "visit",
):
    x, y, t = fetch_hillstrom(target_col=target_col, return_X_y_t=True)
    cat_covar_columns = copy(CAT_COVAR_HILLSTROM)
    num_covar_columns = copy(NUM_COVAR_HILLSTROM)
    all_covar_columns = cat_covar_columns + num_covar_columns
    if conf_var is not None:
        confounder = x[conf_var].values
    else:
        confounder = None
    # Remove the confounding variable from the list of covariate columns
    if conf_var in all_covar_columns:
        all_covar_columns.remove(conf_var)
    if conf_var in num_covar_columns:
        num_covar_columns.remove(conf_var)
    elif conf_var in cat_covar_columns:
        cat_covar_columns.remove(conf_var)

    # Convert categorical columns to integers using Label Encoding
    le = LabelEncoder()
    for column in cat_covar_columns:
        x[column] = le.fit_transform(x[column])

    X_all = x[all_covar_columns].values
    Y_all = y.values
    T_all = np.where(t.values == "No E-Mail", 0, 1)

    if confounder is not None:
        data = {
            "Y": Y_all,
            "T": T_all,
            "C": confounder,
        }
    else:
        data = {
            "Y": Y_all,
            "T": T_all,
        }

    X_df = pd.DataFrame(X_all, columns=all_covar_columns)
    data.update(X_df)
    df = pd.DataFrame(data)

    if split_data:
        if conf_var == support_var:
            raise ValueError("conf_var and support_var cannot have the same value")
        assert support_var in cat_covar_columns
        return split_dataset(
            df=df,
            support_feature=support_var,
            support_feature_values=support_feature_values,
            proportion_full_support=proportion_full_support,
            seed=seed,
        )

    return df


def split_dataset(
    df,
    support_feature="g1surban",
    support_feature_values=None,
    proportion_full_support=0.5,
    seed=42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if support_feature_values is None:
        support_feature_values = [1, 2]

    # Perform train-test split with probability q
    full_support_df, reduced_support_df = train_test_split(
        df, test_size=1 - proportion_full_support, random_state=seed
    )

    # Keep only the points with specific values of support_feature_values in reduced_support_df
    final_reduced_support_df = reduced_support_df[
        reduced_support_df[support_feature].isin(support_feature_values)
    ]
    excluded_datapoints = reduced_support_df[
        ~reduced_support_df[support_feature].isin(support_feature_values)
    ]

    if len(excluded_datapoints) == 0:
        final_full_support_df = full_support_df
    else:
        final_full_support_df = pd.concat([full_support_df, excluded_datapoints])

    return final_full_support_df.reset_index(drop=True), final_reduced_support_df.reset_index(
        drop=True
    )
