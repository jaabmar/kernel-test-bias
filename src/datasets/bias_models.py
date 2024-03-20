import numpy as np
import pandas as pd


def add_bias_scenario_1(
    obs_data,
    bias_for_channel_0=-40,
    bias_for_channel_1=-40,
    bias_for_channel_2=40,
    bias_for_mens=20,
    bias_for_newbie=-10,
):
    # Ensure input is DataFrame
    data = pd.DataFrame(obs_data).copy()
    # Apply biases to observed data
    channel_conditions = [(data["channel"] == i) for i in range(3)]
    biases = [bias_for_channel_0, bias_for_channel_1, bias_for_channel_2]
    for condition, bias in zip(channel_conditions, biases):
        data.loc[(data["T"] == 1) & condition, "Y"] += bias

    data.loc[(data["T"] == 1) & (data["mens"] == 1), "Y"] += bias_for_mens
    data.loc[(data["T"] == 1) & (data["newbie"] == 1), "Y"] += bias_for_newbie

    return data


def add_bias_scenario_2(obs_data, seed=50, degree=2):
    np.random.seed(seed)  # Set the random seed for reproducibility
    data = pd.DataFrame(obs_data).copy()

    # Standard deviation adjustment based on the polynomial degree
    std_dev = 0.01**degree
    # Generating coefficients for polynomial biases for newbie status 0 and 1
    coeffs_newbie_0, coeffs_newbie_1 = [np.random.normal(0, std_dev, degree + 1) for _ in range(2)]

    # Creating a matrix of powers of "history" column
    history_powers = np.vstack([data["history"] ** i for i in range(degree + 1)]).T

    # Computing the biases for each newbie status using the dot product of history powers and coefficients
    bias_newbie_0_obs = np.tanh(history_powers @ coeffs_newbie_0) * 100
    bias_newbie_1_obs = np.tanh(history_powers @ coeffs_newbie_1) * 100

    # Apply the computed biases based on 'newbie' value
    data.loc[(data["T"] == 1) & (data["newbie"] == 0), "Y"] += bias_newbie_0_obs[
        (data["T"] == 1) & (data["newbie"] == 0)
    ]
    data.loc[(data["T"] == 1) & (data["newbie"] == 1), "Y"] += bias_newbie_1_obs[
        (data["T"] == 1) & (data["newbie"] == 1)
    ]

    return data


def add_bias_subgroups(obs_data, mode=3, bias=60):
    data = pd.DataFrame(obs_data).copy()
    bias_values = bias * np.ones(len(data))

    # Define masks for different modes
    modes = {
        0: data["T"] == 1,
        1: (data["channel"] == 1) & (data["T"] == 1),
        2: (data["newbie"] == 1) & (data["channel"] == 1) & (data["T"] == 1),
        3: (data["mens"] == 1) & (data["newbie"] == 1) & (data["channel"] == 1) & (data["T"] == 1),
        4: (data["history"] < 70)
        & (data["mens"] == 1)
        & (data["newbie"] == 1)
        & (data["channel"] == 1)
        & (data["T"] == 1),
        5: (data["zip_code"] == 2)
        & (data["history"] < 70)
        & (data["mens"] == 1)
        & (data["newbie"] == 1)
        & (data["channel"] == 1)
        & (data["T"] == 1),
    }
    mask = modes.get(mode, data["T"] == 1)  # Default to mode 0 if mode is not recognized
    data.loc[mask, "Y"] += bias_values[mask]
    return data
