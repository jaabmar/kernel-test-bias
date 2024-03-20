import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from datasets.bias_models import (
    add_bias_scenario_1,
    add_bias_scenario_2,
    add_bias_subgroups,
)
from datasets.hillstrom import (
    CAT_COVAR_HILLSTROM,
    NUM_COVAR_HILLSTROM,
    load_fetch_hillstrom_data,
)


def load_and_preprocess_hillstrom_data(size_prop=0.8, bias_model="scenario_1", **bias_args):
    obs_data, rct_data = load_fetch_hillstrom_data(
        support_var="zip_code",
        split_data=True,
        support_feature_values=[0, 1, 2],
        proportion_full_support=size_prop,
        seed=52,
        target_col="spend",
    )

    numeric_covariates = NUM_COVAR_HILLSTROM
    categorical_covariates = CAT_COVAR_HILLSTROM

    # Define transformer for encoding and normalization
    transformer = ColumnTransformer(
        [
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
                [col for col in categorical_covariates],
            ),
            ("normalizer", MinMaxScaler(), [col for col in numeric_covariates]),
        ]
    )

    print("RCT num. samples: ", rct_data.shape[0])
    print()

    # Apply selected bias model
    if bias_model == "scenario_1":
        obs_data = add_bias_scenario_1(obs_data, **bias_args)
    elif bias_model == "scenario_2":
        obs_data = add_bias_scenario_2(obs_data, **bias_args)
    elif bias_model == "subgroups":
        obs_data = add_bias_subgroups(obs_data, **bias_args)
    else:
        raise ValueError("Invalid bias model selected")

    # Feature transformation
    x_obs_encoded = transformer.fit_transform(obs_data.drop(["Y", "T"], axis=1))
    x_rct_encoded = transformer.transform(rct_data.drop(["Y", "T"], axis=1))

    # Prepare data for analysis
    t_rct, y_rct, x_rct = rct_data["T"].values, rct_data["Y"].values, x_rct_encoded
    t_obs, y_obs, x_obs = obs_data["T"].values, obs_data["Y"].values, x_obs_encoded

    x = np.concatenate((x_rct, x_obs))
    y = np.concatenate((y_rct, y_obs)).reshape(-1, 1)
    t = np.concatenate((t_rct, t_obs)).reshape(-1, 1)
    s = np.concatenate((np.zeros(x_rct.shape[0]), np.ones(x_obs.shape[0])))

    return x, y, t, s


def estimate_selection_score(x, s, seed):
    clf = RandomForestClassifier(max_depth=5, random_state=seed)
    clf.fit(x, s)
    return clf.predict_proba(x)


def initialize_model(input_size, features, lr, seed):
    if isinstance(features, int):
        features = [features]

    model = NeuralNetwork(features=features, seed=seed)
    params = model.init(jax.random.PRNGKey(seed), jnp.ones((input_size,)))
    tx = optax.adam(learning_rate=lr)
    opt_state = tx.init(params)
    return model, params, tx, opt_state


def get_subset_features(x, subset):
    feature_indices = {
        "channel": slice(9, 12),
        "history": -1,
        "newbie": slice(7, 9),
        "zip_code": slice(4, 7),
        "womens": slice(2, 4),
        "mens": slice(2),
    }
    selected_features = []
    if subset >= 1:
        selected_features.append(x[:, feature_indices["channel"]])
    if subset >= 2:
        selected_features.append(x[:, feature_indices["newbie"]])
    if subset >= 3:
        selected_features.append(x[:, feature_indices["mens"]])
    if subset >= 4:
        selected_features.append(x[:, feature_indices["zip_code"]])
    if subset >= 5:
        selected_features.append(x[:, feature_indices["womens"]])
    if subset >= 6:
        selected_features.append(x[:, [feature_indices["history"]]])
    if subset > 6:
        num_samples = x.shape[0]  # Number of rows in x
        num_dummy_features = subset - 6  # Number of dummy features to add
        for _ in range(num_dummy_features):
            # Generate a dummy feature for each extra subset
            dummy_feature = np.random.normal(loc=0, scale=1, size=(num_samples, 1))
            selected_features.append(dummy_feature)
    # Construct x_kernel by concatenating the selected features
    x_kernel = np.concatenate(selected_features, axis=1)
    return x_kernel


class NeuralNetwork(nn.Module):
    features: list  # List of integers defining the number of neurons in each layer
    seed: int = 0  # Random seed for initialization

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(features=feat)(x)
            if i < len(self.features) - 1:  # Apply ReLU activation to all but the last layer
                x = nn.relu(x)
        # The output layer
        x = nn.Dense(features=1)(x)
        out = nn.sigmoid(x)
        return out
