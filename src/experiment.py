import argparse

import numpy as np
import optax
import scipy.stats as stats
from jax import value_and_grad
from sklearn.linear_model import LogisticRegression

from experiment_utils import (
    estimate_selection_score,
    get_subset_features,
    initialize_model,
    load_and_preprocess_hillstrom_data,
)
from tests.ate_test import ate_test
from tests.kernel_test import KernelLossCalculator, construct_kernel_matrix


def experiment():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run the training model.")
    parser.add_argument(
        "--test_type",
        type=str,
        choices=["kernel_test", "ate_test"],
        default="kernel_test",
        help="Specifies the type of test to conduct.",
    )
    parser.add_argument(
        "--bias_model",
        type=str,
        choices=["scenario_1", "scenario_2", "subgroups"],  # Restricts to these specific scenarios
        default="scenario_1",
        help="Specifies the bias model to use.",
    )
    parser.add_argument(
        "--user_shift",
        type=float,
        default=10.0,
        help="User-defined shift value (must be positive).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=2,
        help="List of integers representing the size of each layer.",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        choices=["gaussian", "laplacian"],
        default="laplacian",
        help="Kernel type for the model.",
    )
    parser.add_argument(
        "--kernel_param",
        type=float,
        default=1.0,
        help="Parameter for the kernel function.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the hypothesis test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility.",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=6,
        help="Number of features responsible for heterogeneity.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Additional validations
    if args.user_shift <= 0:
        parser.error("--user_shift must be positive.")

    # Load and preprocess data
    print("Loading and preprocessing data.")
    np.random.seed(args.seed)
    x, y, t, s = load_and_preprocess_hillstrom_data(bias_model=args.bias_model)
    # Shuffle the data
    p = np.random.permutation(len(x))
    x, y, t, s = x[p], y[p], t[p], s[p]
    # Estimate nuisance functions
    logreg = LogisticRegression(random_state=args.seed)
    logreg.fit(x[s == 1], t[s == 1].ravel())
    e_x = logreg.predict_proba(x)[:, 1]
    print("Data loaded and preprocessed.")
    if args.test_type == "ate_test":
        print("Running ATE test.")
        test_stat = ate_test(x, y, t, s, e_x, bias=args.user_shift, seed=args.seed, bootstraps=500)
        alpha = stats.norm.ppf(args.alpha / 2)
        print()
        print("Test stat:", test_stat[0])
        if test_stat < alpha:
            print("Reject the null hypothesis.")
        else:
            print("Accept the null hypothesis.")
    else:  # kernel test!
        print("Running kernel test.")
        print("Constructing kernel matrix.")
        # Estimate selection score
        pi_s = estimate_selection_score(x, s, seed=args.seed)
        # Construct kernel matrix
        x_kernel = get_subset_features(x, args.num_features)
        kernel_matrix = construct_kernel_matrix(
            x_kernel[s == 0],
            kernel_type=args.kernel,
            kernel_param=args.kernel_param,
        )
        # Initialize model parameters, optimizer and loss tracker
        model, params, tx, opt_state = initialize_model(
            input_size=x_kernel.shape[1],
            features=args.layers,
            lr=args.lr,
            seed=args.seed,
        )

        loss_calculator = KernelLossCalculator(
            model,
            params,
            x,
            y,
            t,
            s,
            e_x=e_x,
            pi_s=pi_s,
            kernel_matrix=kernel_matrix,
            x_kernel=x_kernel,
            bias=args.user_shift,
            seed=args.seed,
        )

        def loss_fn(params, loss_calculator):
            return loss_calculator.compute_loss(params)

        grad_fn = value_and_grad(loss_fn)

        best_loss = float("inf")

        for epoch in range(args.epochs):
            print(f"Epoch {epoch}:")
            # Compute loss and gradients
            loss, grads = grad_fn(params, loss_calculator)

            # Update parameters
            updates, opt_state = tx.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if best_loss > loss:
                best_loss = loss
                best_stat = loss_calculator.tstat

        alpha_plus = stats.norm.ppf(1 - args.alpha / 2)
        alpha_min = stats.norm.ppf(args.alpha / 2)

        print(f"Best loss: {best_loss}")
        print(f"Best stat: {best_stat}")
        print()
        print(f"1-Alpha/2 quantile of N(0,1): {alpha_plus}")
        print(f"Alpha/2 quantile of N(0,1): {alpha_min}")

        # compute test statistic with best params
        if best_stat > alpha_plus or best_stat < alpha_min:
            print("Reject the null hypothesis.")
        else:
            print("Fail to reject the null hypothesis.")


if __name__ == "__main__":
    experiment()
