import numpy as np
import scipy.stats

from tests.utils_test import estimate_phi


def ate_test(x, y, t, s, e_x, bias, seed=50, bootstraps=1000):
    np.random.seed(seed)
    # Filtering RCT data
    x_rct, y_rct, t_rct = x[s == 0], y[s == 0], t[s == 0]
    pi = t_rct.mean()
    nu_obs_est = estimate_phi(x, t, y, s, e_x=e_x, seed=seed)
    psi_0 = y_rct * t_rct / pi - y_rct * (1 - t_rct) / (1 - pi)
    nu_obs = nu_obs_est.predict(x_rct)
    psi_1_minus = nu_obs - bias
    psi_1_plus = nu_obs + bias
    print("Nuisance functions estimated.")
    # Bootstrap to estimate standard errors
    std_rct = scipy.stats.bootstrap(
        (psi_0,), np.mean, n_resamples=bootstraps, random_state=seed
    ).standard_error
    std_os_minus = scipy.stats.bootstrap(
        (psi_1_minus,), np.mean, n_resamples=bootstraps, random_state=seed
    ).standard_error
    std_os_plus = scipy.stats.bootstrap(
        (psi_1_plus,), np.mean, n_resamples=bootstraps, random_state=seed
    ).standard_error
    std_plus_total, std_minus_total = np.sqrt(std_os_plus**2 + std_rct**2), np.sqrt(
        std_os_minus**2 + std_rct**2
    )
    print("Bootstrap estimates for standard errors calculated.")
    # Calculating test statistics
    test_plus = (psi_1_plus.mean() - psi_0.mean()) / std_plus_total
    test_minus = (psi_0.mean() - psi_1_minus.mean()) / std_minus_total
    return min(test_plus, test_minus)
