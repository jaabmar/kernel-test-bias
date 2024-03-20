import jax
import jax.numpy as jnp
from sklearn.ensemble import RandomForestRegressor


@jax.jit
def laplacian_kernel(x, y, gamma=1.0):
    return jnp.exp(-gamma * jnp.sum(jnp.abs(x - y)))


@jax.jit
def gaussian_kernel(x, y, sigma=1):
    return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * sigma**2))


@jax.jit
def polynomial_kernel(x, y, degree=3.0, coefficient=1.0, constant=0.0):
    return (jnp.dot(x, y) * coefficient + constant) ** degree


def estimate_phi(x, t, y, s, e_x, seed=50):
    x_obs = x[s == 1]
    t_obs = t[s == 1].ravel()
    y_obs = y[s == 1].ravel()
    mu1 = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=0.01,
        n_jobs=-2,
        random_state=seed,
    )

    mu0 = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=0.01,
        n_jobs=-2,
        random_state=seed,
    )

    mu1.fit(x_obs[(t_obs == 1).reshape(-1), :], y_obs[t_obs == 1])
    mu0.fit(x_obs[(t_obs == 0).reshape(-1), :], y_obs[t_obs == 0])

    phi1 = mu1.predict(x_obs)
    phi0 = mu0.predict(x_obs)

    phi_obs = (
        phi1
        + t_obs.reshape(-1) * (y_obs.reshape(-1) - phi1) / e_x[s == 1]
        - phi0
        - (1 - t_obs).reshape(-1) * (y_obs.reshape(-1) - phi0) / (1 - e_x[s == 1])
    )
    nu_obs = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=0.01,
        n_jobs=-2,
        random_state=seed,
    )
    nu_obs.fit(x_obs, phi_obs)

    return nu_obs
