import jax.numpy as jnp
from jax import vmap

from tests.utils_test import (
    estimate_phi,
    gaussian_kernel,
    laplacian_kernel,
    polynomial_kernel,
)


def construct_kernel_matrix(x, kernel_type, kernel_param):
    if kernel_type == "gaussian":
        kernel_func = gaussian_kernel
    elif kernel_type == "poly":
        kernel_func = polynomial_kernel
    else:
        kernel_func = laplacian_kernel
    n = x.shape[0]
    m = n // 2

    # Create meshgrid for indices
    idx_i, idx_j = jnp.meshgrid(jnp.arange(m), jnp.arange(m, n))

    # Flatten and index into x to get the pairs
    x_i = x[idx_i.T.flatten()]
    x_j = x[idx_j.T.flatten()]

    # Pairwise kernel with the specified kernel function and parameter
    pairwise_kernel = vmap(lambda xi, xj: kernel_func(xi, xj, kernel_param))
    kernel_matrix = pairwise_kernel(x_i, x_j).reshape(m, n - m)

    return kernel_matrix


class KernelLossCalculator:
    def __init__(self, model, params, x, y, t, s, e_x, x_kernel, pi_s, kernel_matrix, bias, seed):
        self.model = model
        self.params = params
        self.x = x
        self.y = y
        self.t = t
        self.s = s
        self.e_x = e_x
        self.x_kernel = x_kernel
        self.pi_s = pi_s
        self.kernel_matrix = kernel_matrix
        self.bias = bias

        self.m1 = kernel_matrix.shape[0]  # Number of points in D_1
        self.m2 = kernel_matrix.shape[1]  # Number of points in D_2
        self.tstat = None
        self.M = None
        self.sampled_indices = None
        self.Ksampled = []
        self.nu_obs = estimate_phi(x, t, y, s, e_x=e_x, seed=seed)

    def compute_psi_vector(self):
        x_rct = self.x[self.s == 0]
        y_rct = self.y[self.s == 0]
        t_rct = self.t[self.s == 0]
        beta = self.model.apply(self.params, self.x_kernel[self.s == 0])
        pi = self.t[self.s == 0].mean()
        beta_x = beta.reshape(-1)
        nu_obs = self.nu_obs.predict(x_rct)

        psi_0 = y_rct * t_rct / pi - y_rct * (1 - t_rct) / (1 - pi)
        psi_1 = nu_obs - self.bias * (2 * beta_x - 1)
        return psi_0.reshape(-1) - psi_1

    def compute_loss(self, params):
        self.params = params
        psi_vector = self.compute_psi_vector()
        idx_i, idx_j = jnp.meshgrid(jnp.arange(self.m1), jnp.arange(self.m1, self.m1 + self.m2))
        psi_i = psi_vector[idx_i.T.flatten()]
        psi_j = psi_vector[idx_j.T.flatten()]
        kernel_values = self.kernel_matrix.flatten()
        hhh = psi_i * psi_j * kernel_values
        fhs = hhh.reshape(self.m1, self.m2).sum(axis=1) / self.m2
        U = fhs.mean()
        varU = fhs.var()
        loss = jnp.sqrt(self.m1) * U / jnp.sqrt(varU)
        self.tstat = (jnp.sqrt(self.m1) * U / jnp.sqrt(varU)).primal
        print("Computed loss and test statistic. Summary:")
        print(f"    Loss: {jnp.abs(loss).primal:.4f}")
        print(f"    Test statistic: {self.tstat:.4f}")
        print()
        return jnp.abs(loss)
