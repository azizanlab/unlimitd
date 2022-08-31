from jax import numpy as np
from ntk import get_kernel_and_jac_identity_cov
from ntk import get_kernel_and_jac_lowdim_cov
from jax import lax
from jax import scipy
import utils


def nll_batch_average_identity_cov(current_params, current_mean, apply_fn, current_batch_stats, x_a, y_a, maddox_noise):
    _, kernel_self, jacobian = get_kernel_and_jac_identity_cov(apply_fn, current_params, current_batch_stats)

    return np.mean(nll_batch_one_kernel(kernel_self, x_a, y_a, maddox_noise, jacobian, current_mean))

def nll_batch_average_lowdim_cov_singGP(current_params, current_mean, current_scale, apply_fn, current_batch_stats, proj, x_a, y_a, maddox_noise):
    _, kernel_self, jacobian = get_kernel_and_jac_lowdim_cov(apply_fn, current_params, current_scale, current_batch_stats, proj)

    return np.mean(nll_batch_one_kernel(kernel_self, x_a, y_a, maddox_noise, jacobian, current_mean))

def nll_batch_average_lowdim_cov_mixture(current_params, current_mean1, current_mean2, current_scale1, current_scale2, apply_fn, current_batch_stats, proj1, proj2, x_a, y_a, maddox_noise):
    _, kernel_self1, jacobian1 = get_kernel_and_jac_lowdim_cov(apply_fn, current_params, current_scale1, current_batch_stats, proj1)
    _, kernel_self2, jacobian2 = get_kernel_and_jac_lowdim_cov(apply_fn, current_params, current_scale2, current_batch_stats, proj2)

    return np.mean(nll_batch_two_kernels(kernel_self1, kernel_self2, x_a, y_a, maddox_noise, jacobian1, jacobian2, current_mean1, current_mean2))


def nll_batch_one_kernel(kernel_self, x_a, y_a, maddox_noise, jacobian, mean):
    """
    NLL for a batch of tasks, when there is only one kernel (singGP)
    x_a is (n_tasks, batch_size, input_dims) (input_dims are (128, 128, 1) for vision, (1,) for toy problems)
    y_a is (n_tasks, batch_size, reg_dim)
    """
    def f(carry, task_data):
        x_a, y_a = task_data
        y_a = y_a - utils.falseaffine_correction0(jacobian, mean, x_a)
        loss_here = nll(kernel_self, x_a, y_a, maddox_noise)
        return None, loss_here

    _, losses = lax.scan(f, None, (x_a, y_a))

    return np.array(losses)

def nll_batch_two_kernels(kernel_self1, kernel_self2, x_a, y_a, maddox_noise, jacobian1, jacobian2, mean1, mean2):
    """
    NLL for a batch of tasks, when there are two kernels (mixture of GPs)
    x_a is (n_tasks, batch_size, input_dims) (input_dims are (128, 128, 1) for vision, (1,) for toy problems)
    y_a is (n_tasks, batch_size, reg_dim)
    """
    def f(carry, task_data):
        x_a, y_a = task_data
        y_a1 = y_a - utils.falseaffine_correction0(jacobian1, mean1, x_a)
        y_a2 = y_a - utils.falseaffine_correction0(jacobian2, mean2, x_a)
        loss_1 = nll(kernel_self1, x_a, y_a1, maddox_noise)
        loss_2 = nll(kernel_self2, x_a, y_a2, maddox_noise)
        return None, nll_logaddexp(loss_1, loss_2)

    _, losses = lax.scan(f, None, (x_a, y_a))

    return np.array(losses)

def nll(kernel_self, x_a, y_a, maddox_noise):
    """
    Computes the NLL of this data (one task only) wrt the kernel
    x_a is a (batch_size, input_dims) array (! has lost n_tasks)
    y_a is a (batch_size, reg_dim) array (! has lost n_tasks)
    """
    cov_a_a = kernel_self(x_a)
    K = cov_a_a.shape[0]
    cov_a_a = cov_a_a + maddox_noise ** 2 * np.eye(K)
    
    # prior mean is 0
    y_a = np.reshape(y_a, (-1))

    L = scipy.linalg.cho_factor(cov_a_a)
    alpha = scipy.linalg.cho_solve(L, y_a)
    
    return 0.5 * y_a.T @ alpha + np.sum(np.log(np.diag(L[0]))) + 0.5 * K * np.log(2 * np.pi)

def gaussian_posterior(kernel, kernel_self, x_a, y_a, x_b, maddox_noise):
    """
    Computes the gaussian posterior with this kernel and this data, on the queried inputs.
    x_a is a (batch_size, input_dims) array (! has lost n_tasks)
    y_a is a (batch_size, reg_dim) array (! has lost n_tasks)
    Does not return the posterior covariance matrix
    """
    dim = y_a.shape[1]
    y_a = np.reshape(y_a, (-1,))

    cov_a_a = kernel_self(x_a)
    cov_a_a = cov_a_a + maddox_noise ** 2 * np.eye(cov_a_a.shape[0])
    cov_b_a = kernel(x_b, x_a)

    L = scipy.linalg.cho_factor(cov_a_a)
    alpha = scipy.linalg.cho_solve(L, y_a)
    post_mean = cov_b_a @ alpha

    return np.reshape(post_mean, (-1, dim) )

def gaussian_posterior_full(kernel, kernel_self, x_a, y_a, x_b, maddox_noise):
    """
    Computes the gaussian posterior with this kernel and this data, on the queried inputs.
    x_a is a (batch_size, input_dims) array (! has lost n_tasks)
    y_a is a (batch_size, reg_dim) array (! has lost n_tasks)
    Returns the posterior covariance matrix
    """
    dim = y_a.shape[1]
    y_a = np.reshape(y_a, (-1,))

    cov_a_a = kernel_self(x_a)
    cov_a_a = cov_a_a + maddox_noise ** 2 * np.eye(cov_a_a.shape[0])
    cov_b_a = kernel(x_b, x_a)
    cov_b_b = kernel_self(x_b)
    
    L = scipy.linalg.cho_factor(cov_a_a)
    alpha = scipy.linalg.cho_solve(L, y_a)
    post_mean = cov_b_a @ alpha
    
    v = scipy.linalg.cho_solve(L, cov_b_a.T)
    post_cov = cov_b_b - cov_b_a @ v
    
    return np.reshape(post_mean, (-1, dim) ), post_cov

def nll_logaddexp(nll1, nll2):
    return np.log(2) - np.logaddexp(-nll1, -nll2)