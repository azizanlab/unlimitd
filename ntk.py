import utils
from functools import partial
from jax import jacrev
from jax import vmap
from jax import numpy as np

def get_kernel_and_jac_identity_cov(apply_fn, current_params, current_batch_stats):
    """
    Returns the kernel in case of an identity covariance matrix (JJ^\top)
    Also returns the jacobian J
    """
    jacobian = get_jacobian(apply_fn, current_params, current_batch_stats)

    def kernel(x1, x2):
        return jacobian(x1) @ jacobian(x2).T

    def kernel_self(x1):
        j1 = jacobian(x1)
        return j1 @ j1.T

    return kernel, kernel_self, jacobian

def get_kernel_and_jac_lowdim_cov(apply_fn, current_params, current_scale, current_batch_stats, proj):
    """
    Returns the kernel in case of a low-dimensional covariance matrix (J P^\top s P J^\top)
    Also returns the jacobian J
    """

    jacobian = get_jacobian(apply_fn, current_params, current_batch_stats)

    def kernel(x1, x2):
        return np.linalg.multi_dot([jacobian(x1), proj.T, np.diag(current_scale)**2, proj, jacobian(x2).T])

    def kernel_self(x1):
        j1 = jacobian(x1)
        A = j1 @ proj.T
        return np.linalg.multi_dot([A, np.diag(current_scale)**2, A.T])

    return kernel, kernel_self, jacobian

def get_jacobian(apply_fn, current_params, current_batch_stats):
    """
    Returns the jacobian of the network
    """
    def jacobian0(x):
        # x has lost its batch_size dimension (after vmap)
        # * for vision problems, its shape is (128, 128, 1)
        # * for toy problems, its shape is (1,)
        # !! For this function to work, the network should always return a batch_size, even if the input did not have one
        # ========================================
        # out is a tree, where the leaves look like (batch_size = 1, reg_dim, N_params_of_layer)
        # we turn it into a single array of shape (reg_dim, N_total_params)

        # apply_fn must already be wrapped
        out = partial(jacrev(apply_fn, argnums=0), current_params, current_batch_stats)(x)
        return utils.tree_to_vector(out)
    
    jacobian = vmap(jacobian0)

    def final_jacobian(x):
        # here, x is (batch_size, input_dims...)
        # j is of shape (batch_size, reg_dim, N_total_params)
        # we return an array of shape (batch_size * reg_dim, N_total_params)
        j = jacobian(x)
        return np.reshape(j, (-1, j.shape[-1]) )

    return final_jacobian