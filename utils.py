from jax import numpy as np, vmap
from jax import tree_flatten
from jax import lax
from jax import scipy



def p2v(tree):
    # completely flattens a tree
    return np.concatenate(
        [np.ravel(x) for x in tree_flatten(tree)[0]]
    )

def get_param_size(arbitrary_params):
    return np.size(p2v(arbitrary_params))

def tree_to_vector(tree):
    # the leaves in the tree look like (batch_size = 1, reg_dim, N_params_of_layer)
    # this function returns them as (reg_dim, batch_size * N_params_of_layer)
    return np.concatenate(
        [np.reshape(x, (x.shape[1], -1)) for x in tree_flatten(tree)[0]]
    , axis=1)

def apply_fn_wrapper(apply_fn, is_training):
    """
    Wraps apply_fn(variables, inputs) into apply_fn_bis(params, batch_stats, inputs).
    The is_training parameter is used to avoid errors:
    * If is_training=True, then the keyword mutable is set to True for the batch_stats
    * If is_training=False, then the keywork mutable is set to False.

    In either cases, only the output of the network will be returned.
    The updated batch_stats will be lost, and must be computed explicitely apart.
    """

    if is_training:
        def apply_fn2(params, batch_stats, inputs):
            # mutable, but the updated batch_stats is not used
            output, _ = apply_fn({"params": params, "batch_stats": batch_stats}, inputs, mutable=["batch_stats"])
            return output

        return apply_fn2

    else:
        def apply_fn2(params, batch_stats, inputs):
            # not mutable, no updated batch_stats
            output = apply_fn({"params": params, "batch_stats": batch_stats}, inputs)
            return output

        return apply_fn2


def apply_fn_wrapper_no_batch_stats(apply_fn):
    def apply_fn2(params, inputs):
        # not mutable, no updated batch_stats
        output = apply_fn({"params": params}, inputs)
        return output

    return apply_fn2


def falseaffine_correction0(jacobian, mean, x):
    # x is (batch_size, inputs...)
    batch_size = x.shape[0]
    # jacobian(x) @ mean is (batch_size * reg_dim, 1)
    return np.reshape(jacobian(x) @ mean, (batch_size, -1) )

falseaffine_correction = vmap(falseaffine_correction0, in_axes=(None, None, 0))