import trainer
import ntk
import test
import train_states
import models
import utils

import dataset_sines_infinite
import dataset_sines_finite

from jax import random
from jax import numpy as np
from flax.core import FrozenDict
import optax

from scipy.stats import ortho_group
import numpy as raw_np


def unlimtd_r_uni_modal_infinite(seed, pre_n_epochs, pre_n_tasks, pre_K, post_n_epochs, post_n_tasks, post_K, data_noise, maddox_noise, meta_lr, subspace_dimension):
    key = random.PRNGKey(seed)
    key_init, key = random.split(key)

    print("===============")
    print("This is UNLIMTD-R")
    print("For the uni-modal dataset: infinite sine dataset")
    print("This variant of UNLIMTD-R approaches the distribution with a single GP")
    print("===============")
    print("Creating model")
    model = models.small_network(40, "relu", 1)
    batch = random.uniform(key_init, shape=(5,1), minval=-5, maxval=5)
    init_vars = model.init(key_init, batch)
    apply_fn = utils.apply_fn_wrapper(model.apply, True)
    apply_fn_raw = model.apply

    print("Creating optimizers")
    step = trainer.step_lowdim_cov_singGP
    get_train_batch_fn = dataset_sines_infinite.get_training_batch

    optimizer_params = optax.adam(learning_rate = meta_lr)
    optimizer_mean = optax.adam(learning_rate = meta_lr)
    optimizer_scale = optax.adam(learning_rate = meta_lr)
    init_scale = np.ones( (subspace_dimension,) )
    mean_init = np.zeros( (utils.get_param_size(init_vars["params"]),) )

    N = utils.get_param_size(init_vars["params"])
    raw_np.random.seed(5)
    M = ortho_group.rvs(dim=N)
    P1 = M[:, :subspace_dimension].T
    print("Found projection matrix")

    post_state = train_states.TrainStateLowDimCovSingGP.create(apply_fn = apply_fn, apply_fn_raw=apply_fn_raw, params = init_vars["params"], mean=mean_init, scale=init_scale, tx_params = optimizer_params, tx_mean = optimizer_mean, tx_scale = optimizer_scale, batch_stats=FrozenDict(), proj = P1)

    def eval_during_post_training(key, state):
        current_params = state.params
        current_batch_stats = state.batch_stats
        current_mean = state.mean
        current_scale = state.scale
        kernel, kernel_self, jacobian = ntk.get_kernel_and_jac_lowdim_cov(apply_fn, current_params, current_scale, current_batch_stats, P1)

        subkey_1, subkey_2 = random.split(key)
        nlls = test.test_nll_one_kernel(subkey_1, kernel_self, jacobian, dataset_sines_infinite.get_test_batch, K=post_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses = test.test_error_one_kernel(subkey_2, kernel, kernel_self, jacobian, dataset_sines_infinite.get_test_batch, dataset_sines_infinite.error_fn, K=post_K, L=post_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        return np.mean(nlls), np.mean(mses)

    print("Starting training")
    key_post, key = random.split(key)
    post_state, post_losses, post_evals = trainer.train_and_eval(key_post, step, post_n_epochs, post_state, post_n_tasks, post_K, data_noise, maddox_noise, get_train_batch_fn, eval_during_post_training)
    print("Finished training")

    # Returning everything
    return init_vars, None, None, post_state, None, post_losses, post_evals



def unlimtd_r_uni_modal_finite(seed, pre_n_epochs, pre_n_tasks, pre_K, post_n_epochs, post_n_tasks, post_K, data_noise, maddox_noise, meta_lr, subspace_dimension):
    key = random.PRNGKey(seed)
    key_init, key = random.split(key)

    print("===============")
    print("This is UNLIMTD-R")
    print("For the uni-modal dataset: finite sine dataset (make sure that you have initialized dataset_sines_finite.py beforehand)")
    print("This variant of UNLIMTD-R approaches the distribution with a single GP")
    print("===============")
    print("Creating model")
    model = models.small_network(40, "relu", 1)
    batch = random.uniform(key_init, shape=(5,1), minval=-5, maxval=5)
    init_vars = model.init(key_init, batch)
    apply_fn = utils.apply_fn_wrapper(model.apply, True)
    apply_fn_raw = model.apply

    print("Creating optimizers")
    step = trainer.step_lowdim_cov_singGP
    get_train_batch_fn = dataset_sines_finite.get_training_batch

    optimizer_params = optax.adam(learning_rate = meta_lr)
    optimizer_mean = optax.adam(learning_rate = meta_lr)
    optimizer_scale = optax.adam(learning_rate = meta_lr)
    init_scale = np.ones( (subspace_dimension,) )
    mean_init = np.zeros( (utils.get_param_size(init_vars["params"]),) )

    N = utils.get_param_size(init_vars["params"])
    raw_np.random.seed(5)
    M = ortho_group.rvs(dim=N)
    P1 = M[:, :subspace_dimension].T
    print("Found projection matrix")

    post_state = train_states.TrainStateLowDimCovSingGP.create(apply_fn = apply_fn, apply_fn_raw=apply_fn_raw, params = init_vars["params"], mean=mean_init, scale=init_scale, tx_params = optimizer_params, tx_mean = optimizer_mean, tx_scale = optimizer_scale, batch_stats=FrozenDict(), proj = P1)

    def eval_during_post_training(key, state):
        current_params = state.params
        current_batch_stats = state.batch_stats
        current_mean = state.mean
        current_scale = state.scale
        kernel, kernel_self, jacobian = ntk.get_kernel_and_jac_lowdim_cov(apply_fn, current_params, current_scale, current_batch_stats, P1)

        subkey_1, subkey_2 = random.split(key)
        nlls = test.test_nll_one_kernel(subkey_1, kernel_self, jacobian, dataset_sines_infinite.get_test_batch, K=post_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses = test.test_error_one_kernel(subkey_2, kernel, kernel_self, jacobian, dataset_sines_infinite.get_test_batch, dataset_sines_infinite.error_fn, K=post_K, L=post_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        return np.mean(nlls), np.mean(mses)

    print("Starting training")
    key_post, key = random.split(key)
    post_state, post_losses, post_evals = trainer.train_and_eval(key_post, step, post_n_epochs, post_state, post_n_tasks, post_K, data_noise, maddox_noise, get_train_batch_fn, eval_during_post_training)
    print("Finished training")

    # Returning everything
    return init_vars, None, None, post_state, None, post_losses, post_evals