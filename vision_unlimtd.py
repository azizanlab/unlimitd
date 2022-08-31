import trainer
import ntk
import test
import train_states
import models
import utils
import fim

import dataset_shapenet1d

from jax import numpy as np
from jax import random
import optax

import numpy as raw_np
import scipy.linalg



def vision_unlimtd_identitycov(seed, pre_n_epochs, pre_n_tasks, pre_K, post_n_epochs, post_n_tasks, post_K, data_noise, maddox_noise, meta_lr, subspace_dimension):
    print("===============")
    print("This is UNLIMTD")
    print("For the uni-modal dataset: Shapenet1D")
    print("This variant of UNLIMTD approaches the distribution with a single GP")
    print("=> In UNLIMTD-F, this is the first part of training, with identity covariance (before computing the FIM)")
    print("=> In UNLIMTD-I, this is the full training")
    print("===============")

    print("Loading Shapenet1D dataset")
    dataset_shapenet1d.load_shapenet1d()
    
    key = random.PRNGKey(seed)
    key_init, key = random.split(key)

    print("Creating model")
    model = models.deep_network(2, True)
    batch = dataset_shapenet1d.x_train[0, 0:5]
    init_vars = model.init(key_init, batch)
    apply_fn = utils.apply_fn_wrapper(model.apply, True)
    apply_fn_raw = model.apply

    print("Creating optimizers")
    step = trainer.step_identity_cov
    get_train_batch_fn = dataset_shapenet1d.get_training_batch
    optimizer_params = optax.adam(learning_rate = meta_lr)
    optimizer_mean = optax.adam(learning_rate = meta_lr)
    mean_init = np.zeros( (utils.get_param_size(init_vars["params"]),))

    pre_state = train_states.TrainStateIdentityCovariance.create(apply_fn=apply_fn, apply_fn_raw=apply_fn_raw, params=init_vars["params"],mean=mean_init, tx_params=optimizer_params, tx_mean=optimizer_mean, batch_stats=init_vars["batch_stats"])

    def eval_during_pre_training(key, state):
        current_params = state.params
        current_batch_stats = state.batch_stats
        current_mean = state.mean
        kernel, kernel_self, jac = ntk.get_kernel_and_jac_identity_cov(apply_fn, current_params, current_batch_stats)

        subkey_1, subkey_2, subkey_3, subkey_4, subkey_5, subkey_6 = random.split(key, 6)
        nlls_train = test.test_nll_one_kernel(subkey_1, kernel_self, jac, dataset_shapenet1d.get_train_batch_as_val_batch, K=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses_train = test.test_error_one_kernel(subkey_2, kernel, kernel_self, jac, dataset_shapenet1d.get_train_batch_as_val_batch , dataset_shapenet1d.error_fn, K=15, L=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        nlls_val = test.test_nll_one_kernel(subkey_3, kernel_self, jac, dataset_shapenet1d.get_val_batch, K=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses_val = test.test_error_one_kernel(subkey_4, kernel, kernel_self, jac, dataset_shapenet1d.get_val_batch , dataset_shapenet1d.error_fn, K=15, L=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        nlls_test = test.test_nll_one_kernel(subkey_5, kernel_self, jac, dataset_shapenet1d.get_test_batch, K=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses_test = test.test_error_one_kernel(subkey_6, kernel, kernel_self, jac, dataset_shapenet1d.get_test_batch , dataset_shapenet1d.error_fn, K=15, L=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        return np.mean(nlls_train), np.mean(mses_train), np.mean(nlls_val), np.mean(mses_val), np.mean(nlls_test), np.mean(mses_test)

    print("Starting first part of training (identity covariance)")
    key_pre, key = random.split(key)
    pre_state, pre_losses, pre_evals = trainer.train_and_eval(key_pre, step, pre_n_epochs, pre_state, pre_n_tasks, pre_K, data_noise, maddox_noise, get_train_batch_fn, eval_during_pre_training)
    print("Finished first part of training")

    return init_vars, pre_state, pre_evals, None, pre_losses, None, None

def vision_unlimtd_find_proj(seed, intermediate_params, intermediate_batch_stats, subspace_dimension):
    print("===============")
    print("This is UNLIMTD")
    print("For the uni-modal dataset: Shapenet1D")
    print("This variant of UNLIMTD-F approaches the distribution with a single GP")
    print("=> In UNLIMTD-F: this finds the projection matrix (FIM sketching)")
    print("===============")

    print("Loading Shapenet1D dataset")
    dataset_shapenet1d.load_shapenet1d()
    
    key = random.PRNGKey(seed)
    key_init, key = random.split(key)

    print("Creating model")
    
    model_bis = models.deep_network(2, False)
    apply_fn_bis = utils.apply_fn_wrapper(model_bis.apply, False)

    print("Finding projection matrix")
    key_fim, key = random.split(key)
    P1 = fim.proj_sketch(key=key_fim, apply_fn=apply_fn_bis, current_params=intermediate_params, batch_stats=intermediate_batch_stats, batches=dataset_shapenet1d.x_train, subspace_dimension=subspace_dimension)
    print("Found projection matrix")

    return P1

def vision_unlimtd_lowdim_cov(seed, intermediate_params, intermediate_batch_stats, post_n_epochs, post_n_tasks, post_K, data_noise, maddox_noise, meta_lr, subspace_dimension, P1, intermediate_mean):
    print("===============")
    print("This is UNLIMTD")
    print("For the uni-modal dataset: Shapenet1D")
    print("This variant of UNLIMTD approaches the distribution with a single GP")
    print("=> In UNLIMTD-F, this is the last part of training, with a low-dimensional matrix (after computing the FIM)")
    print("===============")

    print("Loading Shapenet1D dataset")
    dataset_shapenet1d.load_shapenet1d()
    get_train_batch_fn = dataset_shapenet1d.get_training_batch
    
    key = random.PRNGKey(seed)
    key_init, key = random.split(key)

    print("Creating model")
    model = models.deep_network(2, True)
    batch = dataset_shapenet1d.x_train[0, 0:5]
    init_vars = model.init(key_init, batch)
    apply_fn = utils.apply_fn_wrapper(model.apply, True)
    apply_fn_raw = model.apply

    # Usual training with projection
    print("Creating optimizers")
    step = trainer.step_lowdim_cov_singGP
    optimizer_params = optax.adam(learning_rate = meta_lr)
    optimizer_scale = optax.adam(learning_rate = meta_lr)
    optimizer_mean = optax.adam(learning_rate = meta_lr)
    init_scale = np.ones( (subspace_dimension,) )

    post_state = train_states.TrainStateLowDimCovSingGP.create(apply_fn = apply_fn, apply_fn_raw=apply_fn_raw, params = intermediate_params, mean = intermediate_mean, scale=init_scale, tx_params = optimizer_params, tx_mean = optimizer_mean, tx_scale = optimizer_scale, batch_stats=intermediate_batch_stats, proj = P1)

    def eval_during_post_training(key, state):
        current_params = state.params
        current_batch_stats = state.batch_stats
        current_scale = state.scale
        current_mean = state.mean
        proj = state.proj
        kernel, kernel_self, jac = ntk.get_kernel_and_jac_lowdim_cov(apply_fn, current_params, current_scale, current_batch_stats, proj)

        subkey_1, subkey_2, subkey_3, subkey_4, subkey_5, subkey_6 = random.split(key, 6)
        nlls_train = test.test_nll_one_kernel(subkey_1, kernel_self, jac, dataset_shapenet1d.get_train_batch_as_val_batch, K=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses_train = test.test_error_one_kernel(subkey_2, kernel, kernel_self, jac, dataset_shapenet1d.get_train_batch_as_val_batch , dataset_shapenet1d.error_fn, K=15, L=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        nlls_val = test.test_nll_one_kernel(subkey_3, kernel_self, jac, dataset_shapenet1d.get_val_batch, K=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses_val = test.test_error_one_kernel(subkey_4, kernel, kernel_self, jac, dataset_shapenet1d.get_val_batch , dataset_shapenet1d.error_fn, K=15, L=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        nlls_test = test.test_nll_one_kernel(subkey_5, kernel_self, jac, dataset_shapenet1d.get_test_batch, K=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses_test = test.test_error_one_kernel(subkey_6, kernel, kernel_self, jac, dataset_shapenet1d.get_test_batch , dataset_shapenet1d.error_fn, K=15, L=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        return np.mean(nlls_train), np.mean(mses_train), np.mean(nlls_val), np.mean(mses_val), np.mean(nlls_test), np.mean(mses_test)

    print("Starting training")
    key_post, key = random.split(key)
    post_state, post_losses, post_evals = trainer.train_and_eval(key_post, step, post_n_epochs, post_state, post_n_tasks, post_K, data_noise, maddox_noise, get_train_batch_fn, eval_during_post_training)
    print("Finished training")

    # Returning everything
    return None, None, post_state, None, post_losses, post_evals



def vision_unlimtd_r(seed, proj_seed, n_epochs, n_tasks, K, data_noise, maddox_noise, meta_lr, subspace_dimension):
    print("===============")
    print("This is UNLIMTD")
    print("For the uni-modal dataset: Shapenet1D")
    print("This variant of UNLIMTD approaches the distribution with a single GP")
    print("=> In UNLIMTD-R, this is the full training")
    print("===============")

    print("Loading Shapenet1D dataset")
    dataset_shapenet1d.load_shapenet1d()
    get_train_batch_fn = dataset_shapenet1d.get_training_batch
    
    key = random.PRNGKey(seed)
    key_init, key = random.split(key)

    print("Creating model")
    model = models.deep_network(2, True)
    batch = dataset_shapenet1d.x_train[0, 0:5]
    init_vars = model.init(key_init, batch)
    apply_fn = utils.apply_fn_wrapper(model.apply, True)
    apply_fn_raw = model.apply

    # Post-training
    print("Creating optimizers")
    step = trainer.step_lowdim_cov_singGP
    optimizer_params = optax.adam(learning_rate = meta_lr)
    optimizer_scale = optax.adam(learning_rate = meta_lr)
    optimizer_mean = optax.adam(learning_rate = meta_lr)

    init_scale = np.ones( (subspace_dimension,) )
    mean_init = np.zeros( (utils.get_param_size(init_vars["params"]),))

    N = utils.get_param_size(init_vars["params"])
    raw_np.random.seed(proj_seed)
    a = raw_np.random.normal(loc=0.0, scale=1.0, size=(N, subspace_dimension))
    # q is of shape (N, subspace_dimension)
    q = scipy.linalg.orth(a)
    P1 = q.T
    print("Found projection matrix")

    post_state = train_states.TrainStateLowDimCovSingGP.create(apply_fn = apply_fn, apply_fn_raw=apply_fn_raw, params = init_vars["params"], mean = mean_init, scale=init_scale, tx_params = optimizer_params, tx_mean = optimizer_mean, tx_scale = optimizer_scale, batch_stats=init_vars["batch_stats"], proj = P1)

    def eval_during_post_training(key, state):
        current_params = state.params
        current_batch_stats = state.batch_stats
        current_scale = state.scale
        current_mean = state.mean
        proj = state.proj
        kernel, kernel_self, jac = ntk.get_kernel_and_jac_lowdim_cov(apply_fn, current_params, current_scale, current_batch_stats, proj)

        subkey_1, subkey_2, subkey_3, subkey_4, subkey_5, subkey_6 = random.split(key, 6)
        nlls_train = test.test_nll_one_kernel(subkey_1, kernel_self, jac, dataset_shapenet1d.get_train_batch_as_val_batch, K=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses_train = test.test_error_one_kernel(subkey_2, kernel, kernel_self, jac, dataset_shapenet1d.get_train_batch_as_val_batch , dataset_shapenet1d.error_fn, K=15, L=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        nlls_val = test.test_nll_one_kernel(subkey_3, kernel_self, jac, dataset_shapenet1d.get_val_batch, K=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses_val = test.test_error_one_kernel(subkey_4, kernel, kernel_self, jac, dataset_shapenet1d.get_val_batch , dataset_shapenet1d.error_fn, K=15, L=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        nlls_test = test.test_nll_one_kernel(subkey_5, kernel_self, jac, dataset_shapenet1d.get_test_batch, K=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses_test = test.test_error_one_kernel(subkey_6, kernel, kernel_self, jac, dataset_shapenet1d.get_test_batch , dataset_shapenet1d.error_fn, K=15, L=15, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        return np.mean(nlls_train), np.mean(mses_train), np.mean(nlls_val), np.mean(mses_val), np.mean(nlls_test), np.mean(mses_test)

    print("Starting training")
    key_post, key = random.split(key)
    post_state, post_losses, post_evals = trainer.train_and_eval(key_post, step, n_epochs, post_state, n_tasks, K, data_noise, maddox_noise, get_train_batch_fn, eval_during_post_training)
    print("Finished training")

    # Returning everything
    # return init_vars, pre_state, post_state, pre_losses, post_losses
    return init_vars["params"], None, None, post_state, None, post_losses, post_evals