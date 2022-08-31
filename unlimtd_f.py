import trainer
import ntk
import test
import train_states
import models
import utils
import fim

import dataset_sines_infinite
import dataset_sines_finite
import dataset_multi_infinite

from jax import random
from jax import numpy as np
from flax.core import FrozenDict
import optax

def unlimtd_f_uni_modal_infinite(seed, pre_n_epochs, pre_n_tasks, pre_K, post_n_epochs, post_n_tasks, post_K, data_noise, maddox_noise, meta_lr, subspace_dimension):
    key = random.PRNGKey(seed)
    key_init, key = random.split(key)
    
    print("===============")
    print("This is UNLIMTD-F")
    print("For the uni-modal dataset: infinite sine dataset")
    print("This variant of UNLIMTD-F approaches the distribution with a single GP")
    print("===============")
    print("Creating model")
    model = models.small_network(40, "relu", 1)
    batch = random.uniform(key_init, shape=(5,1), minval=-5, maxval=5)
    init_vars = model.init(key_init, batch)
    apply_fn = utils.apply_fn_wrapper(model.apply, True)
    apply_fn_raw = model.apply

    # Training before finding the FIM matrix
    print("Creating optimizers")
    step = trainer.step_identity_cov
    get_train_batch_fn = dataset_sines_infinite.get_training_batch
    optimizer_params = optax.adam(learning_rate = meta_lr)
    optimizer_mean = optax.adam(learning_rate = meta_lr)
    mean_init = np.zeros( (utils.get_param_size(init_vars["params"]),) )

    pre_state = train_states.TrainStateIdentityCovariance.create(apply_fn=apply_fn, apply_fn_raw=apply_fn_raw, params=init_vars["params"], mean=mean_init, tx_params=optimizer_params, tx_mean=optimizer_mean, batch_stats=FrozenDict())

    def eval_during_pre_training(key, state):
        current_params = state.params
        current_mean = state.mean
        current_batch_stats = state.batch_stats
        kernel, kernel_self, jacobian = ntk.get_kernel_and_jac_identity_cov(apply_fn, current_params, current_batch_stats)

        subkey_1, subkey_2 = random.split(key)
        nlls = test.test_nll_one_kernel(subkey_1, kernel_self, jacobian, dataset_sines_infinite.get_test_batch, K=pre_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses = test.test_error_one_kernel(subkey_2, kernel, kernel_self, jacobian, dataset_sines_infinite.get_test_batch, dataset_sines_infinite.error_fn, K=pre_K, L=pre_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        return np.mean(nlls), np.mean(mses)

    print("Starting first part of training (identity covariance)")
    key_pre, key = random.split(key)
    pre_state, pre_losses, pre_evals = trainer.train_and_eval(key_pre, step, pre_n_epochs, pre_state, pre_n_tasks, pre_K, data_noise, maddox_noise, get_train_batch_fn, eval_during_pre_training)
    print("Finished first part of training")

    # FIM
    print("Finding projection matrix")
    key_fim, key_data, key = random.split(key, 3)
    # here we use the exact FIM, we do not need to approximate given the (small) size of the network
    # P1 = fim.proj_exact(key=key_fim, apply_fn=apply_fn, current_params=pre_state.params, current_batch_stats=pre_state.batch_stats, subspace_dimension=subspace_dimension)
    P1 = fim.proj_sketch(key=key_fim, apply_fn=apply_fn, current_params=pre_state.params, batch_stats=pre_state.batch_stats, batches=random.uniform(key_data, shape=(100, 1761, 1), minval=-5, maxval=5), subspace_dimension=subspace_dimension)
    print("Found projection matrix")

    # Usual training with projection
    print("Creating optimizers")
    step = trainer.step_lowdim_cov_singGP
    optimizer_params = optax.adam(learning_rate = meta_lr)
    optimizer_mean = optax.adam(learning_rate = meta_lr)
    optimizer_scale = optax.adam(learning_rate = meta_lr)
    init_scale = np.ones( (subspace_dimension,) )

    post_state = train_states.TrainStateLowDimCovSingGP.create(apply_fn = apply_fn, apply_fn_raw=apply_fn_raw, params = pre_state.params, mean=pre_state.mean, scale=init_scale, tx_params = optimizer_params, tx_mean = optimizer_mean, tx_scale = optimizer_scale, batch_stats=pre_state.batch_stats, proj = P1)

    def eval_during_post_training(key, state):
        current_params = state.params
        current_batch_stats = state.batch_stats
        current_mean = state.mean
        current_scale = state.scale
        kernel, kernel_self, jacobian = ntk.get_kernel_and_jac_lowdim_cov(apply_fn, current_params, current_scale, current_batch_stats, P1)

        subkey_1, subkey_2 = random.split(key)
        nlls = test.test_nll_one_kernel(subkey_1, kernel_self, jacobian, dataset_sines_infinite.get_test_batch, K=pre_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses = test.test_error_one_kernel(subkey_2, kernel, kernel_self, jacobian, dataset_sines_infinite.get_test_batch, dataset_sines_infinite.error_fn, K=pre_K, L=pre_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        return np.mean(nlls), np.mean(mses)

    print("Starting training")
    key_post, key = random.split(key)
    post_state, post_losses, post_evals = trainer.train_and_eval(key_post, step, post_n_epochs, post_state, post_n_tasks, post_K, data_noise, maddox_noise, get_train_batch_fn, eval_during_post_training)
    print("Finished training")

    # Returning everything
    return init_vars, pre_state, pre_evals, post_state, pre_losses, post_losses, post_evals


def unlimtd_f_uni_modal_finite(seed, pre_n_epochs, pre_n_tasks, pre_K, post_n_epochs, post_n_tasks, post_K, data_noise, maddox_noise, meta_lr, subspace_dimension):
    key = random.PRNGKey(seed)
    key_init, key = random.split(key)

    print("===============")
    print("This is UNLIMTD-F")
    print("For the uni-modal dataset: finite sine dataset (make sure that you have initialized dataset_sines_finite.py beforehand)")
    print("This variant of UNLIMTD-F approaches the distribution with a single GP")
    print("===============")
    print("Creating model")
    model = models.small_network(40, "relu", 1)
    batch = random.uniform(key_init, shape=(5,1), minval=-5, maxval=5)
    init_vars = model.init(key_init, batch)
    apply_fn = utils.apply_fn_wrapper(model.apply, True)
    apply_fn_raw = model.apply

    # Training before finding the FIM matrix
    print("Creating optimizers")
    step = trainer.step_identity_cov
    get_train_batch_fn = dataset_sines_finite.get_training_batch
    optimizer_params = optax.adam(learning_rate = meta_lr)
    optimizer_mean = optax.adam(learning_rate = meta_lr)
    mean_init = np.zeros( (utils.get_param_size(init_vars["params"]),) )

    pre_state = train_states.TrainStateIdentityCovariance.create(apply_fn=apply_fn, apply_fn_raw=apply_fn_raw, params=init_vars["params"], mean=mean_init, tx_params=optimizer_params, tx_mean=optimizer_mean, batch_stats=FrozenDict())

    def eval_during_pre_training(key, state):
        current_params = state.params
        current_mean = state.mean
        current_batch_stats = state.batch_stats
        kernel, kernel_self, jacobian = ntk.get_kernel_and_jac_identity_cov(apply_fn, current_params, current_batch_stats)

        # test on any sine
        subkey_1, subkey_2 = random.split(key)
        nlls = test.test_nll_one_kernel(subkey_1, kernel_self, jacobian, dataset_sines_infinite.get_test_batch, K=pre_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses = test.test_error_one_kernel(subkey_2, kernel, kernel_self, jacobian, dataset_sines_infinite.get_test_batch, dataset_sines_infinite.error_fn, K=pre_K, L=pre_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        return np.mean(nlls), np.mean(mses)

    print("Starting first part of training (identity covariance)")
    key_pre, key = random.split(key)
    pre_state, pre_losses, pre_evals = trainer.train_and_eval(key_pre, step, pre_n_epochs, pre_state, pre_n_tasks, pre_K, data_noise, maddox_noise, get_train_batch_fn, eval_during_pre_training)
    print("Finished first part of training")

    # FIM
    print("Finding projection matrix")

    key_fim, key_data, key = random.split(key, 3)
    # here we use the exact FIM, we do not need to approximate given the (small) size of the network
    # P1 = fim.proj_exact(key=key_fim, apply_fn=apply_fn, current_params=pre_state.params, current_batch_stats=pre_state.batch_stats, subspace_dimension=subspace_dimension)
    P1 = fim.proj_sketch(key=key_fim, apply_fn=apply_fn, current_params=pre_state.params, batch_stats=pre_state.batch_stats, batches=dataset_sines_finite.x_train, subspace_dimension=subspace_dimension)
    print("Found projection matrix")

    # Usual training with projection
    print("Creating optimizers")
    step = trainer.step_lowdim_cov_singGP
    optimizer_params = optax.adam(learning_rate = meta_lr)
    optimizer_mean = optax.adam(learning_rate = meta_lr)
    optimizer_scale = optax.adam(learning_rate = meta_lr)
    init_scale = np.ones( (subspace_dimension,) )

    post_state = train_states.TrainStateLowDimCovSingGP.create(apply_fn = apply_fn, apply_fn_raw=apply_fn_raw, params = pre_state.params, mean=pre_state.mean, scale=init_scale, tx_params = optimizer_params, tx_mean = optimizer_mean, tx_scale = optimizer_scale, batch_stats=pre_state.batch_stats, proj = P1)

    def eval_during_post_training(key, state):
        current_params = state.params
        current_batch_stats = state.batch_stats
        current_mean = state.mean
        current_scale = state.scale
        kernel, kernel_self, jacobian = ntk.get_kernel_and_jac_lowdim_cov(apply_fn, current_params, current_scale, current_batch_stats, P1)

        # test on any sine
        subkey_1, subkey_2 = random.split(key)
        nlls = test.test_nll_one_kernel(subkey_1, kernel_self, jacobian, dataset_sines_infinite.get_test_batch, K=pre_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses = test.test_error_one_kernel(subkey_2, kernel, kernel_self, jacobian, dataset_sines_infinite.get_test_batch, dataset_sines_infinite.error_fn, K=pre_K, L=pre_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        return np.mean(nlls), np.mean(mses)

    print("Starting training")
    key_post, key = random.split(key)
    post_state, post_losses, post_evals = trainer.train_and_eval(key_post, step, post_n_epochs, post_state, post_n_tasks, post_K, data_noise, maddox_noise, get_train_batch_fn, eval_during_post_training)
    print("Finished training")

    # Returning everything
    return init_vars, pre_state, pre_evals, post_state, pre_losses, post_losses, post_evals


def unlimtd_f_multi_modal_singGP(seed, pre_n_epochs, pre_n_tasks, pre_K, post_n_epochs, post_n_tasks, post_K, data_noise, maddox_noise, meta_lr, subspace_dimension):
    key = random.PRNGKey(seed)
    key_init, key = random.split(key)

    print("===============")
    print("This is UNLIMTD-F")
    print("For the multi-modal dataset: sine + line (both infinite)")
    print("This variant of UNLIMTD-F approaches the distribution with a single GP")
    print("===============")
    print("Creating model")
    model = models.small_network(40, "relu", 1)
    batch = random.uniform(key_init, shape=(5,1), minval=-5, maxval=5)
    init_vars = model.init(key_init, batch)
    apply_fn = utils.apply_fn_wrapper(model.apply, True)
    apply_fn_raw = model.apply

    # Training before finding the FIM matrix
    print("Creating optimizers")
    step = trainer.step_identity_cov
    get_train_batch_fn = dataset_multi_infinite.get_training_batch
    optimizer_params = optax.adam(learning_rate = meta_lr)
    optimizer_mean = optax.adam(learning_rate = meta_lr)
    mean_init = np.zeros( (utils.get_param_size(init_vars["params"]),) )

    pre_state = train_states.TrainStateIdentityCovariance.create(apply_fn=apply_fn, apply_fn_raw=apply_fn_raw, params=init_vars["params"], mean=mean_init, tx_params=optimizer_params, tx_mean=optimizer_mean, batch_stats=FrozenDict())

    def eval_during_pre_training(key, state):
        current_params = state.params
        current_mean = state.mean
        current_batch_stats = state.batch_stats
        kernel, kernel_self, jacobian = ntk.get_kernel_and_jac_identity_cov(apply_fn, current_params, current_batch_stats)

        subkey_1, subkey_2 = random.split(key)
        nlls = test.test_nll_one_kernel(subkey_1, kernel_self, jacobian, dataset_multi_infinite.get_test_batch, K=pre_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses = test.test_error_one_kernel(subkey_2, kernel, kernel_self, jacobian, dataset_multi_infinite.get_test_batch, dataset_multi_infinite.error_fn, K=pre_K, L=pre_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        return np.mean(nlls), np.mean(mses)

    print("Starting first part of training (identity covariance)")
    key_pre, key = random.split(key)
    pre_state, pre_losses, pre_evals = trainer.train_and_eval(key_pre, step, pre_n_epochs, pre_state, pre_n_tasks, pre_K, data_noise, maddox_noise, get_train_batch_fn, eval_during_pre_training)
    print("Finished first part of training")

    # FIM
    print("Finding projection matrix")
    key_fim, key_data, key = random.split(key, 3)
    # here we use the exact FIM, we do not need to approximate given the (small) size of the network
    # P1 = fim.proj_exact(key=key_fim, apply_fn=apply_fn, current_params=pre_state.params, current_batch_stats=pre_state.batch_stats, subspace_dimension=subspace_dimension)
    P1 = fim.proj_sketch(key=key_fim, apply_fn=apply_fn, current_params=pre_state.params, batch_stats=pre_state.batch_stats, batches=random.uniform(key_data, shape=(100, 1761, 1), minval=-5, maxval=5), subspace_dimension=subspace_dimension)
    print("Found projection matrix")

    # Usual training with projection
    print("Creating optimizers")
    step = trainer.step_lowdim_cov_singGP
    optimizer_params = optax.adam(learning_rate = meta_lr)
    optimizer_mean = optax.adam(learning_rate = meta_lr)
    optimizer_scale = optax.adam(learning_rate = meta_lr)
    init_scale = np.ones( (subspace_dimension,) )

    post_state = train_states.TrainStateLowDimCovSingGP.create(apply_fn = apply_fn, apply_fn_raw=apply_fn_raw, params = pre_state.params, mean=pre_state.mean, scale=init_scale, tx_params = optimizer_params, tx_mean = optimizer_mean, tx_scale = optimizer_scale, batch_stats=pre_state.batch_stats, proj = P1)

    def eval_during_post_training(key, state):
        current_params = state.params
        current_batch_stats = state.batch_stats
        current_mean = state.mean
        current_scale = state.scale
        kernel, kernel_self, jacobian = ntk.get_kernel_and_jac_lowdim_cov(apply_fn, current_params, current_scale, current_batch_stats, P1)

        subkey_1, subkey_2 = random.split(key)
        nlls = test.test_nll_one_kernel(subkey_1, kernel_self, jacobian, dataset_multi_infinite.get_test_batch, K=pre_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses = test.test_error_one_kernel(subkey_2, kernel, kernel_self, jacobian, dataset_multi_infinite.get_test_batch, dataset_multi_infinite.error_fn, K=pre_K, L=pre_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        return np.mean(nlls), np.mean(mses)

    print("Starting training")
    key_post, key = random.split(key)
    post_state, post_losses, post_evals = trainer.train_and_eval(key_post, step, post_n_epochs, post_state, post_n_tasks, post_K, data_noise, maddox_noise, get_train_batch_fn, eval_during_post_training)
    print("Finished training")

    # Returning everything
    return init_vars, pre_state, pre_evals, post_state, pre_losses, post_losses, post_evals


def unlimtd_f_multi_modal_mixture(seed, pre_n_epochs, pre_n_tasks, pre_K, post_n_epochs, post_n_tasks, post_K, data_noise, maddox_noise, meta_lr, subspace_dimension):
    key = random.PRNGKey(seed)
    key_init, key = random.split(key)

    print("===============")
    print("This is UNLIMTD-F")
    print("For the multi-modal dataset: sine + line (both infinite)")
    print("This variant of UNLIMTD-F approaches the distribution with a mixture of GPs")
    print("===============")

    if pre_n_tasks % 2:
        raise Exception("pre_n_tasks must be divisible by 2 when facing a multi-modal task dataset (equiprobability assumption)")
        
    if post_n_tasks % 2:
        raise Exception("post_n_tasks must be divisible by 2 when facing a multi-modal task dataset (equiprobability assumption)")

    print("Creating model")
    model = models.small_network(40, "relu", 1)
    batch = random.uniform(key_init, shape=(5,1), minval=-5, maxval=5)
    init_vars = model.init(key_init, batch)
    apply_fn = utils.apply_fn_wrapper(model.apply, True)
    apply_fn_raw = model.apply

    # Training before finding the FIM matrix
    # =========================
    # Implementation detail: we do not learn (\mu_1, \mu_2) in this part: we only learn one \mu.
    # This \mu is then chosen as initialization for (\mu_1, \mu_2) when starting the next part of training.
    # =========================
    print("Creating optimizers")
    step = trainer.step_identity_cov
    get_train_batch_fn = dataset_multi_infinite.get_training_batch
    optimizer_params = optax.adam(learning_rate = meta_lr)
    optimizer_mean = optax.adam(learning_rate = meta_lr)
    mean_init = np.zeros( (utils.get_param_size(init_vars["params"]),) )

    pre_state = train_states.TrainStateIdentityCovariance.create(apply_fn=apply_fn, apply_fn_raw=apply_fn_raw, params=init_vars["params"], mean=mean_init, tx_params=optimizer_params, tx_mean=optimizer_mean, batch_stats=FrozenDict())

    def eval_during_pre_training(key, state):
        current_params = state.params
        current_mean = state.mean
        current_batch_stats = state.batch_stats
        kernel, kernel_self, jacobian = ntk.get_kernel_and_jac_identity_cov(apply_fn, current_params, current_batch_stats)

        subkey_1, subkey_2 = random.split(key)
        nlls = test.test_nll_one_kernel(subkey_1, kernel_self, jacobian, dataset_multi_infinite.get_test_batch, K=pre_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)
        mses = test.test_error_one_kernel(subkey_2, kernel, kernel_self, jacobian, dataset_multi_infinite.get_test_batch, dataset_multi_infinite.error_fn, K=pre_K, L=pre_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean=current_mean)

        return np.mean(nlls), np.mean(mses)

    print("Starting first part of training (identity covariance)")
    key_pre, key = random.split(key)
    pre_state, pre_losses, pre_evals = trainer.train_and_eval(key_pre, step, pre_n_epochs, pre_state, pre_n_tasks, pre_K, data_noise, maddox_noise, get_train_batch_fn, eval_during_pre_training)
    print("Finished first part of training")

    # FIM
    print("Finding projection matrix")
    key_fim, key_data, key = random.split(key, 3)
    # here we use the exact FIM, we do not need to approximate given the (small) size of the network
    # P1, P2 = fim.two_proj_exact(key=key_fim, apply_fn=apply_fn, current_params=pre_state.params, current_batch_stats=pre_state.batch_stats, subspace_dimension=subspace_dimension)
    # P1 = fim.proj_exact(key=key_fim, apply_fn=apply_fn, current_params=pre_state.params, current_batch_stats=pre_state.batch_stats, subspace_dimension=subspace_dimension)
    P1 = fim.proj_sketch(key=key_fim, apply_fn=apply_fn, current_params=pre_state.params, batch_stats=pre_state.batch_stats, batches=random.uniform(key_data, shape=(100, 1761, 1), minval=-5, maxval=5), subspace_dimension=subspace_dimension)
    print("Found projection matrix")

    # Usual training with projection
    print("Creating optimizers")
    step = trainer.step_lowdim_cov_mixture
    optimizer_params = optax.adam(learning_rate = meta_lr)
    optimizer_mean1 = optax.adam(learning_rate = meta_lr)
    optimizer_mean2 = optax.adam(learning_rate = meta_lr)
    optimizer_scale1 = optax.adam(learning_rate = meta_lr)
    optimizer_scale2 = optax.adam(learning_rate = meta_lr)

    subkey1, subkey2, key = random.split(key, 3)
    # we use random initialization for the scales, so that the two Gaussians composing the mixture are not identical at the beginning of training, and get a chance to specialize on each cluster
    init_scale1 = np.ones( (subspace_dimension,) ) + random.normal(subkey1, shape=(subspace_dimension,) ) * 0.5
    init_scale2 = np.ones( (subspace_dimension,) ) + random.normal(subkey2, shape=(subspace_dimension,) ) * 0.5

    subkey1, subkey2, key = random.split(key, 3)
    # we use as initialization for the means the mean yielded by the first part of training (see comment about implementation detail above)
    shape = pre_state.mean.shape
    mean1 = pre_state.mean + random.normal(subkey1, shape=shape) * 0
    mean2 = pre_state.mean + random.normal(subkey2, shape=shape) * 0

    post_state = train_states.TrainStateLowDimCovMixture.create(apply_fn = apply_fn, apply_fn_raw=apply_fn_raw, params = pre_state.params, mean1=mean1, mean2=mean2, scale1=init_scale1, scale2=init_scale2, tx_params = optimizer_params, tx_mean1 = optimizer_mean1, tx_mean2 = optimizer_mean2, tx_scale1 = optimizer_scale1, tx_scale2 = optimizer_scale2, batch_stats=pre_state.batch_stats, proj1 = P1, proj2 = P1)

    def eval_during_post_training(key, state):
        current_params = state.params
        current_batch_stats = state.batch_stats
        current_mean1 = state.mean1
        current_mean2 = state.mean2
        current_scale1 = state.scale1
        current_scale2 = state.scale2
        kernel1, kernel1_self, jacobian1 = ntk.get_kernel_and_jac_lowdim_cov(apply_fn, current_params, current_scale1, current_batch_stats, state.proj1)
        kernel2, kernel2_self, jacobian2 = ntk.get_kernel_and_jac_lowdim_cov(apply_fn, current_params, current_scale2, current_batch_stats, state.proj2)

        subkey_1, subkey_2 = random.split(key)
        nlls = test.test_nll_two_kernels(subkey_1, kernel1_self, kernel2_self, jacobian1, jacobian2, dataset_multi_infinite.get_test_batch, K=pre_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean1=current_mean1, current_mean2=current_mean2)
        mses = test.test_error_two_kernels(subkey_2, kernel1, kernel1_self, kernel2, kernel2_self, jacobian1, jacobian2, dataset_multi_infinite.get_test_batch, dataset_multi_infinite.error_fn, K=pre_K, L=pre_K, n_tasks=1000, data_noise=data_noise, maddox_noise=maddox_noise, current_mean1=current_mean1, current_mean2=current_mean2)

        return np.mean(nlls), np.mean(mses)

    print("Starting training")
    key_post, key = random.split(key)
    post_state, post_losses, post_evals = trainer.train_and_eval(key_post, step, post_n_epochs, post_state, post_n_tasks, post_K, data_noise, maddox_noise, get_train_batch_fn, eval_during_post_training)
    print("Finished training")

    # Returning everything
    return init_vars, pre_state, pre_evals, post_state, pre_losses, post_losses, post_evals