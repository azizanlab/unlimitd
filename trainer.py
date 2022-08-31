from jax import random
from jax import numpy as np
from jax import pmap
import jax
from jax import value_and_grad
from jax import jit
import time
from jax import lax

import nll

def train_and_eval(key, step, n_epochs, state, n_tasks, K, data_noise, maddox_noise, get_train_batch_fn, eval_during_training_fn):
    """
    Available step functions:
    * step_identity_cov
    * step_lowdim_cov_singGP
    * step_lowdim_cov_mixture

    Available get_train_batch_fn functions:
    * dataset_sines_infinite.get_training_batch
    * dataset_sines_finite.get_training_batch
    * dataset_lines_infinite.get_training_batch
    * dataset_multi_infinite.get_training_batch
    * dataset_shapenet1d.get_training_batch
    
    """
    n_devices = jax.local_device_count()

    print("Starting training with:")
    print(f"-n_epochs={n_epochs}")
    print(f"-n_tasks={n_tasks}")
    print(f"-K={K}")
    print(f"-data_noise={data_noise}")
    print(f"-maddox_noise={maddox_noise}")

    losses = []
    evals = []
    t = time.time_ns()

    for epoch_index in range(n_epochs):
        key, subkey = random.split(key)
        state, current_loss = step(subkey, state, n_tasks, K, data_noise, maddox_noise, n_devices, get_train_batch_fn)

        if(np.isnan(current_loss)):
            print("Nan, aborting")
            break
        
        losses.append(current_loss)

        if epoch_index % 10 == 0:
            print(f"{epoch_index}  | {current_loss:.4f} ({(time.time_ns() - t)/ 10**9:.4f} s)")
        t = time.time_ns()

        if epoch_index % 500 == 0:
            key, subkey = random.split(key)
            current_eval = eval_during_training_fn(subkey, state)
            evals.append( current_eval )
            print(f"Eval: {current_eval}")
    print("Completed training")

    return state, losses, evals



def step_identity_cov(key, current_state, n_tasks, K, data_noise, maddox_noise, n_devices, get_train_batch_fn):
    # Draw the samples for this step, and split it to prepare for pmap (jit'd)
    x_a, y_a, x_a_div, y_a_div = get_train_batch_fn(key, n_tasks, K, data_noise, n_devices)
    
    # Compute loss and gradient through gpu parallelization
    unaveraged_losses, (unaveraged_gradients_p, unaveraged_gradients_m) = pmap(pmapable_loss_identity_cov,
                             in_axes=(None, 0, 0, None),
                             static_broadcasted_argnums=(3)
                            )(current_state, x_a_div, y_a_div, maddox_noise)
    
    current_loss = np.mean(unaveraged_losses)
    current_gradients_p = jax.tree_map(lambda array: np.mean(array, axis=0), unaveraged_gradients_p)
    current_gradients_m = jax.tree_map(lambda array: np.mean(array, axis=0), unaveraged_gradients_m)
    
    # Update batch_stats "manually" (jit'd)
    new_batch_stats = batch_stats_updater(current_state, x_a)
    
    # Update state (parameters and optimizer)
    current_state = grad_applier_identity_cov(current_state, current_gradients_p, current_gradients_m, new_batch_stats)
    
    return current_state, current_loss

def step_lowdim_cov_singGP(key, current_state, n_tasks, K, data_noise, maddox_noise, n_devices, get_train_batch_fn):
    # Draw the samples for this step, and split it to prepare for pmap (jit'd)
    x_a, y_a, x_a_div, y_a_div = get_train_batch_fn(key, n_tasks, K, data_noise, n_devices)
    
    # Compute loss and gradient through gpu parallelization
    unaveraged_losses, (unaveraged_gradients_p, unaveraged_gradients_m, unaveraged_gradients_s) = pmap(pmapable_loss_lowdim_cov_singGP,
                             in_axes=(None, 0, 0, None),
                             static_broadcasted_argnums=(3)
                            )(current_state, x_a_div, y_a_div, maddox_noise)
    
    current_loss = np.mean(unaveraged_losses)
    current_gradients_p = jax.tree_map(lambda array: np.mean(array, axis=0), unaveraged_gradients_p)
    current_gradients_m = jax.tree_map(lambda array: np.mean(array, axis=0), unaveraged_gradients_m)
    current_gradients_s = jax.tree_map(lambda array: np.mean(array, axis=0), unaveraged_gradients_s)
    
    # Update batch_stats "manually" (jit'd)
    new_batch_stats = batch_stats_updater(current_state, x_a)
    
    # Update state (parameters and optimizer)
    current_state = grad_applier_lowdim_cov_singGP(current_state, current_gradients_p, current_gradients_m, current_gradients_s, new_batch_stats)
    
    return current_state, current_loss

def step_lowdim_cov_mixture(key, current_state, n_tasks, K, data_noise, maddox_noise, n_devices, get_train_batch_fn):
    # Draw the samples for this step, and split it to prepare for pmap (jit'd)
    x_a, y_a, x_a_div, y_a_div = get_train_batch_fn(key, n_tasks, K, data_noise, n_devices)
    
    # Compute loss and gradient through gpu parallelization
    unaveraged_losses, (unaveraged_gradients_p, unaveraged_gradients_m1, unaveraged_gradients_m2, unaveraged_gradients_s1, unaveraged_gradients_s2) = pmap(pmapable_loss_lowdim_cov_mixture,
                             in_axes=(None, 0, 0, None),
                             static_broadcasted_argnums=(3)
                            )(current_state, x_a_div, y_a_div, maddox_noise)
    
    current_loss = np.mean(unaveraged_losses)
    current_gradients_p = jax.tree_map(lambda array: np.mean(array, axis=0), unaveraged_gradients_p)
    current_gradients_m1 = jax.tree_map(lambda array: np.mean(array, axis=0), unaveraged_gradients_m1)
    current_gradients_m2 = jax.tree_map(lambda array: np.mean(array, axis=0), unaveraged_gradients_m2)
    current_gradients_s1 = jax.tree_map(lambda array: np.mean(array, axis=0), unaveraged_gradients_s1)
    current_gradients_s2 = jax.tree_map(lambda array: np.mean(array, axis=0), unaveraged_gradients_s2)
    
    # Update batch_stats "manually" (jit'd)
    new_batch_stats = batch_stats_updater(current_state, x_a)
    
    # Update state (parameters and optimizer)
    current_state = grad_applier_lowdim_cov_mixture(current_state, current_gradients_p, current_gradients_m1, current_gradients_m2, current_gradients_s1, current_gradients_s2, new_batch_stats)
    
    return current_state, current_loss





def pmapable_loss_identity_cov(current_state, x_a, y_a, maddox_noise):
    # we can't pass current_state because we have to explicitely show the variable
    loss, (gradients_p, gradients_m) = value_and_grad(nll.nll_batch_average_identity_cov, argnums = (0, 1) )(current_state.params,
                                                              current_state.mean,
                                                              current_state.apply_fn,
                                                              current_state.batch_stats,
                                                              x_a,
                                                              y_a,
                                                              maddox_noise)
    
    return loss, (gradients_p, gradients_m)

def pmapable_loss_lowdim_cov_singGP(current_state, x_a, y_a, maddox_noise):
    # we can't pass current_state because we have to explicitely show the variable
    loss, (gradients_p, gradients_m, gradients_s) = value_and_grad(nll.nll_batch_average_lowdim_cov_singGP, argnums = (0, 1, 2) )(current_state.params,
                                                              current_state.mean,
                                                              current_state.scale,
                                                              current_state.apply_fn,
                                                              current_state.batch_stats,
                                                              current_state.proj,
                                                              x_a,
                                                              y_a,
                                                              maddox_noise)
    
    return loss, (gradients_p, gradients_m, gradients_s)

def pmapable_loss_lowdim_cov_mixture(current_state, x_a, y_a, maddox_noise):
    # we can't pass current_state because we have to explicitely show the variable
    loss, (gradients_p, gradients_m1, gradients_m2, gradients_s1, gradients_s2) = value_and_grad(nll.nll_batch_average_lowdim_cov_mixture, argnums = (0, 1, 2, 3, 4) )(current_state.params,
                                                              current_state.mean1,
                                                              current_state.mean2,
                                                              current_state.scale1,
                                                              current_state.scale2,
                                                              current_state.apply_fn,
                                                              current_state.batch_stats,
                                                              current_state.proj1,
                                                              current_state.proj2,
                                                              x_a,
                                                              y_a,
                                                              maddox_noise)
    
    return loss, (gradients_p, gradients_m1, gradients_m2, gradients_s1, gradients_s2)


@jit
def grad_applier_identity_cov(current_state, gradients_p, gradients_m, new_batch_stats):
    return current_state.apply_gradients(grads_params=gradients_p, grads_mean=gradients_m, new_batch_stats=new_batch_stats)

@jit
def grad_applier_lowdim_cov_singGP(current_state, gradients_p, gradients_m, gradients_s, new_batch_stats):
    return current_state.apply_gradients(grads_params=gradients_p, grads_mean=gradients_m, grads_scale=gradients_s, new_batch_stats=new_batch_stats)

@jit
def grad_applier_lowdim_cov_mixture(current_state, gradients_p, gradients_m1, gradients_m2, gradients_s1, gradients_s2, new_batch_stats):
    return current_state.apply_gradients(grads_params=gradients_p, grads_mean1=gradients_m1, grads_mean2=gradients_m2, grads_scale1=gradients_s1, grads_scale2=gradients_s2, new_batch_stats=new_batch_stats)

@jit
def batch_stats_updater(current_state, x_a):
    # shape of x_a is (n_tasks, batch_size, inputs_dims...)
    
    batch_stats = current_state.batch_stats
    
    def f(old_batch_stats, _x_a):
        # shape of _x_a is (batch_size, input_dims)
        _, mutated_vars = current_state.apply_fn_raw({"params":current_state.params,
                                                      "batch_stats": old_batch_stats},
                                                     _x_a,
                                                     mutable=["batch_stats"])
        
        new_batch_stats = mutated_vars["batch_stats"]
        return new_batch_stats, None
    
    batch_stats, _ = lax.scan(f, batch_stats, x_a)
    
    return batch_stats