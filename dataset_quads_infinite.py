from functools import partial
from jax import jit
from jax import numpy as np
from jax import random
from jax import lax
from jax import vmap

@jit
def error_fn(prediction, ground_truth):
    return np.mean( (ground_truth - prediction)**2 )

@partial(jit, static_argnums=(1, 2, 3, 4))
def get_training_batch(key, n_tasks, K, data_noise, n_devices):
    x_a, y_a = get_raw_batch(key, n_tasks, K, 0, data_noise)

    x_a_div, y_a_div = np.reshape(x_a, (n_devices, n_tasks//n_devices, K, 1)), np.reshape(y_a, (n_devices, n_tasks//n_devices, K, 1))
    
    return x_a, y_a, x_a_div, y_a_div

@partial(jit, static_argnums=(1, 2, 3, 4))
def get_test_batch(key, n_tasks, K, L, data_noise):
    x, y = get_raw_batch(key, n_tasks, K, L, data_noise)
    
    return x[:, :K], y[:, :K], x[:, K:], y[:, K:]

def get_fancy_test_batch(key, K, L, data_noise):
    reg_dim = 1

    key_x, key = random.split(key)
    key_fun, key_noise, key = random.split(key, 3)
    function = draw_multi(key_fun, reg_dim)
    
    x = random.uniform(key_x, shape = (1, K+L, 1), minval=-5, maxval=5)
    y = function(x)
    y = y.at[:, :K].set(y[:, :K] + random.normal(key_noise, shape=(1,K, reg_dim)) * data_noise)

    return x, y, function

def get_raw_batch(key, n_tasks, K, L, data_noise):
    # set this higher for a multi-dimensional regression
    reg_dim = 1

    key_x, key = random.split(key)
    x = random.uniform(key_x, shape = (n_tasks, K+L, 1), minval=-5, maxval=5)
    
    y = np.empty( (n_tasks, K+L, reg_dim) )
    
    def f(task_index, value):
        y, key = value

        key_fun, key_noise, key = random.split(key, 3)

        function = draw_multi(key_fun, reg_dim)
        y = y.at[task_index, :K, :].set(function(x[task_index, :K]) + random.normal(key_noise, shape=(K, reg_dim)) * data_noise)
        y = y.at[task_index, K:, :].set(function(x[task_index, K:]))

        return (y, key)
            
    return x, lax.fori_loop(0, n_tasks, f, (y, key) )[0]

def draw_multi(key, reg_dim, a_low=-0.2, a_high=0.2, phase_low=-2, phase_high=2):
    key_a, key_phase = random.split(key)
    
    a = random.uniform(key_a, shape=(reg_dim,), minval=a_low, maxval=a_high)
    phase = random.uniform(key_phase, shape=(reg_dim,), minval=phase_low, maxval=phase_high)
    
    def function(x):
        return a * (x - phase) ** 2 + 0.5
        
    return vmap(function)