from functools import partial
from jax import jit
from jax import numpy as np
from jax import random
from jax import lax
from jax import vmap
import dataset_sines_infinite

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

@partial(jit, static_argnums=(1, 2, 3, 4))
def get_test_batch_only_sines(key, n_tasks, K, L, data_noise):
    x, y = dataset_sines_infinite.get_raw_batch(key, n_tasks, K, L, data_noise)
    
    return x[:, :K], y[:, :K], x[:, K:], y[:, K:]

@partial(jit, static_argnums=(1, 2, 3, 4))
def get_test_batch_only_lines(key, n_tasks, K, L, data_noise):
    x, y = get_raw_batch_only_lines(key, n_tasks, K, L, data_noise)
    
    return x[:, :K], y[:, :K], x[:, K:], y[:, K:]

def get_fancy_test_batch_sine(key, K, L, data_noise):
    reg_dim = 1

    key_x, key = random.split(key)
    key_fun, key_noise, key = random.split(key, 3)
    function = draw_multi_sine(key_fun, reg_dim)
    
    x = random.uniform(key_x, shape = (1, K+L, 1), minval=-5, maxval=5)
    y = function(x)
    y = y.at[:, :K].set(y[:, :K] + random.normal(key_noise, shape=(1,K, reg_dim)) * data_noise)

    return x, y, function

def get_fancy_test_batch_line(key, K, L, data_noise):
    reg_dim = 1

    key_x, key = random.split(key)
    key_fun, key_noise, key = random.split(key, 3)
    function = draw_multi_line(key_fun, reg_dim)
    
    x = random.uniform(key_x, shape = (1, K+L, 1), minval=-5, maxval=5)
    y = function(x)
    y = y.at[:, :K].set(y[:, :K] + random.normal(key_noise, shape=(1,K, reg_dim)) * data_noise)

    return x, y, function

def get_raw_batch(key, n_tasks, K, L, data_noise):
    # set this higher for a multi-dimensional regression
    reg_dim = 1

    key_x_line, key_x_sine, key = random.split(key, 3)
    x_line = random.uniform(key_x_line, shape = (n_tasks // 2, K+L, 1), minval=-5, maxval=5)
    x_sine = random.uniform(key_x_sine, shape = (n_tasks // 2, K+L, 1), minval=-5, maxval=5)
    
    y_sine = np.empty( (n_tasks // 2, K+L, reg_dim) )
    y_line = np.empty( (n_tasks // 2, K+L, reg_dim) )
    
    def f(task_index, value):
        y, key = value

        key_fun, key_noise, key = random.split(key, 3)

        function_sine = draw_multi_sine(key_fun, reg_dim)
        y = y.at[task_index, :K, :].set(function_sine(x_sine[task_index, :K]) + random.normal(key_noise, shape=(K, reg_dim)) * data_noise)
        y = y.at[task_index, K:, :].set(function_sine(x_sine[task_index, K:]))

        return (y, key)

    def g(task_index, value):
        y, key = value

        key_fun, key_noise, key = random.split(key, 3)

        function_line = draw_multi_line(key_fun, reg_dim)
        y = y.at[task_index, :K, :].set(function_line(x_line[task_index, :K]) + random.normal(key_noise, shape=(K, reg_dim)) * data_noise)
        y = y.at[task_index, K:, :].set(function_line(x_line[task_index, K:]))

        return (y, key)
    
    key_sine, key_line = random.split(key)
    y_sine = lax.fori_loop(0, n_tasks // 2, f, (y_sine, key_sine) )[0]
    y_line = lax.fori_loop(0, n_tasks // 2, g, (y_line, key_line) )[0]

    return np.concatenate( (x_sine, x_line), axis=0 ), np.concatenate( (y_sine, y_line), axis=0)

def get_raw_batch_only_lines(key, n_tasks, K, L, data_noise):
    # set this higher for a multi-dimensional regression
    reg_dim = 1

    key_x, key = random.split(key)
    x_line = random.uniform(key_x, shape = (n_tasks, K+L, 1), minval=-5, maxval=5)

    y_line = np.empty( (n_tasks, K+L, reg_dim) )

    def g(task_index, value):
        y, key = value

        key_fun, key_noise, key = random.split(key, 3)

        function_line = draw_multi_line(key_fun, reg_dim)
        y = y.at[task_index, :K, :].set(function_line(x_line[task_index, :K]) + random.normal(key_noise, shape=(K, reg_dim)) * data_noise)
        y = y.at[task_index, K:, :].set(function_line(x_line[task_index, K:]))

        return (y, key)
    
    y_line = lax.fori_loop(0, n_tasks, g, (y_line, key) )[0]

    return x_line, y_line

def draw_multi_sine(key, reg_dim, amp_low=0.1, amp_high=5, phase_low=0, phase_high=np.pi):
    key_amp, key_phase = random.split(key)
    
    amps = random.uniform(key_amp, shape=(reg_dim,), minval=amp_low, maxval=amp_high)
    phases = random.uniform(key_phase, shape=(reg_dim,), minval=phase_low, maxval=phase_high)
    
    def function(x):
        return amps * np.sin(x + phases) + 1
        
    return vmap(function)

def draw_multi_line(key, reg_dim, slope_low=-1, slope_high=1):
    slopes = random.uniform(key, shape=(reg_dim,), minval=slope_low, maxval=slope_high)
    
    def function(x):
        return slopes * x
        
    return vmap(function)