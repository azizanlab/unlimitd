from functools import partial
from jax import jit
from jax import numpy as np
from jax import random
from jax import lax
import dataset_sines_infinite
import pickle

@jit
def error_fn(prediction, ground_truth):
    return np.mean( (ground_truth - prediction)**2 )

@partial(jit, static_argnums=(1, 2, 3, 4))
def get_training_batch(key, n_tasks, K, data_noise, n_devices):
    # we keep data_noise for compat
    x_a, y_a = get_raw_batch(key, n_tasks, K, 0, x_train, y_train, y_train_unnoised)

    x_a_div, y_a_div = np.reshape(x_a, (n_devices, n_tasks//n_devices, K, 1)), np.reshape(y_a, (n_devices, n_tasks//n_devices, K, 1))
    
    return x_a, y_a, x_a_div, y_a_div

@partial(jit, static_argnums=(1, 2, 3, 4))
def get_train_batch_as_val_batch(key, n_tasks, K, L, data_noise):
    # we keep data_noise for compat
    x, y = get_raw_batch(key, n_tasks, K, L, x_train, y_train, y_train_unnoised)
    
    return x[:, :K], y[:, :K], x[:, K:], y[:, K:]

@partial(jit, static_argnums=(1, 2, 3, 4))
def get_val_batch(key, n_tasks, K, L, data_noise):
    # we keep data_noise for compat
    x, y = get_raw_batch(key, n_tasks, K, L, x_val, y_val, y_val_unnoised)
    
    return x[:, :K], y[:, :K], x[:, K:], y[:, K:]

@partial(jit, static_argnums=(1, 2, 3, 4))
def get_test_batch(key, n_tasks, K, L, data_noise):
    # we keep data_noise for compat
    x, y = get_raw_batch(key, n_tasks, K, L, x_test, y_test, y_test_unnoised)
    
    return x[:, :K], y[:, :K], x[:, K:], y[:, K:]

def get_raw_batch(key, n_tasks, K, L, x_source, y_source, y_unnoised_source):
    key, subkey_tasks = random.split(key)
    n_datapoints = K+L

    n_max_tasks = x_source.shape[0]
    n_max_datapoints = x_source.shape[1]

    chosen_tasks = random.randint(subkey_tasks, shape = (n_tasks, 1), minval = 0, maxval = n_max_tasks)
    chosen_datapoints_indexes0 = np.empty( (n_tasks, n_datapoints), dtype=int)

    def f(task_index, value):
        key, chosen_datapoints_indexes = value

        key, key_datapoints = random.split(key)
        chosen_datapoints_indexes = chosen_datapoints_indexes.at[task_index, :].set(random.choice(key_datapoints, n_max_datapoints, shape=(n_datapoints,), replace=False))

        return key, chosen_datapoints_indexes

    chosen_datapoints_indexes = lax.fori_loop(0, n_tasks, f, (key, chosen_datapoints_indexes0) )[1]

    x = x_source[chosen_tasks, chosen_datapoints_indexes, :]
    y = y_source[chosen_tasks, chosen_datapoints_indexes[:, :K], :]
    y_unnoised = y_unnoised_source[chosen_tasks, chosen_datapoints_indexes[:, K:], :]

    return x, np.concatenate( (y, y_unnoised), axis=1)



x_train = None
y_train = None
y_train_unnoised = None
fun_train = None
n_train_tasks = None
n_train_datapoints = None

x_val = None
y_val = None
y_val_unnoised = None
n_val_tasks = None
n_val_datapoints = None

x_test = None
y_test = None
y_test_unnoised = None
fun_test = None
n_test_tasks = None
n_test_datapoints = None

def init_dataset(key, data_noise, _n_train_tasks=50, _n_train_datapoints=50, _n_val_datapoints=50, _n_test_tasks=50, _n_test_datapoints=50):
    reg_dim = 1

    def build_data(key, fun_list, n_total_datapoints):
        x = []
        y = []
        y_unnoised = []
        n_tasks = len(fun_list)

        for i in range(n_tasks):
            key, key_x, key_noise = random.split(key, 3)
            
            _fun = fun_list[i]
            _x = random.uniform(key_x, shape = (1, n_total_datapoints, 1), minval=-5, maxval=5)
            _y_unnoised = _fun(_x)
            _y = _y_unnoised + random.normal(key_noise, shape=(1,n_total_datapoints, reg_dim)) * data_noise

            x.append(_x)
            y.append(_y)
            y_unnoised.append(_y_unnoised)
        
        return np.concatenate(x, axis=0), np.concatenate(y, axis=0), np.concatenate(y_unnoised, axis=0)

    def build_fun_list(key, n_tasks):
        fun_list = []

        for _ in range(n_tasks):
            key, key_fun = random.split(key)
            _fun = dataset_sines_infinite.draw_multi(key_fun, reg_dim)
            fun_list.append(_fun)

        return fun_list

    global x_train, y_train, y_train_unnoised, fun_train, n_train_tasks, n_train_datapoints
    n_train_tasks = _n_train_tasks
    n_train_datapoints = _n_train_datapoints

    key, subkey1, subkey2 = random.split(key, 3)
    fun_train = build_fun_list(subkey1, n_train_tasks)
    x_train, y_train, y_train_unnoised = build_data(subkey2, fun_train, n_train_datapoints)

    global x_val, y_val, y_val_unnoised, n_val_tasks, n_val_datapoints
    n_val_tasks = _n_train_tasks
    n_val_datapoints = _n_val_datapoints

    key, subkey = random.split(key)
    x_val, y_val, y_val_unnoised = build_data(subkey, fun_train, n_val_datapoints)

    global x_test, y_test, y_test_unnoised, fun_test, n_test_tasks, n_test_datapoints
    n_test_tasks = _n_test_tasks
    n_test_datapoints = _n_test_datapoints

    key, subkey1, subkey2 = random.split(key, 3)
    fun_test = build_fun_list(subkey1, n_test_tasks)
    x_test, y_test, y_test_unnoised = build_data(subkey2, fun_test, n_test_datapoints)

def get_fancy_train_batch(key, K, L, data_noise):
    # data_noise is kept for compat
    subkey1, subkey2 = random.split(key)
    chosen_task = random.randint(subkey1, shape = (1,), minval = 0, maxval = n_train_tasks)
    chosen_datapoints_indexes = random.choice(subkey2, n_train_datapoints, shape=(K+L,), replace=False)

    x = x_train[chosen_task, chosen_datapoints_indexes, :]
    y = y_train[chosen_task, chosen_datapoints_indexes[:K], :]
    y_unnoised = y_train_unnoised[chosen_task, chosen_datapoints_indexes[K:], :]

    return x[np.newaxis, ...], np.concatenate( (y, y_unnoised), axis=0)[np.newaxis, ...], fun_train[int(chosen_task)]

def get_fancy_val_batch(key, K, L, data_noise):
    # data_noise is kept for compat
    subkey1, subkey2 = random.split(key)
    chosen_task = random.randint(subkey1, shape = (1,), minval = 0, maxval = n_val_tasks)
    chosen_datapoints_indexes = random.choice(subkey2, n_val_datapoints, shape=(K+L,), replace=False)

    x = x_val[chosen_task, chosen_datapoints_indexes, :]
    y = y_val[chosen_task, chosen_datapoints_indexes[:K], :]
    y_unnoised = y_val_unnoised[chosen_task, chosen_datapoints_indexes[K:], :]

    return x[np.newaxis, ...], np.concatenate( (y, y_unnoised), axis=0)[np.newaxis, ...], fun_train[int(chosen_task)]

def get_fancy_test_batch(key, K, L, data_noise):
    # data_noise is kept for compat
    subkey1, subkey2 = random.split(key)
    chosen_task = random.randint(subkey1, shape = (1,), minval = 0, maxval = n_test_tasks)
    chosen_datapoints_indexes = random.choice(subkey2, n_test_datapoints, shape=(K+L,), replace=False)

    x = x_test[chosen_task, chosen_datapoints_indexes, :]
    y = y_test[chosen_task, chosen_datapoints_indexes[:K], :]
    y_unnoised = y_test_unnoised[chosen_task, chosen_datapoints_indexes[K:], :]

    return x[np.newaxis, ...], np.concatenate( (y, y_unnoised), axis=0)[np.newaxis, ...], fun_test[int(chosen_task)]