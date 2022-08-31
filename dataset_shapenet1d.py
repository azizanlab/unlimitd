from functools import partial
from jax import jit
from jax import numpy as np
from jax import random
from jax import lax

import pickle

@jit
def error_fn(pred_cos_sin, gd_angle):
    # pred_cos_sin is of shape(L, 2)
    # ground_truth_angle is of shape (L,)

    # find pred_angles, in [0, 2pi]
    pred_angle = np.arctan2(pred_cos_sin[:, 1], pred_cos_sin[:, 0])
    pred_angle = np.where(pred_angle < 0 , 2*np.pi + pred_angle, pred_angle)
    
    angle_errors =  angle_error(gd_angle, pred_angle)

    # returns the average for this batch (in degrees!)
    return 360 * np.mean(angle_errors) / (2 * np.pi)

def angle_error(gd_angle, pred_angle):
    error_a = np.abs(pred_angle + 2 * np.pi - gd_angle)[:, np.newaxis]
    error_b = np.abs(pred_angle - 2 * np.pi - gd_angle)[:, np.newaxis]
    error_c = np.abs(pred_angle - gd_angle)[:, np.newaxis]
    
    errors = np.concatenate( (error_a, error_b, error_c), axis=1 )
    
    return errors.min(axis=1)

@partial(jit, static_argnums=(1, 2, 3, 4))
def get_test_batch(key, n_tasks, K, L, data_noise):
    x, y = get_raw_batch(key, n_tasks, K, L, data_noise, x_test, y_test)

    # y_a is (cos, sin), they are noised
    # y_b is angle, not noised (gt)
    x_a = x[:, :K]
    y_a = y[:, :K, 0:2]
    x_b = x[:, K:]
    y_b = y[:, K:, 2]
    
    return x_a, y_a, x_b, y_b

@partial(jit, static_argnums=(1, 2, 3, 4))
def get_val_batch(key, n_tasks, K, L, data_noise):
    x, y = get_raw_batch(key, n_tasks, K, L, data_noise, x_val, y_val)

    # y_a is (cos, sin), they are noised
    # y_b is angle, not noised (gt)
    x_a = x[:, :K]
    y_a = y[:, :K, 0:2]
    x_b = x[:, K:]
    y_b = y[:, K:, 2]
    
    return x_a, y_a, x_b, y_b

def get_maml_train_batch(key, n_tasks, K, L, data_noise):
    x, y = get_raw_batch(key, n_tasks, K, L, data_noise, x_train, y_train)

    # y_a is (cos, sin), they are noised
    # y_b is angle, not noised (gt)
    x_a = x[:, :K]
    y_a = y[:, :K, 0:2]
    x_b = x[:, K:]  
    #y_b = y[:, K:, 0:2]
    y_b = y[:, K:, :]
    
    return x_a, y_a, x_b, y_b

def get_maml_val_batch(key, n_tasks, K, L, data_noise):
    x, y = get_raw_batch(key, n_tasks, K, L, data_noise, x_val, y_val)

    # y_a is (cos, sin), they are noised
    # y_b is angle, not noised (gt)
    x_a = x[:, :K]
    y_a = y[:, :K, 0:2]
    x_b = x[:, K:]  
    #y_b = y[:, K:, 0:2]
    y_b = y[:, K:, :]
    
    return x_a, y_a, x_b, y_b

def get_maml_test_batch(key, n_tasks, K, L, data_noise):
    x, y = get_raw_batch(key, n_tasks, K, L, data_noise, x_test, y_test)

    # y_a is (cos, sin), they are noised
    # y_b is angle, not noised (gt)
    x_a = x[:, :K]
    y_a = y[:, :K, 0:2]
    x_b = x[:, K:]  
    #y_b = y[:, K:, 0:2]
    y_b = y[:, K:, :]
    
    return x_a, y_a, x_b, y_b

#@partial(jit, static_argnums=(1, 2, 3, 4))
def get_train_batch_as_val_batch(key, n_tasks, K, L, data_noise):
    # temporary
    x, y = get_raw_batch(key, n_tasks, K, L, data_noise, x_train, y_train)

    # y_a is (cos, sin), they are noised
    # y_b is angle, not noised (gt)
    x_a = x[:, :K]
    y_a = y[:, :K, 0:2]
    x_b = x[:, K:]  
    y_b = y[:, K:, 2]
    
    return x_a, y_a, x_b, y_b

@partial(jit, static_argnums=(1, 2, 3, 4))
def get_training_batch(key, n_tasks, K, data_noise, n_devices):
    x_a, y_a = get_raw_batch(key, n_tasks, K, 0, data_noise, x_train, y_train)
    # train on cos/sin, not on angle
    y_a = y_a[..., 0:2]

    x_a_div, y_a_div = np.reshape(x_a, (n_devices, n_tasks//n_devices, K, 128, 128, 1)), np.reshape(y_a, (n_devices, n_tasks//n_devices, K, 2))
    
    return x_a, y_a, x_a_div, y_a_div

def get_raw_batch(key, n_tasks, K, L, data_noise, x_source, y_source):
    """
    Returns x_a, y_a of shape:
    x_a (n_tasks, n_datapoints, 128, 128, 1)
    y_a (n_tasks, n_datapoints, 3) (3 because cos, sin and angle)

    Following our convention, n_datapoints is K when training (not the case during val or test, where we would add test inputs)
    An extra noise has been added to the first K datapoints (considered "training" data)
    The last datapoints (K to K+L) are not noised (considered as "ground truth")
    """
    key, subkey_tasks = random.split(key)
    n_datapoints = K+L

    n_max_tasks = x_source.shape[0]
    n_max_datapoints = x_source.shape[1]

    # print(n_max_tasks)
    # print(n_max_datapoints)

    # x_train[ chosen_tasks[i], ... ] returns all samples for the ith chosen task
    chosen_tasks = random.randint(subkey_tasks, shape = (n_tasks, 1), minval = 0, maxval = n_max_tasks)
    # x_train[ task_index, chosen_datapoints_indexes, ...] returns the chosen datapoints with shape (n_datapoints, 128, 128)
    chosen_datapoints_indexes0 = np.empty( (n_tasks, n_datapoints), dtype=int)

    def f(task_index, value):
        key, chosen_datapoints_indexes = value

        key, key_datapoints = random.split(key)
        chosen_datapoints_indexes = chosen_datapoints_indexes.at[task_index, :].set(random.choice(key_datapoints, n_max_datapoints, shape=(n_datapoints,), replace=False))

        return key, chosen_datapoints_indexes

    chosen_datapoints_indexes = lax.fori_loop(0, n_tasks, f, (key, chosen_datapoints_indexes0) )[1]
    # print("Chosen datapoints", chosen_datapoints_indexes)

    # print("Chosen tasks", chosen_tasks)

    x, y = x_source[chosen_tasks, chosen_datapoints_indexes, :, :], y_source[chosen_tasks, chosen_datapoints_indexes, :]

    # add noise
    # y = y.at[:, :K, 0:2].set(y[:, :K, 0:2] + random.normal(key, shape=(n_tasks, K, 2)) * data_noise)

    return x, y


x_train = None
y_train = None
x_val = None
y_val = None
x_test = None
y_test = None

n_train_tasks = None
n_train_datapoints = None

n_val_tasks = None
n_val_datapoints = None

n_test_tasks = None
n_test_datapoints = None

def load_shapenet1d():
    # ================
    # DATA PREPARATION
    # After preparation:
    # - x_train is of shape (1350, 50, 128, 128, 1) (27*50 tasks, 50 samples per task, 128x128 images)
    # - y_train is of shape (1350, 50, 3) (27*50 tasks, 50 samples per task, angle (cos, sin and angle))
    #
    # - x_val is of shape (266, 50, 128, 128, 1) (~27*10 tasks, 50 samples per task, 128x128 images)
    # - y_val is of shape (266, 50, 3) (~27*10 tasks, 50 samples per task, angle (cos, sin and angle))
    #
    # - x_test is of shape (60, 50, 128, 128, 1) (3*20 tasks, 50 samples per task, 128x128 images)
    # - y_test is of shape (60, 50, 3) (3*20 tasks, 50 samples per task, angle (cos, sin and angle))
    #
    # ================

    print("Loading training dataset")

    data_size = "large"
    global x_train, y_train, n_train_tasks, n_train_datapoints
    # Replace the path data with your own below
    x_train, y_train = pickle.load(open(f"~/ShapeNet1D/train_data_{data_size}.pkl", 'rb'))

    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]

    y_train = np.array(y_train)
    y_train = y_train[:, :, -1, None]
    y_train = y_train*2*np.pi
    y_train = np.concatenate([np.cos(y_train), np.sin(y_train), y_train], axis=-1)

    n_train_tasks = x_train.shape[0]
    n_train_datapoints = x_train.shape[1]

    print("Training ataset ready")

    print("Loading val dataset")

    global x_val, y_val, n_val_tasks, n_val_datapoints
    x_val, y_val = pickle.load(open("/home/gridsan/calmecija/research/what-matters/what-matters-for-meta-learning/data/ShapeNet1D/val_data.pkl", 'rb'))

    x_val = np.array(x_val)
    x_val = x_val[..., np.newaxis]

    y_val = np.array(y_val)
    y_val = y_val[:, :, -1, None]
    y_val = y_val*2*np.pi
    y_val = np.concatenate([np.cos(y_val), np.sin(y_val), y_val], axis=-1)

    n_val_tasks = x_val.shape[0]
    n_val_datapoints = x_val.shape[1]

    print("Loaded val dataset")

    print("Loading test dataset")

    global x_test, y_test, n_test_tasks, n_test_datapoints
    x_test, y_test = pickle.load(open("/home/gridsan/calmecija/research/what-matters/what-matters-for-meta-learning/data/ShapeNet1D/test_data.pkl", 'rb'))

    x_test = np.array(x_test)
    x_test = x_test[..., np.newaxis]

    y_test = np.array(y_test)
    y_test = y_test[:, :, -1, None]
    y_test = y_test*2*np.pi
    y_test = np.concatenate([np.cos(y_test), np.sin(y_test), y_test], axis=-1)

    n_test_tasks = x_test.shape[0]
    n_test_datapoints = x_test.shape[1]

    print("Loaded test dataset")

