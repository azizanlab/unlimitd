# shapenet config (from what-matters)
config = {}
config["n_epochs"] = 50000
config["n_tasks_per_epoch"] = 10
config["K"] = 15
config["L"] = 15
config["n_updates"] = 5
config["n_updates_test"]=20
config["meta_lr"] = 0.0005
config["inner_lr"] = 0.002
config["data_noise"] = 0
config["n_test_tasks"] = 100

import dataset_shapenet1d
import models

from jax import vmap
from jax import numpy as np
from jax import random
from jax import value_and_grad, grad
from jax.tree_util import tree_map
from jax.lax import scan
from jax import jit
from flax import struct
from flax import core
import optax

from typing import Any, Callable
import time
from functools import partial
import pickle

print("Testing dumping...")

with open("logs_final/shapenet_maml_20.pickle", "wb") as handle:
    pickle.dump({}, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Dumping done!")

def error_fn_train(pred_cos_sin, gt_cos_sin):
    # shape of cos_sin_pred / cos_sin_gt is (batch_size, 2)
    tmp = np.sum((pred_cos_sin - gt_cos_sin)**2, axis = 1)
    return np.mean(tmp)

def angle_error(gd_angle, pred_angle):
    error_a = np.abs(pred_angle + 2 * np.pi - gd_angle)[:, np.newaxis]
    error_b = np.abs(pred_angle - 2 * np.pi - gd_angle)[:, np.newaxis]
    error_c = np.abs(pred_angle - gd_angle)[:, np.newaxis]
    
    errors = np.concatenate( (error_a, error_b, error_c), axis=1 )
    
    return errors.min(axis=1)

def error_fn_test(pred_cos_sin, gt_angle):
    pred_angle = np.arctan2(pred_cos_sin[:, 1], pred_cos_sin[:, 0])
    pred_angle = np.where(pred_angle < 0 , 2*np.pi + pred_angle, pred_angle)
    angle_errors =  angle_error(gt_angle, pred_angle)

    # returns the average for this batch (in degrees!)
    return 360 * np.mean(angle_errors) / (2 * np.pi)

def inner_loss(current_params, x_a, y_a, apply_fn):
    predictions = apply_fn(current_params, x_a)
    
    return error_fn_train(predictions, y_a)

def gd_step0(inner_lr, param_value, param_grad):
    return param_value - inner_lr * param_grad

def inner_updates(current_params, x_a, y_a, n_updates, inner_lr, apply_fn):
    def f(parameters, x):
        inner_gradients = grad(inner_loss)(parameters, x_a, y_a, apply_fn)
        parameters = tree_map(partial(gd_step0, inner_lr), parameters, inner_gradients)
        
        return parameters, None
    
    updated_params, _ = scan(f, current_params, None, n_updates)
    
    return updated_params

def outer_loss_single_task(current_params, x_a, y_a, x_b, y_b, n_updates, inner_lr, apply_fn):
    updated_params = inner_updates(current_params, x_a, y_a, n_updates, inner_lr, apply_fn)
    
    predictions = apply_fn(updated_params, x_b)
    
    return error_fn_train(predictions, y_b)

def outer_loss(current_params, x_a, y_a, x_b, y_b, n_updates, inner_lr, apply_fn):
    unaveraged_losses = vmap(partial(outer_loss_single_task, current_params=current_params, n_updates=n_updates, inner_lr=inner_lr, apply_fn=apply_fn))(x_a=x_a, y_a=y_a, x_b=x_b, y_b=y_b)
    return np.mean(unaveraged_losses)

class TrainState(struct.PyTreeNode):
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    tx_params: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state_params: optax.OptState
    inner_lr: float
    n_updates: int
    
    def apply_gradients(self, *, grads_params, **kwargs):
        """
        Updates both the params and the scaling matrix
        Also requires new_batch_stats to keep track of what has been seen by the network
        """
        # params part
        updates_params, new_opt_state_params = self.tx_params.update(grads_params, self.opt_state_params, self.params)
        new_params = optax.apply_updates(self.params, updates_params)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state_params=new_opt_state_params,
            **kwargs,
        )


    @classmethod
    def create(cls, *, apply_fn, params, tx_params, inner_lr, n_updates, **kwargs):
        opt_state_params = tx_params.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx_params=tx_params,
            opt_state_params=opt_state_params,
            inner_lr = inner_lr,
            n_updates = n_updates,
            **kwargs,
        )

@partial(jit, static_argnums=(5, 6, 7))
def get_loss_and_gradients(params, x_a, y_a, x_b, y_b, n_updates, inner_lr, apply_fn):
    return value_and_grad(outer_loss)(params, x_a, y_a, x_b, y_b, n_updates, inner_lr, apply_fn)

def step(key, state):
    x_a, y_a, x_b, y_b = get_train_batch_fn(key)
    
    loss, gradients = get_loss_and_gradients(state.params, x_a, y_a, x_b, y_b, state.n_updates, state.inner_lr, state.apply_fn)
    
    state = state.apply_gradients(grads_params = gradients)
    
    return state, loss

def get_train_batch_fn(key):
    # context: cos and sin (for fast adapt)
    # query: cos and sin (to compute outer loss)
    x_a, y_a, x_b, y_b = dataset_shapenet1d.get_maml_train_batch(key, config["n_tasks_per_epoch"], config["K"], config["L"], config["data_noise"])
    return x_a, y_a, x_b, y_b[:, :, 0:2]
    
def get_train_batch_as_val_batch_fn(key):
    # context: cos and sin (for fast adapt)
    # query: angle (just to compare, because no outer loss here)
    x_a, y_a, x_b, y_b = dataset_shapenet1d.get_maml_train_batch(key, config["n_tasks_per_epoch"], config["K"], config["L"], config["data_noise"])
    return x_a, y_a, x_b, y_b[:, :, 2]

def get_val_batch_fn(key):
    # return sine_dataset.get_test_batch(k
    # context: cos and sin (for fast adapt)
    # query: angle (just to compare, because no outer loss here)
    x_a, y_a, x_b, y_b = dataset_shapenet1d.get_maml_val_batch(key, config["n_tasks_per_epoch"], config["K"], config["L"], config["data_noise"])
    return x_a, y_a, x_b, y_b[:, :, 2]

def get_test_batch_fn(key):
    # context: cos and sin (for fast adapt)
    # query: angle (just to compare, because no outer loss here)
    x_a, y_a, x_b, y_b = dataset_shapenet1d.get_maml_test_batch(key, config["n_tasks_per_epoch"], config["K"], config["L"], config["data_noise"])
    return x_a, y_a, x_b, y_b[:, :, 2]

def do_test(state, x_a, y_a, x_b, y_b):
    def f(carry, task):
        x_a, y_a, x_b, y_b = task
        
        updated_params = inner_updates(state.params, x_a, y_a, config["n_updates_test"], state.inner_lr, state.apply_fn)
        predictions = state.apply_fn(updated_params, x_b)
        
        return None, error_fn_test(predictions, y_b)
    
    _, errors = scan(f, None, (x_a, y_a, x_b, y_b))
    return np.mean(errors)

def test_during_training(key, state):
    x_a, y_a, x_b, y_b = get_train_batch_as_val_batch_fn(key)
    error_train = do_test(state, x_a, y_a, x_b, y_b)
    
    x_a, y_a, x_b, y_b = get_val_batch_fn(key)
    error_val = do_test(state, x_a, y_a, x_b, y_b)
    
    x_a, y_a, x_b, y_b = get_test_batch_fn(key)
    error_test = do_test(state, x_a, y_a, x_b, y_b)
    
    return (error_train, error_val, error_test)

dataset_shapenet1d.load_shapenet1d()
key = random.PRNGKey(0)

model = models.deep_network(2, True)

def apply_fn(params, inputs):
    return model.apply({"params": params}, inputs, mutable=["batch_stats"])[0]

key, key_init0, key_init1 = random.split(key, 3)
batch = get_train_batch_fn(key_init0)
init_vars = model.init(key_init1, batch[0][0])

optimizer_params = optax.adam(learning_rate = config["meta_lr"])

state = TrainState.create(apply_fn=apply_fn, params=init_vars["params"], tx_params=optimizer_params, inner_lr=config["inner_lr"], n_updates=config["n_updates"])

losses = []
errors_test = []

for epoch_index in range(config["n_epochs"]):
#for epoch_index in range(10):
    t = time.time_ns()
    key, subkey = random.split(key)
    state, current_loss = step(subkey, state)
    
    if epoch_index % 10 == 0:
        print(f"{epoch_index} | {current_loss:.4f} ({(time.time_ns() - t) / 10**9:.4f} s / epoch)")
        
    if epoch_index % 500 == 0:
        # test time
        key, subkey = random.split(key)
        mse_test = test_during_training(subkey, state)
        errors_test.append(mse_test)
        print(f"Error: {mse_test}")
    
    losses.append(current_loss)

output = {}
output["trained_params"] = state.params
output["losses"]=losses
output["errors_test"]=errors_test
output["config"] = config

with open("logs_final/shapenet_maml_20.pickle", "wb") as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)