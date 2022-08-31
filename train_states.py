from typing import Any, Callable

from flax import core
from flax import struct
from jax import numpy as np
import optax

#Train state for the identity covariance training (UNLIMTD-I, or the first part of UNLIMTD-F)

class TrainStateIdentityCovariance(struct.PyTreeNode):
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    apply_fn_raw: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    mean: np.ndarray
    tx_params: optax.GradientTransformation = struct.field(pytree_node=False)
    tx_mean: optax.GradientTransformation = struct.field(pytree_node=False)
    batch_stats: core.FrozenDict[str, Any]
    opt_state_params: optax.OptState
    opt_state_mean: optax.OptState
    
    def apply_gradients(self, *, grads_params, grads_mean, new_batch_stats, **kwargs):
        """
        Updates both the params and the scaling matrix
        Also requires new_batch_stats to keep track of what has been seen by the network
        """

        # params part
        updates_params, new_opt_state_params = self.tx_params.update(grads_params, self.opt_state_params, self.params)
        new_params = optax.apply_updates(self.params, updates_params)

        # mean part
        updates_mean, new_opt_state_mean = self.tx_mean.update(grads_mean, self.opt_state_mean, self.mean)
        new_mean = optax.apply_updates(self.mean, updates_mean)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            mean=new_mean,
            opt_state_params=new_opt_state_params,
            opt_state_mean=new_opt_state_mean,
            batch_stats=new_batch_stats,
            **kwargs,
        )


    @classmethod
    def create(cls, *, apply_fn, apply_fn_raw, params, mean, tx_params, tx_mean, batch_stats, **kwargs):
        opt_state_params = tx_params.init(params)
        opt_state_mean = tx_mean.init(mean)
        return cls(
            step=0,
            apply_fn=apply_fn,
            apply_fn_raw=apply_fn_raw,
            params=params,
            mean=mean,
            tx_params=tx_params,
            tx_mean=tx_mean,
            batch_stats=batch_stats,
            opt_state_params=opt_state_params,
            opt_state_mean=opt_state_mean,
            **kwargs,
        )


#Train state in the singGP case with a low-dimensional covariance matrix (UNLIMTD-R, or the second part of UNLIMTD-F)

class TrainStateLowDimCovSingGP(struct.PyTreeNode):
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    apply_fn_raw: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    mean: np.ndarray
    scale: np.ndarray
    tx_params: optax.GradientTransformation = struct.field(pytree_node=False)
    tx_mean: optax.GradientTransformation = struct.field(pytree_node=False)
    tx_scale: optax.GradientTransformation = struct.field(pytree_node=False)
    batch_stats: core.FrozenDict[str, Any]
    opt_state_params: optax.OptState
    opt_state_mean: optax.OptState
    opt_state_scale: optax.OptState
    proj: np.ndarray
    
    def apply_gradients(self, *, grads_params, grads_mean, grads_scale, new_batch_stats, **kwargs):
        """
        Updates both the params and the scaling matrix
        Also requires new_batch_stats to keep track of what has been seen by the network
        """

        # params part
        updates_params, new_opt_state_params = self.tx_params.update(grads_params, self.opt_state_params, self.params)
        new_params = optax.apply_updates(self.params, updates_params)

        #mean part
        updates_mean, new_opt_state_mean = self.tx_mean.update(grads_mean, self.opt_state_mean, self.mean)
        new_mean = optax.apply_updates(self.mean, updates_mean)

        # scaling matrix part
        updates_scale, new_opt_state_scale = self.tx_scale.update(grads_scale, self.opt_state_scale, self.scale)
        new_scale = optax.apply_updates(self.scale, updates_scale)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            mean=new_mean,
            scale=new_scale,
            opt_state_params=new_opt_state_params,
            opt_state_mean=new_opt_state_mean,
            opt_state_scale=new_opt_state_scale,
            batch_stats=new_batch_stats,
            **kwargs,
        )


    @classmethod
    def create(cls, *, apply_fn, apply_fn_raw, params, mean, scale, tx_params, tx_mean, tx_scale, proj, batch_stats, **kwargs):
        opt_state_params = tx_params.init(params)
        opt_state_mean = tx_mean.init(mean)
        opt_state_scale = tx_scale.init(scale)
        return cls(
            step=0,
            apply_fn=apply_fn,
            apply_fn_raw=apply_fn_raw,
            params=params,
            mean=mean,
            scale=scale,
            tx_params=tx_params,
            tx_mean=tx_mean,
            tx_scale=tx_scale,
            batch_stats=batch_stats,
            opt_state_params=opt_state_params,
            opt_state_mean=opt_state_mean,
            opt_state_scale=opt_state_scale,
            proj=proj,
            **kwargs,
        )



#Train state in the mixture case with two low-dimensional covariance matrices (second part of UNLIMTD-F, in the mixture case)

class TrainStateLowDimCovMixture(struct.PyTreeNode):
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    apply_fn_raw: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]
    mean1: np.ndarray
    mean2: np.ndarray
    scale1: np.ndarray
    scale2: np.ndarray
    tx_params: optax.GradientTransformation = struct.field(pytree_node=False)
    tx_mean1: optax.GradientTransformation = struct.field(pytree_node=False)
    tx_scale1: optax.GradientTransformation = struct.field(pytree_node=False)
    tx_mean2: optax.GradientTransformation = struct.field(pytree_node=False)
    tx_scale2: optax.GradientTransformation = struct.field(pytree_node=False)
    batch_stats: core.FrozenDict[str, Any]
    opt_state_params: optax.OptState
    opt_state_mean1: optax.OptState
    opt_state_scale1: optax.OptState
    opt_state_mean2: optax.OptState
    opt_state_scale2: optax.OptState
    proj1: np.ndarray
    proj2: np.ndarray
    
    def apply_gradients(self, *, grads_params, grads_mean1, grads_mean2, grads_scale1, grads_scale2, new_batch_stats, **kwargs):
        """
        Updates both the params and the scaling matrix
        Also requires new_batch_stats to keep track of what has been seen by the network
        """

        # params part
        updates_params, new_opt_state_params = self.tx_params.update(grads_params, self.opt_state_params, self.params)
        new_params = optax.apply_updates(self.params, updates_params)

        #mean part
        updates_mean1, new_opt_state_mean1 = self.tx_mean1.update(grads_mean1, self.opt_state_mean1, self.mean1)
        new_mean1 = optax.apply_updates(self.mean1, updates_mean1)
        updates_mean2, new_opt_state_mean2 = self.tx_mean2.update(grads_mean2, self.opt_state_mean2, self.mean2)
        new_mean2 = optax.apply_updates(self.mean2, updates_mean2)

        # scaling matrix part
        updates_scale1, new_opt_state_scale1 = self.tx_scale1.update(grads_scale1, self.opt_state_scale1, self.scale1)
        new_scale1 = optax.apply_updates(self.scale1, updates_scale1)
        updates_scale2, new_opt_state_scale2 = self.tx_scale2.update(grads_scale2, self.opt_state_scale2, self.scale2)
        new_scale2 = optax.apply_updates(self.scale2, updates_scale2)
        

        return self.replace(
            step=self.step + 1,
            params=new_params,
            mean1=new_mean1,
            mean2=new_mean2,
            scale1=new_scale1,
            scale2=new_scale2,
            opt_state_params=new_opt_state_params,
            opt_state_mean1=new_opt_state_mean1,
            opt_state_mean2=new_opt_state_mean2,
            opt_state_scale1=new_opt_state_scale1,
            opt_state_scale2=new_opt_state_scale2,
            batch_stats=new_batch_stats,
            **kwargs,
        )


    @classmethod
    def create(cls, *, apply_fn, apply_fn_raw, params, mean1, mean2, scale1, scale2, tx_params, tx_mean1, tx_mean2, tx_scale1, tx_scale2, proj1, proj2, batch_stats, **kwargs):
        opt_state_params = tx_params.init(params)
        opt_state_mean1 = tx_mean1.init(mean1)
        opt_state_mean2 = tx_mean2.init(mean2)
        opt_state_scale1 = tx_scale1.init(scale1)
        opt_state_scale2 = tx_scale2.init(scale2)
        return cls(
            step=0,
            apply_fn=apply_fn,
            apply_fn_raw=apply_fn_raw,
            params=params,
            mean1=mean1,
            mean2=mean2,
            scale1=scale1,
            scale2=scale2,
            tx_params=tx_params,
            tx_mean1=tx_mean1,
            tx_mean2=tx_mean2,
            tx_scale1=tx_scale1,
            tx_scale2=tx_scale2,
            batch_stats=batch_stats,
            opt_state_params=opt_state_params,
            opt_state_mean1=opt_state_mean1,
            opt_state_mean2=opt_state_mean2,
            opt_state_scale1=opt_state_scale1,
            opt_state_scale2=opt_state_scale2,
            proj1=proj1,
            proj2=proj2,
            **kwargs,
        )