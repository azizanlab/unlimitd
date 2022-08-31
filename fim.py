import jax
from jax import random
from jax import numpy as np
from jax import jit
import time
from flax import linen as nn

import ntk
import utils

# =========
# SKETCHING
# =========

def proj_sketch(key, apply_fn, current_params, batch_stats, batches, subspace_dimension):
    t = time.time_ns()

    T = 6 * subspace_dimension + 4 
    k = (T - 1) // 3                    # k = 2 * subspace_dimension + 1
    l = T - k                           # l = 4 * subspace_dimension + 3

    U, D = sketch(key, apply_fn, current_params, batch_stats, batches, k, l)

    idx = D.argsort()[::-1]
    P1 = U[:, idx[:subspace_dimension]].T

    print(f"Done sketching in {(time.time_ns() - t)/10**9:.4f} s")

    return P1

def sketch(key, apply_fn, current_params, batch_stats, batches, k, l):
    """
    Returns a good rank 2k approximation of the FIM
    l must be greater than k (often, l = k to limit memory usage)
    """
    M = batches.shape[0]
    N_params = utils.get_param_size(current_params)
    
    n_devices = jax.local_device_count()
    
    if M % n_devices != 0:
        raise Exception(f"Number of batches ({M}) not divisible by number of devices ({n_devices})!")
    
    # This is Algorithm 1 and 2 in Tropp et al
    
    # The matrix we wish to approximate is (m, n) (Tropp's notation)
    # add comment on why we do not need the middle matrix
    # Here, m = n = N_params
    jacobian = ntk.get_jacobian(apply_fn, current_params, batch_stats)
    
    # omega is (k, n) -> (k, N_params) (differs from Tropp) (but it is what we have in sharma's)
    # psi is (l, m) -> (l, N_params)
    # for now we use normal, but why not SFRT
    key_om, key_psi = random.split(key)
    om = random.normal(key_om, shape=(k, N_params))
    psi = random.normal(key_psi, shape=(l, N_params))
    
    def pmapable_fun(sub_batches):
        # sub_batches is an array of batches (n_batches//n_devices, batch_size, input_dims)
        Y0 = np.zeros( (N_params, k) )
        W0 = np.zeros( (l, N_params) )
        
        def f(carry, batch):
            JT = jacobian(batch).T
            
            _Y0, _W0 = carry
            _Y0 = _Y0 + 1/M * ( (om @ JT) @ JT.T ).T
            _W0 = _W0 + 1/M * ( (psi @ JT) @ JT.T )
            
            return (_Y0, _W0), None
        
        (Y0, W0), _ = jax.lax.scan(f, (Y0, W0), sub_batches)
        
        return Y0, W0
    
    # batches is an array of batches (n_batches=M, batch_size, input_dims)

    # Y is (m, k) -> (N_params, k)
    # W is (l, n) -> (l, N_params)
    newshape = (n_devices, M // n_devices) + batches.shape[1:]
    batches = np.reshape(batches, newshape )
    Y, W = jax.pmap(pmapable_fun)(batches)
    Y = np.sum(Y, axis=0)
    W = np.sum(W, axis=0)
    
    # now we have W and Y, sketches of matrix
    # let's find the eigen things
    
    U, D = fixed_rank_eig_approx(Y, W, psi, 2 * k)
    
    return U, D

def fixed_rank_eig_approx(Y, W, psi, r):
    """
    returns U (N x r), D (r) such that A ~= U diag(D) U^T
    """
    
    # this function is FixedRankSymApprox in Algorithm 8 in Tropp et al
    
    # U is (n, 2k) -> (N_params, 2k)
    # S is (2k, 2k)
    U, S = sym_low_rank_approx(Y, W, psi)
    
    # D is (2k) and V is (2k, 2k)
    D, V = np.linalg.eigh(S)
    
    # we truncate the eigendecomposition
    # D is (r) and V is (2k, r)
    D = D[-r:]
    V = V[:, -r:]
    
    # U is (n, r) -> (N_params, r)
    U = U @ V
    
    return U, D

def sym_low_rank_approx(Y, W, psi):
    """
    given Y = A @ Om, (N, k)
    and W = Psi @ A, (l, N)
    and Psi_fn(X) = Psi @ X, (N,...) -> (l,...)
    where Om and Psi and random sketching operators
    returns U (N x 2k), S (2k x 2k) such that A ~= U S U^T
    """
    
    # this function is LowRankSymApprox in Algorithm 5 in Tropp et al
    
    # Q is (m, k) -> (N_params, k)
    # X is (k, n) -> (k, N_params)
    Q, X = low_rank_approx(Y, W, psi)
    k = Q.shape[-1]
    
    # U is (n, 2k) -> (N_params, 2k)
    # T is (2k, 2k)
    tmp = np.concatenate( (Q, X.T), axis=1)
    U, T = np.linalg.qr(tmp, "reduced")
    
    # T1 and T2 are (2k, k)
    T1 = T[:, :k]
    T2 = T[:, k:2*k]
    
    # S is (2k, 2k)
    S = (T1 @ T2.T + T2 @ T1.T) / 2

    return U, S

def low_rank_approx(Y, W, psi):
    """
    given Y = A @ Om, (N, k)
    and W = Psi @ A, (l, M)
    and Psi_fn(X) = Psi @ X, (N,...) -> (l,...)
    where Om and Psi and random sketching operators
    returns Q (N x k), X (k x M) such that A ~= QX
    """
    
    # this function is LowRankApprox Algorithm 4 in Tropp et al
    
    # Y is (m, k) -> (N_params, k)
    # psi is (l, m) -> (l, N_params)
    
    # Q is (m, k) -> (N_params, k) (reduced)
    Q, _ = np.linalg.qr(Y, "reduced")
    
    # U is (l, k)
    # T is (k, k)
    U, T = np.linalg.qr(psi @ Q, "reduced")
    
    # X is (k, n) -> (k, N_params)
    X = jax.scipy.linalg.solve_triangular(T, U.T @ W)

    return Q, X


# =========
# EXACT
# =========


def proj_exact(key, apply_fn, current_params, current_batch_stats, subspace_dimension):
    """
    Finds ONE projection matrix on a subspace of the parameter space (of size subspace_dimension)
    The projection directions are the eigenvectors of the FIM (ie the main directions of the parameter space)
    These projections (and more generally the FIM) are both architecture and parameter dependent.
    """
    F = fim_exact(key, apply_fn, current_params, current_batch_stats)

    w, v = np.linalg.eigh(F)
    w = w.real # eigenvalues
    v = v.real # eigenvectors

    # from highest to lowest
    idx = w.argsort()[::-1]

    P1 = v[:, idx[:subspace_dimension]].T

    return P1

def fim_exact(key, apply_fn, current_params, current_batch_stats, n_JTJ = 100, K = None):
    """
    Computes the Fisher Information Matrix (FIM) of the network.
    The FIM is both parameter-dependent and architecture-dependent.
    FIM = E_{x \sim inputs}(J^\top J). Here we use the empirical mean to estimate this quantity.
    n_JTJ tells how many matrices should be used to compute this approximation of the expectation.
    K is the number of samples used to compute one J^\top J matrix. If K=None, K is set to the number of parameters of the network (so that J is square)
    """
    # 1. Size of matrices
    N = utils.get_param_size(current_params)
    if K is None:
        K = N
    
    jacobian = ntk.get_jacobian(apply_fn, current_params, current_batch_stats)

    def JTJ(inputs):
        J = jacobian(inputs)
        return np.matmul(J.T, J)
    
    @jit
    def update_fim(key, previous_fim):
        key, subkey = random.split(key)
        samples = random.uniform(subkey, shape = (K,1), minval=-5, maxval=5)
        return key, previous_fim + JTJ(samples)
    
    # 3. Computation
    F = np.zeros( (N, N) )
    for _ in range(n_JTJ):
        key, F = update_fim(key, F)
        
    return F / n_JTJ