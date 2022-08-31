from jax import numpy as np
from flax import linen as nn

def deep_network(reg_dim, training):
    """
    Returns a deep neural network for the vision problem on Shapenet1D.
    Use reg_dim to control the dimension of the output.
    training regulates the use of the running statistics
    """

    if training:
        use_running_average = False
    else:
        use_running_average = True

    class CNN(nn.Module):
        @nn.compact
        def __call__(self, x):
            # WARNING: NHWC
            # -1, 128x128x1
            
            x = nn.Conv(features = 32, kernel_size = (3,3), strides = 2, padding = 1)(x)
            x = nn.relu(x)
            # -1, 64x64x32
            
            x = nn.Conv(features = 48, kernel_size = (3,3), strides = 2, padding = 1)(x)
            x = nn.relu(x)
            # -1, 32x32x48
            
            x = nn.max_pool(x, (2,2), strides = (2,2))
            # -1, 16x16x48
            
            x = nn.Conv(features = 64, kernel_size = (3,3), strides = 2, padding = 1)(x)
            x = nn.relu(x)
            # -1, 8x8x64
            
            x = np.reshape(x, (-1, 4096))
            # -1, 4096
            
            x = nn.Dense(196)(x)
            # -1, 196
            
            x = np.reshape(x, (-1, 14, 14, 1))
            # -1, 14x14x1
            
            x = nn.Conv(features = 64, kernel_size = (3,3), strides = 1, padding = 1)(x)
            x = nn.BatchNorm(use_running_average=use_running_average, momentum=0)(x) # we use batch statistics only
            x = nn.relu(x)
            # -1, 14x14x64
            
            x = nn.Conv(features = 64, kernel_size = (3,3), strides = 1, padding = 1)(x)
            x = nn.BatchNorm(use_running_average=use_running_average, momentum=0)(x) # we use batch statistics only
            x = nn.relu(x)
            # -1, 14x14x64
            
            x = nn.Conv(features = 64, kernel_size = (3,3), strides = 1, padding = 1)(x)
            x = nn.BatchNorm(use_running_average=use_running_average, momentum=0)(x) # we use batch statistics only
            x = nn.relu(x)
            # -1, 14x14x64
            
            x = nn.avg_pool(x, (14, 14))
            # -1, 1x1x64
            
            x = np.reshape(x, (-1, 64))
            # -1, 64
            
            x = nn.Dense(reg_dim)(x)
            
            return x
    
    return CNN()

def small_network(n_neurons, activation, reg_dim):
    """
    Returns a small neural network (two layers, n_neurons per layer with specified activation)
    Use reg_dim to control the dimension of the output

    Compatible activations: "relu" and "tanh" (note: all experiments run with ReLU).
    """

    if activation == "relu":
        act_fn = nn.relu
    elif activation == "tanh":
        act_fn = nn.tanh

    class Regressor(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(n_neurons)(x)
            x = act_fn(x)

            x = nn.Dense(n_neurons)(x)
            x = act_fn(x)

            x = nn.Dense(reg_dim)(x)

            x = np.reshape(x, (-1, reg_dim))

            return x
    
    return Regressor()