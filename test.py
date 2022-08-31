import jax
import nll
import utils
from jax import lax

def test_nll_one_kernel(key, kernel_self, jacobian, get_test_batch_fn, K, n_tasks, data_noise, maddox_noise, current_mean):
    """
    Returns the NLLs for n_tasks random tasks, in the singGP case.
    """
    x_a, y_a, _, _ = get_test_batch_fn(key, n_tasks, K, 0, data_noise)
    all_nlls = nll.nll_batch_one_kernel(kernel_self, x_a, y_a, maddox_noise, jacobian, current_mean)

    return all_nlls

def test_error_one_kernel(key, kernel, kernel_self, jacobian, get_test_batch_fn, error_fn, K, L, n_tasks, data_noise, maddox_noise, current_mean):
    """
    Returns the error for n_tasks random tasks, in the singGP case.
    """
    x_a, y_a, x_b, y_b = get_test_batch_fn(key, n_tasks, K, L, data_noise)

    def f(carry, task_data):
        _x_a, _y_a, _x_b, _y_b = task_data
        _y_a = _y_a - utils.falseaffine_correction0(jacobian, current_mean, _x_a)
        predictions = nll.gaussian_posterior(kernel, kernel_self, _x_a, _y_a, _x_b, maddox_noise)
        predictions = predictions + utils.falseaffine_correction0(jacobian, current_mean, _x_b)
        return None, error_fn(predictions, _y_b)
    
    _, all_errors = lax.scan(f, None, (x_a, y_a, x_b, y_b))

    return all_errors


def test_nll_two_kernels(key, kernel_self1, kernel_self2, jacobian1, jacobian2, get_test_batch_fn, K, n_tasks, data_noise, maddox_noise, current_mean1, current_mean2):
    """
    Returns the NLLs for n_tasks random tasks, in the mixture case.
    """

    x_a, y_a, _, _ = get_test_batch_fn(key, n_tasks, K, 0, data_noise)
    all_nlls = nll.nll_batch_two_kernels(kernel_self1, kernel_self2, x_a, y_a, maddox_noise, jacobian1, jacobian2, current_mean1, current_mean2)

    return all_nlls

def test_error_two_kernels(key, kernel1, kernel_self1, kernel2, kernel_self2, jacobian1, jacobian2, get_test_batch_fn, error_fn, K, L, n_tasks, data_noise, maddox_noise, current_mean1, current_mean2):
    """
    Returns the error for n_tasks random tasks, in the mixture case.
    """
    x_a, y_a, x_b, y_b = get_test_batch_fn(key, n_tasks, K, L, data_noise)

    def f(carry, task_data):
        _x_a, _y_a, _x_b, _y_b = task_data

        _y_a1 = _y_a - utils.falseaffine_correction0(jacobian1, current_mean1, _x_a)
        _y_a2 = _y_a - utils.falseaffine_correction0(jacobian2, current_mean2, _x_a)

        nll1 = nll.nll(kernel_self1, _x_a, _y_a1, maddox_noise)
        nll2 = nll.nll(kernel_self2, _x_a, _y_a2, maddox_noise)

        def f_error1():
            predictions = nll.gaussian_posterior(kernel1, kernel_self1, _x_a, _y_a1, _x_b, maddox_noise)
            predictions = predictions + utils.falseaffine_correction0(jacobian1, current_mean1, _x_b)
            return error_fn(predictions, _y_b)

        def f_error2():
            predictions = nll.gaussian_posterior(kernel2, kernel_self2, _x_a, _y_a2, _x_b, maddox_noise)
            predictions = predictions + utils.falseaffine_correction0(jacobian2, current_mean2, _x_b)
            return error_fn(predictions, _y_b)
        
        return None, lax.cond(nll1 < nll2, f_error1, f_error2)
    
    _, all_errors = lax.scan(f, None, (x_a, y_a, x_b, y_b))

    return all_errors