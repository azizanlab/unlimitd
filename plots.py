from jax import numpy as np
from matplotlib import pyplot as plt
import utils
import nll
import dataset_multi_infinite

def plot_notebooks(key, kernel, kernel_self, jac, mean, K, dataset_provider):
    """
    Make an informative prediction plot in the singGP case (for the kernel specified)
    K is the number of context inputs
    Change dataset_provider to test on other datasets (e.g. dataset_sines_infinite)
    """
    x, y, fun = dataset_provider.get_fancy_test_batch(key, K=10, L=0, data_noise=0.05)

    x_a_all = x[0, :10]
    y_a_all = y[0, :10]
    x_b = np.linspace(-5, 5, 100)[:, np.newaxis]
    y_b = fun(x_b)

    y_min, y_max = np.min(y_b) - 0.5, np.max(y_b) + 0.5

    correction_a_all = utils.falseaffine_correction0(jac, mean, x_a_all)
    correction_b = utils.falseaffine_correction0(jac, mean, x_b)

    x_a = x_a_all[:K]
    y_a = y_a_all[:K]
    correction_a = correction_a_all[:K]

    prediction, cov = nll.gaussian_posterior_full(kernel, kernel_self, x_a, y_a - correction_a, x_b, 0.05)
    prediction = prediction + correction_b

    error = dataset_provider.error_fn(prediction, y_b)
    loss = nll.nll(kernel_self, x_a, y_a - correction_a, maddox_noise=0.05)

    variances = np.diag(cov)
    stds = np.sqrt(variances)

    plt.plot(x_b, y_b, "g--", label="Target")
    plt.plot(x_a, y_a, "ro", label="Context data")
    plt.plot(x_b, prediction, "b", label="Prediction")
    plt.fill_between(x_b[:, 0], prediction[:, 0] - 1.96 * stds, prediction[:, 0] + 1.96 * stds, color='blue', alpha=0.1, label="+/- 1.96$\sigma$")
    plt.title(f"NLL={loss:.4f}, MSE={error:.4f} ($K$={K})")
    plt.legend()
    plt.gca().set_ylim([y_min, y_max])
    plt.gca().set_xlabel("$x$")
    plt.gca().set_ylabel("$y$")
    plt.legend()

def plot_notebooks_two_kernels(key, kernel1, kernel1_self, kernel2, kernel2_self, jac, mean1, mean2, K, task_type):
    """
    Make an informative prediction plot in the mixture case (for the kernels specified)
    K is the number of context inputs
    Change dataset_provider to test on other datasets (e.g. dataset_sines_infinite)
    """

    if task_type == "sine":
        x, y, fun = dataset_multi_infinite.get_fancy_test_batch_sine(key, K=10, L=0, data_noise=0.05)
    elif task_type == "line":
        x, y, fun = dataset_multi_infinite.get_fancy_test_batch_line(key, K=10, L=0, data_noise=0.05)
    else:
        raise Exception("Invalid task type")

    x_a_all = x[0, :10]
    y_a_all = y[0, :10]
    x_b = np.linspace(-5, 5, 100)[:, np.newaxis]
    y_b = fun(x_b)

    y_min, y_max = np.min(y_b) - 0.5, np.max(y_b) + 0.5

    correction1_a_all = utils.falseaffine_correction0(jac, mean1, x_a_all)
    correction2_a_all = utils.falseaffine_correction0(jac, mean2, x_a_all)

    x_a = x_a_all[:K]
    y_a = y_a_all[:K]
    correction1_a = correction1_a_all[:K]
    correction2_a = correction2_a_all[:K]

    nll1 = nll.nll(kernel1_self, x_a, y_a - correction1_a, maddox_noise=0.05)
    nll2 = nll.nll(kernel2_self, x_a, y_a - correction2_a, maddox_noise=0.05)

    loss = nll.nll_logaddexp(nll1, nll2)

    if nll1 < nll2:
        print("Most probable kernel is 1")
        mean = mean1
        kernel = kernel1
        kernel_self = kernel1_self
        correction_a = correction1_a
    else:
        print("Most probable kernel is 2")
        mean = mean2
        kernel = kernel2
        kernel_self = kernel2_self
        correction_a = correction2_a

    correction_b = utils.falseaffine_correction0(jac, mean, x_b)

    prediction, cov = nll.gaussian_posterior_full(kernel, kernel_self, x_a, y_a - correction_a, x_b, 0.05)
    prediction = prediction + correction_b

    error = dataset_multi_infinite.error_fn(prediction, y_b)

    variances = np.diag(cov)
    stds = np.sqrt(variances)

    plt.plot(x_b, y_b, "g--", label="Target")
    plt.plot(x_a, y_a, "ro", label="Context data")
    plt.plot(x_b, prediction, "b", label="Prediction")
    plt.fill_between(x_b[:, 0], prediction[:, 0] - 1.96 * stds, prediction[:, 0] + 1.96 * stds, color='blue', alpha=0.1, label="+/- 1.96$\sigma$")
    plt.title(f"NLL={loss:.4f}, MSE={error:.4f} ($K$={K})")
    plt.legend()
    plt.gca().set_ylim([y_min, y_max])
    plt.gca().set_xlabel("$x$")
    plt.gca().set_ylabel("$y$")
    plt.legend()


"""
BELOW: other type of plots that are useful in some notebooks
"""


def plot_1d_pred(key, kernel, kernel_self, get_fancy_test_batch, error_fn, K, L, data_noise, maddox_noise, plot_cov = False):
    x, y, fun = get_fancy_test_batch(key, K, L, data_noise)

    x_a = x[0, :K]
    y_a = y[0, :K]
    x_b = x[0, K:]
    y_b = y[0, K:]

    x_plot = np.linspace(-5, 5, 100)[:, np.newaxis]

    loss = nll.nll(kernel_self, x_a, y_a, maddox_noise)
    if plot_cov:
        predictions, cov = nll.gaussian_posterior_full(kernel, kernel_self, x_a, y_a, x_plot, maddox_noise)    
        variances = np.diag(cov)
        stds = np.sqrt(variances)
    else:
        predictions = nll.gaussian_posterior(kernel, kernel_self, x_a, y_a, x_plot, maddox_noise)
    error = error_fn(nll.gaussian_posterior(kernel, kernel_self, x_a, y_a, x_b, maddox_noise), y_b)

    
    plt.plot(x_plot, fun(x_plot), "--g", label="True function")
    plt.plot(x_plot, predictions, "b", label="Prediction")
    plt.plot(x_a, y_a, "or", label="Adapt data")
    plt.plot(x_b, y_b, "xr", label="Test data")

    if plot_cov:
        plt.fill_between(x_plot[:, 0], predictions[:, 0] - 1.96 * stds, predictions[:, 0] + 1.96 * stds,
                 color='blue', alpha=0.1, label="+/- 1.96σ")

    plt.legend()
    plt.title(f"NLL: {float(loss):.4f} | Error: {float(error):.4f}")
    plt.show()

    if plot_cov:
        plt.plot(x_plot, stds, label="σ")
        plt.plot(x_a, np.ones( (K,1) ) * np.nanmin(stds), "or", label="Adapt data")
        plt.legend()
        plt.title("Posterior standard deviation")
        plt.show()

def plot_1d_pred_falseaffine(key, kernel, kernel_self, get_fancy_test_batch, error_fn, K, L, data_noise, maddox_noise, jacobian, trained_mean, plot_cov = False):
    x, y, fun = get_fancy_test_batch(key, K, L, data_noise)

    x_a = x[0, :K]
    y_a = y[0, :K]
    x_b = x[0, K:]
    y_b = y[0, K:]

    x_plot = np.linspace(-5, 5, 100)[:, np.newaxis]

    correction_a = utils.falseaffine_correction0(jacobian, trained_mean, x_a)
    correction_plot = utils.falseaffine_correction0(jacobian, trained_mean, x_plot)
    correction_b = utils.falseaffine_correction0(jacobian, trained_mean, x_b)

    loss = nll.nll(kernel_self, x_a, y_a - correction_a, maddox_noise)
    
    if plot_cov:
        predictions, cov = nll.gaussian_posterior_full(kernel, kernel_self, x_a, y_a - correction_a, x_plot, maddox_noise)    
        predictions = predictions + correction_plot
        variances = np.diag(cov)
        stds = np.sqrt(variances)
    else:
        predictions = nll.gaussian_posterior(kernel, kernel_self, x_a, y_a - correction_a, x_plot, maddox_noise)
        predictions = predictions + correction_plot
    error = error_fn(nll.gaussian_posterior(kernel, kernel_self, x_a, y_a - correction_a, x_b, maddox_noise) + correction_b, y_b)

    
    plt.plot(x_plot, fun(x_plot), "--g", label="True function")
    plt.plot(x_plot, predictions, "b", label="Prediction")
    plt.plot(x_a, y_a, "or", label="Adapt data")
    plt.plot(x_b, y_b, "xr", label="Test data")

    if plot_cov:
        plt.fill_between(x_plot[:, 0], predictions[:, 0] - 1.96 * stds, predictions[:, 0] + 1.96 * stds,
                 color='blue', alpha=0.1, label="+/- 1.96σ")

    plt.legend()
    plt.title(f"NLL: {float(loss):.4f} | Error: {float(error):.4f}")
    plt.show()

    if plot_cov:
        plt.plot(x_plot, stds, label="σ")
        plt.plot(x_a, np.ones( (K,1) ) * np.nanmin(stds), "or", label="Adapt data")
        plt.legend()
        plt.title("Posterior standard deviation")
        plt.show()


def plot_1d_pred_falseaffine_two_kernels(key, kernel1, kernel_self1, kernel2, kernel_self2, get_fancy_test_batch, error_fn, K, L, data_noise, maddox_noise, jacobian1, jacobian2, trained_mean1, trained_mean2, plot_cov = False):
    x, y, fun = get_fancy_test_batch(key, K, L, data_noise)

    x_a = x[0, :K]
    y_a = y[0, :K]
    x_b = x[0, K:]
    y_b = y[0, K:]

    x_plot = np.linspace(-5, 5, 100)[:, np.newaxis]

    correction_a1 = utils.falseaffine_correction0(jacobian1, trained_mean1, x_a)
    correction_a2 = utils.falseaffine_correction0(jacobian2, trained_mean2, x_a)

    nll1 = nll.nll(kernel_self1, x_a, y_a - correction_a1, maddox_noise)
    nll2 = nll.nll(kernel_self2, x_a, y_a - correction_a2, maddox_noise)

    loss = nll.nll_logaddexp(nll1, nll2)

    if nll1 < nll2:
        jacobian = jacobian1
        trained_mean = trained_mean1
        kernel = kernel1
        kernel_self = kernel_self1
        correction_a = correction_a1
        print("Most probable kernel is 1")
    else:
        jacobian = jacobian2
        trained_mean = trained_mean2
        kernel = kernel2
        kernel_self = kernel_self2
        correction_a = correction_a2
        print("Most probable kernel is 2")

    correction_plot = utils.falseaffine_correction0(jacobian, trained_mean, x_plot)
    correction_b = utils.falseaffine_correction0(jacobian, trained_mean, x_b)
    
    if plot_cov:
        predictions, cov = nll.gaussian_posterior_full(kernel, kernel_self, x_a, y_a - correction_a, x_plot, maddox_noise)    
        predictions = predictions + correction_plot
        variances = np.diag(cov)
        stds = np.sqrt(variances)
    else:
        predictions = nll.gaussian_posterior(kernel, kernel_self, x_a, y_a - correction_a, x_plot, maddox_noise)
        predictions = predictions + correction_plot
    error = error_fn(nll.gaussian_posterior(kernel, kernel_self, x_a, y_a - correction_a, x_b, maddox_noise) + correction_b, y_b)

    
    plt.plot(x_plot, fun(x_plot), "--g", label="True function")
    plt.plot(x_plot, predictions, "b", label="Prediction")
    plt.plot(x_a, y_a, "or", label="Adapt data")
    plt.plot(x_b, y_b, "xr", label="Test data")

    if plot_cov:
        plt.fill_between(x_plot[:, 0], predictions[:, 0] - 1.96 * stds, predictions[:, 0] + 1.96 * stds,
                 color='blue', alpha=0.1, label="+/- 1.96σ")

    plt.legend()
    plt.title(f"NLL: {float(loss):.4f} | Error: {float(error):.4f}")
    plt.show()

    if plot_cov:
        plt.plot(x_plot, stds, label="σ")
        plt.plot(x_a, np.ones( (K,1) ) * np.nanmin(stds), "or", label="Adapt data")
        plt.legend()
        plt.title("Posterior standard deviation")
        plt.show()

def plot_fancy_kernel(kernel, title="Kernel"):
    x_plot = np.linspace(-5, 5, 100)[:, np.newaxis]
    
    plt.plot(x_plot, kernel(np.array([[0]]), x_plot)[0, :], label="k(0,.)")
    plt.plot(x_plot, kernel(np.array([[np.pi]]), x_plot)[0, :], label="k(pi,.)")
    plt.legend()
    plt.title(title)
    plt.show()

def return_fancy_kernel(kernel):
    x_plot = np.linspace(-5, 5, 100)[:, np.newaxis]
    y_plot_0 = kernel(np.array([[0]]), x_plot)[0, :]
    y_plot_pi = kernel(np.array([[np.pi]]), x_plot)[0, :]
    
    return x_plot, y_plot_0, y_plot_pi

def plot_kernel(kernel, title="Kernel"):
    x_plot = np.linspace(-5, 5, 100)[:, np.newaxis]
    
    plt.plot(x_plot, kernel(np.array([[0]]), x_plot)[0, :], label="k(0,.)")
    plt.plot(x_plot, kernel(np.array([[1]]), x_plot)[0, :], label="k(1,.)")
    plt.plot(x_plot, kernel(np.array([[2]]), x_plot)[0, :], label="k(2,.)")
    plt.plot(x_plot, kernel(np.array([[3]]), x_plot)[0, :], label="k(3,.)")
    plt.plot(x_plot, kernel(np.array([[4]]), x_plot)[0, :], label="k(4,.)")
    plt.legend()
    plt.title(title)
    plt.show()

def plot_loss(losses, moving_average=None, title="Loss", y_min=None, y_max=None, other_losses = None, other_labels=None, other_period=None):
    if type(losses) is list:
        losses = np.array(losses)

    if moving_average is None:
        plt.plot(np.arange(losses.shape[0]), losses)
    else:
        average = np.convolve(losses, np.ones(moving_average), 'valid') / moving_average
        plt.plot(np.arange(average.shape[0]), average, label=f"NLL (m.a. of {moving_average})")

    if not(other_losses is None):
        if type(other_losses) is list:
            other_losses = np.array(other_losses)
        
        n = other_losses.shape[1]
        x_axis = other_period * np.arange(other_losses.shape[0])
        for k in range(n):
            plt.plot(x_axis, other_losses[:, k], "x", label=other_labels[k])

    plt.legend()
    plt.title(title)
    if not(y_min is None) and not(y_max is None):
        plt.gca().set_ylim([y_min, y_max])
    
    plt.show()

def plot_various_losses(losses, period, legends, title="Loss", y_min=None, y_max=None):
    for i, loss in enumerate(losses):
        if type(loss) is list:
            loss = np.array(loss)
        plt.plot(period * np.arange(loss.shape[0]), loss, "x", label=legends[i])
        
    plt.title(title)
    plt.legend()
    if not(y_min is None) and not(y_max is None):
        plt.gca().set_ylim([y_min, y_max])
    plt.show()

def plot_occ_loss(losses, period, title="Loss", y_min=None, y_max=None):
    if type(losses) is list:
        losses = np.array(losses)
        
    plt.plot(period * np.arange(losses.shape[0]), losses, "x")
    plt.title(title)
    if not(y_min is None) and not(y_max is None):
        plt.gca().set_ylim([y_min, y_max])
    plt.show()