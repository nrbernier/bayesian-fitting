"""Tools to plot the results of the Bayesian fitting algorithm

Here are provided a few functions that help visualize the result of the fitting algorithm of the module bayesian_fit.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas.plotting as ptp
import corner
import time

iter_stable_default = 200


def corner_plot_distribution(results_pd, 
        prefix='', iter_stable = iter_stable_default):
    """Plots the distributions and correlations of each parameters in a so-called corner plot

    Parameters:
        results_pd (Pandas dataframe): result of Bayesian fitting
        prefix (str): path and prefix to prepend to the generated plot
        iter_stable (int): iteration number at which the chain is assumed to have converged. The data is plotted after that step only. 

    Output:
        One beautiful corner plot
    """
    varnames = results_pd.columns
    inds_after_transient = results_pd.index.get_level_values('Iteration') > iter_stable
    means = results_pd.loc[inds_after_transient].mean()
    fig = corner.corner(results_pd[inds_after_transient], 
            labels=varnames,
            truths= means,
            bins=25,)
    fig.savefig(prefix + 'Bayesian_fit_corner.png')

def plot_fitting_uncertainty(results_pd, data_x, data_y, fitfunc,
        prefix='', iter_stable=iter_stable_default, sample_size=1000,
        xlabel='x', ylabel='y', use_std=True):
    """Plots the spread of the fitting in data space

    For each point x, the spread in parameter space is translated to a spread in y-space. The original data points are plotted in blue alongside the "best fit" (corresponding to the mean of the fitted parameters and a spread representing one standard deviation from the mean. Alternatively (if use_std is False), a sample of parameters are drawn and the functions they represent are all plotted in gray with some transparency, to otbain a similar effect. 

    Parameters:
        results_pd (Pandas dataframe): result of Bayesian fitting
        data_x (num array): data x values
        data_y (num array): data y values
        fitfunc (function): function that was fitted
        prefix (str): path and prefix to prepend to the generated plot
        iter_stable (int): iteration number at which the chain is assumed to have converged. Only parameters values after that step are used in the computation. 
        sample_size (int): number of random parameter samples to use to compute the spread
        xlabel (str): label of the x axis in the plot
        ylabel (str): label of the y axis in the plot
        use_std (bool): if true, plots one standard deviation away from the mean, if false, plots all curves that are sampled semi-transparently.
    
    """

    inds_after_transient = results_pd.index.get_level_values('Iteration') > iter_stable
    means = results_pd.loc[inds_after_transient].mean()
    x_array_fit = np.linspace(data_x.min(), data_x.max(), 401)
    fig, ax = plt.subplots()
    ax.scatter(data_x, data_y, color='xkcd:cerulean')

    if use_std:
        y_models = np.zeros((x_array_fit.size, sample_size))
        i=0
        for ind, fparams in results_pd[inds_after_transient].sample(n=sample_size).iterrows():
            y_models[:,i] = fitfunc(x_array_fit, fparams[:-1])
            i +=1

        ax.fill_between(x_array_fit, 
                np.mean(y_models, axis=1) - np.std(y_models, axis=1),
                np.mean(y_models, axis=1) + np.std(y_models, axis=1),
                color='gray', alpha=0.7, zorder=3)
    else:
        for ind, fparams in results_pd[inds_after_transient].sample(n=sample_size).iterrows():
            ax.plot(x_array_fit, 
                    fitfunc(x_array_fit, fparams[:-1]),
                        color='gray', alpha=0.01)

    ax.plot(x_array_fit, fitfunc(x_array_fit, means[:-1]), 
            '-', color='xkcd:crimson', zorder=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(data_x.min(), data_x.max())
    fig.savefig(prefix + 'Bayesian_fit_uncertainty.png', dpi=400)

def plot_walker_statistics(results_pd, n_walkers_autocor=3, 
        prefix='', iter_stable = iter_stable_default):
    """Plots autocorrelation for each parameter and how fast the walkers converge

    Parameters:
        results_pd (Pandas dataframe): result of Bayesian fitting
        n_walkers_autocor (int): number of walkers picked randomly to compute the autocorrelation (the computation can be slightly time-consuming)
        prefix (str): path and prefix to prepend to the generated plot
        iter_stable (int): iteration number at which the chain is assumed to have converged (here it is used only to determine the window to compute the autocorrelation)

    Output:
        One plot for each parameter with the autocorrelation and the trajectory of the walkers in parameter space
    """

    t0 = time.time()
    varnames = results_pd.columns
    for var in varnames:
        single_parameter_plots(results_pd[var], iter_stable, var, 
                n_walkers_autocor, prefix)

def single_parameter_plots(df_var, iter_stable, varname, n_walkers_autocor, prefix=''):
    """Plots autocorrelation and statistics for a single parameter

    function used by plot_walker_statistics to plot for a single parameter

    Parameters:
        df_var (Pandas dataframe): results of fitting for a single parameter
        iter_stable (int): number of iterations after which the chain is assumed to converge
        varname (str): name of the parameter
        n_walkers_autocor (int): number of chains to randomly choose to compute the autocorrelation
        prefix (str): path and prefix to prepend to the generated plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle("Parameter {}".format(varname))

    t0 = time.time()
 
    # Autocorrelation for each chain:
    axes[0].set_xlim(0, iter_stable*2)
    pd_walker = df_var.unstack(level=0)
    for walker in pd_walker.sample(axis=1, n=n_walkers_autocor):
        ptp.autocorrelation_plot(pd_walker[walker][:iter_stable*2], axes[0])

    # Trace plot:
    axes[1].set_ylabel('Parameter value')
    pd_walker.plot(ax=axes[1], marker='.', ls='', ms=1)

    # remove legend
    for ax in axes:
        if ax.legend_ is not None:
            ax.legend_.remove()
 
    # Save figure
    fig.savefig(prefix + 'Bayesian_fit_parameter_{}.png'.format(varname))

def save_fit_summary(results_pd, prefix='', iter_stable = iter_stable_default,
        fitted_func=None, n_walkers=None, n_iterations=None, 
        pickle_data=False):
    """Outputs a summary of the fitting in a text file

    Creates a text file with an executive summary of the results of the Bayesian fit.

    Parameters:
        results_pd (Pandas dataframe): result of Bayesian fitting
        prefix (str): path and prefix to prepend to the generated file
        iter_stable (int): iteration number at which the chain is assumed to have converged. Only data after this step are used to compute statistics. 
        fitted_func (function): function that was fitted, the docstring of which is printed to the file
        n_walkers (int): number of walkers used in the fitting
        n_iterations (int): number of iterations for each walking in the fitting
        pickle_data (bool): if True, saves the full dataframe of results in a python pickle file
    """
    text = 'Bayesian fitting\n\n'
    if fitted_func:
        text += 'Function fitted:\n'
        if fitted_func.__doc__ is not None:
            text += fitted_func.__doc__
            text += '\n'
    if n_walkers:
        text += 'Number of walkers: {}\n'.format(n_walkers)
    if n_iterations:
        text += 'Number of iterations: {}\n'.format(n_iterations)

    # Mean values
    inds_after_transient = results_pd.index.get_level_values('Iteration') > iter_stable
    stable_data = results_pd.loc[inds_after_transient]
    text += '\nStatistics summary of variables\n'
    text += stable_data.describe().to_string()

    with open(prefix + 'Bayesian_fit.dat', 'w') as f:
        f.write(text)

    if pickle_data:
        results_pd.to_pickle(prefix + 'Bayesian_fit_result.pkl')

