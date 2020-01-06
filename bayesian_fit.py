"""Fitting functions using Bayesian analysis

This module gives tools that use the mystical powers of Bayesian analysis in order to fit any function. There are a few advantages compared to traditional least-square-minimization, including:
    - less likely to get stuck in local maxima of likelihood
    - can converge even with a large number of parameters
    - gives the full probability distribution function of the fitting parameters
    - it is well anchored in probability theory

The nitty-gritty implementation, of a Monte Carlo Markov Chain (MCMC) comes from the module emcee. Here a interface is built on top of it which aims to make function fitting as close as possible in ease of use to least-quare minimization.

The specific fitting models the measured data as
y = f(x, p) + N
with N random noise of constant std sigma. The parameters of the function p as well as the size of the noise sigma are fitted.
"""

import numpy as np
import emcee
import pandas as pd


def bayesian_fit(func, data_x, data_y,
        lnprior_parameters, init_params,
        n_walkers=10, n_iterations=1000, varnames = None):
    """Main function to perform the fit

    Takes the function and data to fit as an argument and gives back a dataframe with the sampling of the fitting parameters. Except for an initial transitory phase where the algorithm converges to a steady state (the length of which should be checked, but should be no more than a few hundred steps), this sampling gives the probability distribution of the fitting parameters given the data and priors.

    Parameters:
        func (function): function to be fitted that has exactly two arguments, input x and a list of parameters. See functions_library for examples.
        data_x (num array): data array of input to the function
        data_y (num array): data array of the outputs of the function to data_x
        lnprior_parameters (list of functions): list of the prior functions corresponding to the list of parameters to be fitted. Those represent initial probability distributions (on a logarithmic scale, with -np.inf for zero probability) of each parameter. In practice, for any data that is not mostly noise, this should not influence much the outcome (but ideally that fact should be checked). It should give a vague idea of the expected scale of the parameter or of a possible range (for instance maybe the parameter must be positive to make sense). See the example of priors defined below.
        init_params (list of functions): functions used to generate the initial parameters for each Markov chain. Usually they are chosen randomly around a value that roughly makes sense for the data.
        n_walkers (int>0): number of Markov chains to use
        n_iterations (int>0): number of iterations for each chain
        varnames (list of str): names of the parameters to fit

    Returns:
       Pandas dataframe: dataframe containing the values of all the parameters (each represented by a data column) for each walker and each chain (represented with a multiindindex, see https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)
    """

    if len(lnprior_parameters) != len(init_params):
        raise Exception('The length of lnprior_parameters and init_params do not match!') 
    if varnames:
        if len(varnames) != len(init_params):
            raise Exception('The length of varnames does not match the length of init_params!')

    def lnprior(theta):
        """Prior function in the Bayesian inference, in log scale
        """
        sigma = theta[-1]
        lnprior_sum = lnprior_parameters[-1](sigma)
        for p, lnprior_p in zip(theta[:-1], lnprior_parameters[:-1]):
            lnprior_sum += lnprior_p(p)
        return lnprior_sum

    def lnlike(theta, x, y):
        """Likelihood function in the Bayesian inference, in log scale
        """
        sigma = theta[-1]
        ymod = func(x, theta[:-1])
        #return -0.5 * np.sum( ((y-ymod)/sigma)**2 + 2*np.log(sigma) )
        return -0.5 * np.sum( 
                (((y.real - ymod.real)**2 + (y.imag-ymod.imag)**2)/sigma**2) 
                + 2*np.log(sigma) 
                )

    def lnprob(theta, x, y):
        """Unnormalized Bayesian probability in log scale
        """
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lnprior(theta) + lnlike(theta, x, y)

    params_0 = [
            np.array([ip() for ip in init_params])
            for i in range(n_walkers)]

    ndim = len(init_params)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, lnprob, 
            args=(data_x, data_y))
    sampler.run_mcmc(params_0, n_iterations)

    if varnames is None:
        varnames = [ 'p{}'.format(i) for i in range(ndim-1) ]
        varnames.append('sigma')
    varnames = list(varnames)

    iterations = range(n_iterations)
    walkers = range(n_walkers)
    index = pd.MultiIndex.from_product([walkers, iterations], 
            names=('Walker', 'Iteration'))
    samples_df = pd.DataFrame(
            sampler.chain.reshape((n_walkers*n_iterations, len(varnames))),
                index=index, columns=varnames)

    return samples_df

def prior_exponential(pexp, p0=0, p1=np.inf):
    """Prior function with exponential distribution (log scale)

    Useful to define a characteristic scale for the parameter to be on the order of pexp

    Parameters:
        pexp (number): scale of the distribution
        p0 (number): hard minimal value allowed for parameter 
        p1 (number): hard maximal value allowed for parameter

    Returns:
        function: prior function on log scale
    """
    def prior(p):
        if p < p0 or p > p1:
            return -np.inf
        else:
            return -p/pexp
    return prior

def prior_uniform(p0=-np.inf, p1=np.inf):
    """Prior function with uniform distribution (log scale)

    When a prior scale for the parameter is not known but a range of possible values is. For instance the resonance frequency that should be within an interval.
    
    Parameters:
        p0 (number): hard minimal value allowed for parameter 
        p1 (number): hard maximal value allowed for parameter

    Returns:
        function: prior function on log scale
    """
    def prior(p):
        if p < p0 or p > p1:
            return -np.inf
        else:
            return 0
    return prior

