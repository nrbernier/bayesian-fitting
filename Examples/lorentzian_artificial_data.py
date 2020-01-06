import numpy as np
import random as r
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import bayesian_fitting.bayesian_fit as bf
import bayesian_fitting.bayesian_plot as bp
import bayesian_fitting.functions_library as fl
import timeit


def fitfunc(f, params):
    f0, a, w, c = params
    return c+a* (w/2.)**2/ ((f-f0)**2 + (w/2.)**2)
fitfunc = fl.lorentzian

# Artificial experimental data with noise
f0_0=1.0; c_0=1.0; a_0=0.2; w_0=0.02
N=251; er=0.1
freq = np.linspace(0.9, 1.1, N)
ampl = fitfunc(freq, (f0_0, a_0, w_0, c_0)) + np.random.normal(0, er, N)


lnprior_parameters = (
        bf.prior_uniform(p0=min(freq), p1=max(freq)),
        bf.prior_exponential(a_0, p0=0),
        bf.prior_exponential(w_0, p0=0),
        bf.prior_exponential(c_0, p0=0),
        bf.prior_exponential(er, p0=0),
        )

init_params = [ 
    lambda: r.gauss(f0_0, np.sqrt(1e-5)), 
    lambda: r.expovariate(1/a_0), 
    lambda: r.expovariate(1/w_0), 
    lambda: r.expovariate(1/c_0),
    lambda: r.expovariate(1/er)
    ]

result = bf.bayesian_fit(fitfunc, freq, ampl,
        lnprior_parameters, init_params,
        varnames = ('f0', 'a', 'w', 'c', 'sigma'),
        n_walkers=20)

print(result.tail())

prefix = 'graphs/lorentzian_artificial_'

bp.plot_walker_statistics(result, n_walkers_autocor=4, prefix=prefix)

bp.corner_plot_distribution(result, prefix)

bp.plot_fitting_uncertainty(result, freq, ampl, fitfunc,
        prefix=prefix, sample_size=2000)

bp.save_fit_summary(result, prefix=prefix,
        fitted_func=fitfunc,
        n_walkers=20, n_iterations=1000, pickle_data=True)

