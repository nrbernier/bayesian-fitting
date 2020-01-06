import numpy as np
import random as r
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import bayesian_fitting.bayesian_fit as bf
import bayesian_fitting.bayesian_plot as bp
import bayesian_fitting.functions_library as fl

lorentzian = fl.lorentzian
double_lorentzian = fl.double_lorentzian
func_fitted = double_lorentzian

# Artificial experimental data with noise
datafile = 'data/double_lorentzian_data'
data = np.loadtxt(datafile)
freqs = data[:,0]
fswpowers = data[:,1]
fswpowers_lin = data[:,2]

fswpowers_fitted = fswpowers_lin/ fswpowers_lin[0]

# initial guesses:
f0_0 = freqs[np.argmax(fswpowers[:len(freqs)//2])]
f0_1 = freqs[np.argmax(fswpowers[len(freqs)//2:]) + len(freqs)//2]
deltaf_0 = f0_1 - f0_0
c_0 = fswpowers_fitted.min()
a_1_0 = fswpowers_fitted.max() - c_0
a_2_0 = a_1_0
w_1_0 = (freqs.max() - freqs.min()) / 5.0
w_2_0 = w_1_0

er = fswpowers_fitted[0:50].std()

varnames = (
        'res 1', 'amp 1', 'width 1', 
        'res2 - res1', 'amp 2', 'width 2', 
        'background',
        '$\sigma$',
        )

lnprior_parameters = (
        bf.prior_uniform(p0=min(freqs), p1=max(freqs)),
        bf.prior_exponential(a_1_0, p0=0),
        bf.prior_exponential(w_1_0, p0=0),
        bf.prior_uniform(p0=0, p1=max(freqs)-min(freqs)),
        bf.prior_exponential(a_2_0, p0=0),
        bf.prior_exponential(w_2_0, p0=0),
        bf.prior_exponential(c_0, p0=0),
        bf.prior_exponential(er, p0=0),
        )

#init_params = [ 
#    lambda: r.gauss(f0_0, w_1_0/10.), 
#    lambda: r.expovariate(1/a_1_0), 
#    lambda: r.expovariate(1/w_1_0), 
#    lambda: r.gauss(f0_1-f0_0, w_1_0/10.), 
#    lambda: r.expovariate(1/a_1_0), 
#    lambda: r.expovariate(1/w_1_0), 
#    lambda: r.expovariate(1/c_0),
#    lambda: r.expovariate(1/er)
#    ]
init_params = [ 
    lambda: r.gauss(f0_0, w_1_0/10.), 
    lambda: r.gauss(a_1_0, a_1_0*0.01), 
    lambda: r.gauss(w_1_0, w_1_0*0.01), 
    lambda: r.gauss(f0_1-f0_0, w_1_0/10.), 
    lambda: r.gauss(a_2_0, a_2_0*0.01), 
    lambda: r.gauss(w_2_0, w_2_0*0.01), 
    lambda: r.gauss(c_0, c_0*0.01),
    lambda: r.gauss(er, er*0.01)
    ]

prefix = 'graphs/double_lorentzian_' 

n_walkers = 30
n_iterations = 5000
n_conv = 2000
result = bf.bayesian_fit(func_fitted, freqs, fswpowers_fitted,
        lnprior_parameters, init_params,
        varnames = varnames,
        n_walkers=n_walkers, n_iterations=n_iterations)

print(result.tail())

bp.plot_walker_statistics(result, prefix=prefix, iter_stable=n_conv)

bp.plot_fitting_uncertainty(result, freqs, fswpowers_fitted, 
        func_fitted,
        prefix=prefix, sample_size=4000, iter_stable=n_conv)

bp.save_fit_summary(result, prefix=prefix,
        fitted_func=func_fitted, iter_stable=n_conv,
        n_walkers=n_walkers, n_iterations=n_iterations, 
        pickle_data=True)

bp.corner_plot_distribution(result, prefix, iter_stable=n_conv)



