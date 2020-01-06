"""Library of functions in the format to use the Bayesian fitting module

The functions should have exactly two arguments, the first is the input to the function to fit and the second is a list of parameters to fit.
"""

def lorentzian(f, params):
    """Lorentzian
    params = f0, a, w, c
    f0: resonance frequency
    a: amplitude of the lorentzian
    w: width
    c: background level
    """
    f0, a, w, c = params
    return c+a* (w/2.)**2/ ((f-f0)**2 + (w/2.)**2)

def double_lorentzian(f, params):
    f0_1, a_1, w_1, deltaf, a_2, w_2, c = params
    lor1 = a_1* (w_1/2.)**2/ ((f-f0_1)**2 + (w_1/2.)**2)
    lor2 = a_2* (w_2/2.)**2/ ((f-(f0_1+deltaf))**2 + (w_2/2.)**2)
    return c + lor1 + lor2


