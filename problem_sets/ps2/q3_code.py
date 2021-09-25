import numpy as np
import matplotlib.pyplot as plt

def small_log2(x):
    xx = np.linspace(0.5,1,101)
    yy = np.log2(xx)

    xx = xx * 4 - 3
    param = np.polynomial.chebyshev.chebfit(xx,yy,15)

    return np.polynomial.chebyshev.chebval(x * 4 - 3, param[:8]) #I found that 8 coefficients was about enough through testing

def mylog2(x):
    mant, exp = np.frexp(x)
    #log2(m * 2^e) = log2(m) + e
    return small_log2(mant) + exp #the mantissa is never smaller than 0.5 because we would just decrease the exponent in that case
