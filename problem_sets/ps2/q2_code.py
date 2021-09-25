import numpy as np

def integrate_adaptive(fun,a,b,tol,extra=None):
    m = (a + b) / 2
    if extra == None:
        extra = {a : fun(a), b : fun(b), m : fun(m)}
    int1 = (b-a)/6 * (extra[a] + 4*extra[m] + extra[b])
    extra[(a + m) / 2] = fun((a + m) / 2)
    extra[(m + b) / 2] = fun((m + b) / 2)
    int2 = (b-a)/12 * (extra[a] + 4*extra[(a+m)/2] + 2*extra[m] + 4*extra[(m+b)/2] + extra[b])
    if np.abs(int2 - int1) < tol:
        return int2
    else:
        return integrate_adaptive(fun,a,m,tol/2,extra) + integrate_adaptive(fun,m,b,tol/2,extra)
