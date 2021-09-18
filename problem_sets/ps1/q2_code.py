import numpy as np
#I decided to use the coefficients for a little bit more accuracy
#err = ef/d + 1/6d^2f'''
#=> d ~= (3e)^(1/3)
def ndiff(fun,x,full=False):
    epsilon = np.power(2.0,-53)
    dx = np.power(3.0 * epsilon, 1.0/3) 
    deriv = (fun(x+dx) - fun(x-dx))/(2*dx)
    error = np.abs(fun(x) * (epsilon / dx + dx * dx / 6)) #estimating f''' to be close to f
    if full:
        return deriv, dx, error
    else:
        return deriv
