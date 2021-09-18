import numpy as np
from scipy import interpolate

def lakeshore(V,data):
    if len(data) > 3: #transpose the data if it's the wrong way around
        data = np.transpose(data)
    x = data[0]
    y = data[1]
    deriv = data[2] * 1e-3
    spln = interpolate.splrep(x,y)
    temp = interpolate.splev(V,spln)
    
    if hasattr(V, '__len__'): #thing to detect if V is a list or a number
        err = []
        for i in range(len(V)):
            v = V[i]
            j = np.searchsorted(x, v, 'right')
            #extrapolate from the derivative of the point immediately before
            #to roughly estimate how big the error is between the spline
            #interpolation and the true answer
            err.append(np.abs((y[j-1] + deriv[j - 1] * (v - x[j - 1])) - temp[i]))
    else:
        j = np.searchsorted(x, V, 'right')
        err = np.abs((y[j-1] + deriv[j - 1] * (V - x[j - 1])) - temp)
    
        
    return temp, err
