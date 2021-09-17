import numpy as np

def lakeshore(V,data):
    x = data[0]
    y = data[1]
    dy = data[2]
    ret = []
    for v in V:
        ind = np.searchsorted(x, v)
        ans = (v - x[ind - 1]) * (y[ind] - y[ind - 1]) / (x[ind] - x[ind - 1])
        err = np.abs(ans - dy[ind - 1] * (v - x[ind - 1]))
        ret.append([ans,err])
    return ret
