import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def window(x, N):
    return 0.5 - 0.5*np.cos(2*np.pi*x/N)

def inv_square(x, P0):
    return P0 / x**2

N = 1000
x = np.linspace(0,N-1,N)
y = np.cumsum(np.random.randn(N))
yft = fft(y)
yft_smooth = 0.5 * yft - 0.25*np.roll(yft,1) - 0.25*np.roll(yft,-1)
yft_normed = yft_smooth / (np.mean(window(x, N)) ** 2)
P = np.abs(yft_smooth)**2

n = N//2

plt.plot(x[0:n],P[0:n])
plt.xlabel("k")
plt.title("Random Walk Power Spectrum")
plt.savefig("rw_ps.png")

popt, pcov = curve_fit(inv_square, x[1:n], P[1:n])
P0 = popt[0]
err = np.sqrt(pcov[0,0])
plt.clf()
plt.plot(np.log(x[1:n]),np.log(P[1:n]),'.')
plt.plot(np.log(x[1:n]),np.log(inv_square(x[1:n],P0)), label = "error = " + str(err))
plt.legend()
plt.xlabel("log(k)")
plt.title("Random Walk Power Spectrum Log-Log with regression")
plt.savefig("rw_ps_loglog.png")
