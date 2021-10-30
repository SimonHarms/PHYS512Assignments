import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

#q1
def conv(f,g):
    ft1 = fft(f)
    ft2 = fft(g)
    return np.real(ifft(ft1*ft2))

def shift(arr, n):
    delt = np.zeros(len(arr))
    delt[n] = 1
    return conv(arr, delt)

def gauss(x, mean, sig):
    return 1 / (sig * (2*np.pi)**0.5) * np.exp(-(x-mean)**2/(2*sig**2))
    
x = np.linspace(0,1000,1001)
y = gauss(x, 500, 100)
plt.plot(x,y, label = "normal gaussian")
plt.plot(x,shift(y,500), label = "shifted gaussian")
plt.title("gauss_shift")
plt.xlabel("k")
plt.legend()
plt.savefig("gauss_shift.png")

#q2
def correlation(arr1, arr2):
    return ifft(fft(arr1) * np.conj(fft(arr2)))

plt.clf()
plt.plot(x, correlation(y,y))
plt.title("gauss_correlation")
plt.xlabel("k")
plt.savefig("gauss_correlation.png")

#q3
plt.clf()
plt.plot(x, correlation(y,shift(y,100)))
plt.title("gauss_correlation_shift100")
plt.xlabel("k")
plt.savefig("gauss_correlation_shift100.png")

plt.clf()
plt.plot(x, correlation(y,shift(y,200)))
plt.title("gauss_correlation_shift200")
plt.xlabel("k")
plt.savefig("gauss_correlation_shift200.png")

#q4
def conv_safe(f,g):
    if len(f) < len(g):
        ft1 = fft(f.append(np.zeros(len(g)-len(f))))
        ft2 = fft(g)
    else:
        ft1 = fft(f)
        ft2 = fft(g.append(np.zeros(len(f)-len(g))))
    return np.real(ifft(ft1*ft2))


