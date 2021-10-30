import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

#c)

def fsin(k, k0, N):
    J = complex(0,1)
    term1 = (1 - np.exp(J*2*np.pi*(k0 - k))) / (1 - np.exp(J*2*np.pi*(k0 / N - k / N)))
    term2 = (1 - np.exp(J*2*np.pi*(-k0 - k))) / (1 - np.exp(J*2*np.pi*(-k0 / N - k / N)))
    return 1/(2*J) * (term1 - term2)

N = 10000
k0 = 2340.54

k = np.linspace(0,N-1,N) #I'm using the same interval for k and x
F1 = fft(np.sin(2*np.pi*k0/N*k))
F2 = fsin(k, k0, N)
F1 = F1
F2 = F2

plt.plot(k, F1)
plt.title("DFT of sin")
plt.xlabel("k")
plt.savefig('q5_dft.png')

plt.clf()
plt.plot(k, F2)
plt.title("Analytic approximation of fft(sin(wx))")
plt.xlabel("k")
plt.savefig('q5_analytic.png')

plt.clf()
plt.plot(k, np.abs(np.abs(F2) - np.abs(F1)))
plt.title("residuals")
plt.xlabel("k")
plt.savefig("q5_error.png")

#d)

def window(x, N):
    return 0.5 - 0.5*np.cos(2*np.pi*x/N)

F1d = fft(np.sin(2*np.pi*k0/N*k) * window(k, N)) * 1.45

plt.plot(k, F1d)
plt.title("DFT of sin windowed")
plt.xlabel("k")
plt.savefig('q5d_dft.png')
