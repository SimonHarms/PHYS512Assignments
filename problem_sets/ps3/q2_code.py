import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def fun(x,y,half_life=[4.468e9 * 365, 24.10, 6.7 / 24, 245500 * 365, 75380 * 365, 1600 * 365, 3.8235, 3.1 / 24 / 60, 26.8 / 24 / 60, 19.9 / 24 / 60, 164.3 / 24 / 3600 / 1000, 22.3 * 365, 5.015, 138.376]):  #unit is days
    dydx = np.zeros(len(half_life) + 1)
    dydx[0] = -y[0]/half_life[0]
    dydx[-1] = y[-2]/half_life[-1]
    for i in range(len(half_life) - 1):
        dydx[i + 1] = y[i]/half_life[i] - y[i+1]/half_life[i+1]
    return dydx

y0 = np.zeros(15)
y0[0] = 1
x0 = 0
x1 = 365 * 4e9 #4 billion years
ans = integrate.solve_ivp(fun,[x0,x1],y0, method="Radau")
plt.plot(ans.t / 365, ans.y[0], label = "U238")
plt.plot(ans.t / 365, ans.y[14], label = "Pb206")
plt.xlabel("time (yrs)")
plt.ylabel("amount")
plt.legend()
plt.title("U238 vs. Pb206")
plt.savefig("U238 vs. Pb206.png")
plt.clf()
plt.plot(ans.t / 365, ans.y[1], label = "Th230")
plt.plot(ans.t / 365, ans.y[3], label = "U234")
plt.xlabel("time (yrs)")
plt.ylabel("amount")
plt.legend()
plt.title("Th230 vs. U234")
plt.savefig("Th230 vs. U234.png")
