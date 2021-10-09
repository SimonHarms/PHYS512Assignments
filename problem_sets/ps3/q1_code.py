import numpy as np
import matplotlib.pyplot as plt

def rk4_step(fun,x,y,h):
    k1 = h*fun(x,y)
    k2 = h*fun(x+h/2,y+k1/2)
    k3 = h*fun(x+h/2,y+k2/2)
    k4 = h*fun(x+h,y+k3)
    dy = (k1+2*k2+2*k3+k4)/6
    return y+dy

def rk4_stepd(fun,x,y,h):
    #y1 error h^5*c
    #y2 error 2*(h/2)^5*c
    #eliminate error -> (16y2 - y1) / 15
    y1 = rk4_step(fun,x,y,h) #three times as many function evaluations
    y2 = rk4_step(fun,x,y,h/2)
    y2 = rk4_step(fun,x + h/2,y2,h/2)
    return (16*y2 - y1) / 15

def f(x,y):
    return y / (1 + x**2)

def real_y(x):
    return np.exp(np.arctan(x) + np.arctan(20))

npt = 201
y = np.zeros(npt)
x = np.linspace(-20,20,npt)
y[0] = 1
for i in range(1,npt):
    h = x[i] - x[i - 1]
    y[i] = rk4_step(f,x[i-1],y[i-1],h)

nptd = 67
yd = np.zeros(nptd)
xd = np.linspace(-20,20,nptd)
yd[0] = 1
for i in range(1,nptd):
    h = xd[i] - xd[i - 1]
    yd[i] = rk4_stepd(f,xd[i-1],yd[i-1],h)


plt.plot(x,np.abs(y - real_y(x)),label = "rk4")
plt.plot(xd,np.abs(yd-real_y(xd)), label = "rk4d")
plt.legend()
plt.savefig("residue comparison.png")
