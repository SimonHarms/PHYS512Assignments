import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def rat_eval(p,q,x):
    top = 0
    for i in range(len(p)):
        top = top+p[i]*x**i
    bot = 1
    for i in range(len(q)):
        bot=bot+q[i]*x**(i+1)
    return top/bot

def compare_interp(func,func_name,xmin,xmax,n,m):

    order = n+m
    x = np.linspace(xmin, xmax, n+m-1)
    y = func(x)
    xx = np.linspace(x[0], x[-1], 2001)

    plt.ion()
    plt.clf()
    plt.plot(xx,func(xx),label = "Original Function")
    plt.savefig(func_name + "_original.png")

    #polynomial
    yy = np.zeros(len(xx))
    for i  in range(len(x)):
        x_use = np.append(x[:i],x[i+1:])
        x0=x[i]
        mynorm=np.prod(x0-x_use)
        p0=1.0
        for xi in x_use:
            p0=p0*(xi-xx)
        p0=p0/mynorm
        yy += p0 * y[i]

    plt.clf()
    plt.plot(xx,yy,label = "Polynomial Interpolation")
    plt.savefig(func_name + "_poly.png")

    #spline
    spln=interpolate.splrep(x,y)
    yy = interpolate.splev(xx,spln)

    plt.clf()
    plt.plot(xx,yy,label = "Spline Interpolation")
    plt.savefig(func_name + "_spline.png")

    #rational
    mat = np.zeros([n+m-1,n+m-1])
    for i in range(n):
        mat[:,i]=x**i
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    pars=np.dot(np.linalg.pinv(mat),y)
    p=pars[:n]
    q=pars[n:]

    yy = rat_eval(p,q,xx)

    plt.clf()
    plt.plot(xx,yy,label = "Rational Interpolation")
    plt.savefig(func_name + "_rational.png")

compare_interp(np.cos,"cos",-np.pi/2,np.pi/2,5,5)
compare_interp(lambda x : 1 / (1 + x*x),"lorentz", -1, 1, 4, 5)
