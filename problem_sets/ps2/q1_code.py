import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from q2_code import integrate_adaptive

def ring_field(x,z,R,sigma):
    h = z - x
    r = np.sqrt(R*R - x*x)
    permittivity = 1 #setting the permittivity to one for convenience
    return h / (2 * permittivity * np.power(h*h + r*r, 1.5)) * sigma

def shell_field_quad(z,R,sigma):
    ret, err = scipy.integrate.quad(ring_field, -R, R, args=(z,R,sigma))
    return ret

def shell_field_adaptive(z,R,sigma,tol):
    if z == R:   #The adaptive method breaks down at z==R because of a singularity
        return 0 #This code is just here to make a pretty graph in spite of this
    fun = lambda x : ring_field(x,z,R,sigma)
    return integrate_adaptive(fun,-R,R,tol)

R = 5
sigma = 5
z = np.linspace(0,10,2001)
plt.plot(z,[shell_field_quad(zz,R,sigma) for zz in z])
plt.savefig("quad_integral.png")
plt.clf()
plt.plot(z,[shell_field_adaptive(zz,R,sigma,0.0001) for zz in z])
plt.savefig("adaptive_integral.png")
