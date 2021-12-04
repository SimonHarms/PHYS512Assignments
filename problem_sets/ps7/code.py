import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft2, irfft2

def remove_inferred_charge(G):
    p = G - neighbour_average(G)
    p[0,0] -= 1
    dif = irfft2(rfft2(G) * rfft2(p))
    G -= dif
    G += 1 - G[0,0]

def neighbour_average(V):
    ans = 0*V
    ans += np.roll(V,1,axis=0)
    ans += np.roll(V,-1,axis=0)
    ans += np.roll(V,1,axis=1)
    ans += np.roll(V,-1,axis=1)
    return ans / 4
    
n = 200
mid = n // 2
V = np.zeros([n, n])
V[mid, mid] = 1
#expected charge distribution
p = np.zeros([n, n])
p[0,0] = 1

#I set up the potential distribution with the charge in the middle and then shift it to be at [0,0] for convience sake
for i in range(n):
    for j in range(n):
        if not (i == mid and j == mid):
            V[i,j] = -np.log(np.sqrt((i-mid)**2 + (j-mid)**2)) / (2 * np.log(2))


G = np.roll(np.roll(V,-mid,axis = 0),-mid,axis = 1)

#there's a charge of 1 spread out accros the distribution that doesn't go away no matter how many iterations I do so I just subtract it out when I do my analysis
print("Initially the inferred charge is: ",np.sum(np.abs(G - neighbour_average(G) - p + 1/G.size)) - 1)

for i in range(5):
    remove_inferred_charge(G)
    print("After", i + 1, "iterations the inferred charge is: ",np.sum(np.abs(G - neighbour_average(G) - p + 1/G.size)))

print("V[1,0] =", G[1,0])
print("V[2,0] =", G[2,0])
print("V[5,0] =", G[5,0])

def convolute(x,mask):
    tmp = 0*x
    tmp[mask] = x[mask]
    ret = 0*x
    ret[mask] = irfft2(rfft2(G) * rfft2(tmp))[mask]
    return ret

def conjgrad(x,b,mask,niter=20,fun=convolute):
    #conjugate gradient method from the notes
    r=b-fun(x,mask)
    p=r
    rr=np.sum(r*r)
    for iter in range(niter):
        Ap=fun(p,mask)
        pAp=np.sum(p*Ap)
        alpha=rr/pAp
        x=x+alpha*p
        r=r-alpha*Ap
        rr_new=np.sum(r*r)
        beta=rr_new/rr
        p=r+beta*p
        rr=rr_new
    return x

V = np.zeros([200,200])
mask = np.zeros([200,200],dtype='bool')
mask[50:150,50:150] = True
V[mask] = 1
p = conjgrad(0*V, V, mask, 200)
p_side = p[50,50:150]
plt.plot(np.linspace(50,149,100),p_side)
plt.title("Charge Distribution on the Top Side of the Box")
plt.xlabel("position")
plt.ylabel("charge density")
plt.savefig("charge_dist_topbox.png")

V_complete = convolute(p,np.ones([200,200],dtype='bool'))
plt.clf()
plt.imshow(V_complete)
plt.title("Potential Distribution Inside and Outside the Box")
plt.colorbar()
plt.savefig("potential.png")

E = np.gradient(V_complete)
E_side_y = E[0][50,:]
E_side_x = E[1][50,:]
plt.clf()
plt.plot(E_side_x, label = "x component")
plt.plot(E_side_y, label = "y component")
plt.legend()
plt.xlabel("position")
plt.ylabel("field strength")
plt.title("X and Y Components of E Field on Top Side of the Box")
plt.savefig("E_field_top_side.png")

plt.clf()
plt.imshow(E[0])
plt.title("Y Component of E Field Everywhere")
plt.colorbar()
plt.savefig("E_field_y.png")
plt.clf()
plt.imshow(E[1])
plt.colorbar()
plt.title("X Component of E Field Everywhere")
plt.savefig("E_field_x.png")
