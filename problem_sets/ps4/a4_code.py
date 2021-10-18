import camb
import numpy as np

def get_spectrum(pars,lmax=3000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:]

def chi_square(pars, spec, errs):
    model = get_spectrum(pars)
    model = model[:len(spec)]
    resid = spec - model
    chisq=np.sum((resid/errs)**2)
    return chisq
                  
pars = np.asarray([69, 0.022, 0.12, 0.06, 2.1e-9, 0.95])
planck = np.loadtxt('mcmc/COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
spec = planck[:,1]
errs = 0.5*(planck[:,2] + planck[:,3])
print("chisq is ",chi_square(pars, spec, errs)," for ",len(spec)-len(pars)," degrees of freedom.")

#Q2
def numerical_grad(func, pars, cutoff):
    #get a matrix of derivatives with respect to different parameters
    f0 = func(pars)[:cutoff]
    grad = np.zeros([cutoff, len(pars)])
    eps = 2**-52
    h = eps**(1/2)
    for i in range(len(pars)):
        temp_pars = pars.copy()
        temp_pars[i] += h
        grad[:,i] = (func(temp_pars)[:cutoff] - f0) / h
    return grad, f0

def newton(pars, stop_condition, N_inv, x):
    old_chisq = 1000000.0
    chisq = 0.0
    curve = np.zeros([len(spec), len(pars)])
    while np.abs(chisq - old_chisq) >= stop_condition:
        grad, pred = numerical_grad(get_spectrum, pars, len(spec))
        grad = np.matrix(grad)
        
        r = x - pred
        r = np.matrix(r).transpose()

        lhs = grad.transpose()*N_inv*grad
        rhs = grad.transpose()*N_inv*r
        dp = np.linalg.inv(lhs)*(rhs)

        for i in range(len(pars)):
            pars[i] = pars[i] + dp[i][0,0]

        old_chisq = chisq
        chisq = np.sum(r.transpose() * N_inv * r)
        
        if np.abs(chisq - old_chisq) < stop_condition:
            curve = dp
    return pars, np.asarray(curve.transpose())[0]

N = np.diag(0.5*(planck[:,2] + planck[:,3]))
N = N**2
q2pars, curve = newton([69, 0.022, 0.12, 0.06, 2.1e-9, 0.95], 0.01, np.linalg.pinv(N), spec)
print("Newton's Method: ", q2pars, "X2 = ", chi_square(q2pars, spec, errs))
np.savetxt("planck_fit_params.txt", np.vstack((q2pars, np.abs(curve))))
    
#Q3
def prior_chisq(pars,par_priors,par_errs):
    if par_priors is None:
        return 0
    par_shifts = pars-par_priors
    return np.sum((par_shifts/par_errs)**2)

def mcmc(nstep, start_pars, step_size, spec, noise, expected_pars = None, err_pars = None):
    nparam = len(start_pars)
    params = np.zeros([nstep, nparam])
    params[0,:] = start_pars
    pchisq = np.zeros([nstep, 1])
    cur_chisq = chi_square(start_pars, spec, noise)
    cur_pars = np.array(start_pars.copy())
    pchisq[0,0] = cur_chisq
    for i in range(1,nstep):
        new_pars = cur_pars + step_size * np.random.randn(nparam)
        new_chisq = chi_square(new_pars, spec, noise)
        if new_chisq<cur_chisq:
            accept=True
        else:
            delt = new_chisq-cur_chisq
            prob=np.exp(-0.5*delt)
            if np.random.rand()<prob:
                accept=True
            else:
                accept=False
        if accept:
            cur_pars = new_pars
            cur_chisq = new_chisq
        params[i,:]=cur_pars
        pchisq[i,0]=cur_chisq
    return params, pchisq

chain, chchisq = mcmc(10000,q2pars,curve,spec,errs)
np.savetxt("planck_chain.txt", np.hstack((chchisq, chain)))
q3pars = np.mean(chain, axis=0)
print("MCMC: ", q3pars, "X2 = ", chi_square(q3pars,spec,errs))
#dark energy = 1 - 1/(p[0]/100)**2 * (p[1] + p[2])
#err_dark energy = (1 - dark_energy) * sqrt((2 * err_p[0] / 100 / (p[0] / 100))**2 + ((err_p[1])**2 + (err_p[2])**2)/(p[1] + p[2]))
std = np.std(chain, axis = 0)
dark_energy = 1 - 1/((q3pars[0] / 100)**2) * (q3pars[1] + q3pars[2])
err_dark_energy = (1 - dark_energy) * np.sqrt((2 * std[0] / q3pars[0])**2 + (std[1]**2 + std[2]**2)/(q3pars[1] + q3pars[2])**2)
print("The value for dark energy is: ", dark_energy, "with error: ", err_dark_energy)

#Q4
#implement importance sampling like we did in class
start_pars = q3pars
expected_pars = start_pars*0
expected_pars[3] = 0.054
err_pars = start_pars*0 + np.Infinity
err_pars[3] = 0.0074

nsamp = chain.shape[0]
weight = np.zeros(nsamp)
for i in range(nsamp):
    chisq = prior_chisq(chain[i,:],expected_pars,err_pars)
    weight[i] = np.exp(-0.5*chisq)
is_mean = np.average(chain, axis=0, weights=weight)
is_std = np.sqrt(np.average([[np.abs(chain[i,j] - is_mean[j])**2 for j in range(len(start_pars))] for i in range(nsamp)], axis = 0, weights=weight))
step_size = is_std
chain2, chchisq2 = mcmc(10000,start_pars,step_size,spec,errs,expected_pars,err_pars)
np.savetxt("planck_chain_tauprior.txt", np.hstack((chchisq2, chain2)))
print("MCMC with fixed tau: ", np.mean(chain2,axis=0), "X2 = ", chi_square(np.mean(chain2,axis=0),spec,errs))
print("Importance Sampled: ", is_mean, "X2 = ", chi_square(is_mean, spec, errs))
