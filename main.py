import emcee
import numpy as np
from schwimmbad import MPIPool


def main(p0,nwalkers,niter,ndim,lnprob,backend):
    with MPIPool() as pool:
    	if not pool.is_master():
    	pool.wait()
    	sys.exit(0)
    	
        	sampler = emcee.EnsembleSampler(nwalkers,ndim, lnprob,pool=pool,backend=backend)

		print("Running burn-in...")
		p0,_,_ = sampler.run_mcmc(p0, 100,progress=True)
		sampler.reset()

		index = 0
		autocorr = np.empty(niter)
		old_tau = np.inf

        	print("Running production...")

		for sample in sampler.sample(p0,iterations=niter,progress=True):
		    if sampler.iteration % 100:
		        continue
		    tau = sampler.get_autocorr_time(tol =0)
		    autocorr[index] = np.mean(tau)
		    index += 1

		    converged = np.all(tau*100 < sampler.iteration)
		    converged &= np.all( np.abs(old_tau - tau)/tau <  1e-3)
		    if converged:
		        break
		    old_tau = tau

        #pos, prob, state = sampler.run_mcmc(p0, niter,progress=True)
#        print(pos,prob)

    return sample
