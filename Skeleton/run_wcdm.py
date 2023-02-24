import numpy as np
from multiprocess import Pool
import emcee
from cosmo import wcdm
import main


#Calling data only one time to make the code more efficient

#CC
from call_data import CC
from call_covariance import COV_CC
z_cc, H_cc,eH_cc = CC()
matrix_cc = COV_CC()
icov_cc = np.linalg.inv(matrix_cc)


#Supernovae
from call_data import PANTHEON, PANTHEON_BINNED
from call_covariance import COV_PANTHEON, COV_PANTHEON_BINNED
z_p, mb_p, emb_p = PANTHEON()
matrix_pantheon = COV_PANTHEON()
matrix = matrix_pantheon + np.diag(emb_p**2)
S = np.trace(matrix)
icov_pantheon = np.linalg.inv(matrix)

#BAO 6dFGS r_over DV
from call_data import BAO_6dF
z_rdv, rdv, erdv = BAO_6dF()
#BAO SDSS DR7
from call_data import BAO_SDSSDR7
z_dv, dv, edv = BAO_SDSSDR7()
r_fiddr7 = 148.69 #rd fiducial for this measurement
#BAO SDSS DR12
from call_data import BAO_H, BAO_DM
from call_covariance import COV_BAO
z_dm, dm = BAO_DM()
z_h, h = BAO_H()
bao_matrix = COV_BAO()
icov_bao = np.linalg.inv(bao_matrix)
rfiddr12 = 147.78 #rd fiducial for this measurement

#QSO
from call_data import QUASARS_FLUX, QUASARS_v2
z_qso, mu_qso, emu_qso = QUASARS_v2()
z_q, fx, fuv, efx, efuv = QUASARS_FLUX()

priors = np.array([[73.01,-19.260,"shoes"],
                   [74.03,-19.231,"gaia" ],
                   [74.5,-19.216,"trgb"],
                   [67.27,-19.437,"planck"],
                   [73.3,-19.251,"holicow"],
                   [70,-19.351, "wmap"],
                   [68.8,-19.388, "spt"],
                   [67.9,-19.417, "act"],
                   [67,-19.446,"gw"]])

def lnlike_cc(theta):
    m, w, beta, gamma = theta
    delta = wcdm.H(z_cc,H0,m,w) - H_cc
    likk = -0.5*np.dot(delta, np.dot(icov_cc,delta))
    return likk

def lnlike_sn(theta):
    m, w, beta, gamma = theta
    mb_teor = wcdm.distance_modulus(z_p,H0,m,w) + M
    delta = mb_teor - mb_p
    likk = -0.5*np.dot(delta,np.dot(icov_pantheon,delta)) - 0.5*np.log(S/(2*np.pi))
    return likk

def lnlike_6dF(theta):
    m, w, beta, gamma = theta
    delta = wcdm.rdoverDV(z_rdv,H0,m,w) - rdv
    return -0.5*np.sum(delta**2/erdv**2) + np.log(erdv**2)
def lnlike_SDSS7(theta):
    m, w, beta, gamma = theta
    delta = wcdm.DVoverrd(z_dv,H0,m,w,r_fid=r_fiddr7) - dv
    return -0.5*np.sum(delta**2/edv**2) + np.log(edv**2)
def lnlike_BAO2(theta):
    m, w, beta, gamma = theta
    delta = np.array([wcdm.DVoverrd(z_dv,H0,m,w,r_fid=r_fiddr7) - dv,wcdm.rdoverDV(z_rdv,H0,m,w) - rdv])
    err = np.array([edv,erdv])
    return -0.5*np.sum((delta**2/edv**2) + np.log(edv**2))
def lnlike_BAO1(theta):
    m, w, beta, gamma = theta
    deltadm = wcdm.DMoverrd(z_dm,H0,m,w,r_fid=rfiddr12) - dm
    deltah = wcdm.Hoverrd(z_h,H0,m,w,r_fid=rfiddr12) - h
    delta = np.array([deltadm[0],deltah[0],deltadm[1],deltah[1],deltadm[2],deltah[2]])
    return -0.5*np.dot(delta, np.dot(icov_bao,delta))
def lnlike_BAO(theta):
    return lnlike_BAO1(theta) + lnlike_BAO2(theta)
def lnlike_qso(theta):
    m, w, beta, gamma = theta
    mu_teor = wcdm.distance_modulus(z_qso,H0,m,w)
    delta = mu_teor - mu_qso
    likk = -0.5*np.sum((delta**2/emu_qso**2)+np.log(emu_qso**2))
    return likk

def lnlike_qso_flux(theta): #Lusso prior or nUVX sample
    m, w, beta, gamma = theta
    logdl_teor = np.log10(wcdm.D_L(z_q,H0,m,w))
    Psi = beta + gamma*(fuv+27.5)+2*(gamma-1)*(logdl_teor - 28.5)
    si = efuv**2 + gamma**2*efx + np.exp(2*np.log(0.21))
    likk = -0.5*np.sum((fx-Psi)**2/si**2 - np.log(si**2))
    return likk


def lnlike_base(theta):
    return lnlike_cc(theta) + lnlike_sn(theta) + lnlike_BAO(theta)


#FINAL LNLIKE
def lnlike(theta):
    return lnlike_base(theta) + lnlike_qso_flux(theta) #Base + nUVX

def lnprior(theta):
    m, w, beta, gamma = theta
    if 0.1 <= m <= 0.9 and -2.0 <= w <= 0.0 and 0.1 <= beta <= 2.5 and 0.0 < gamma <= 1.0:
        return 0.0
    return -np.inf
def lnprob(theta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta)
initial = np.array([0.25,-1.0,1.0,0.5])
nwalkers= 10
for i in range(len(priors)):
    print("Calculating "+priors[i][-1])
    H0 = float(priors[i][0]); M = float(priors[i][1])
    p0 = [np.array(initial) + 1e-5 * np.random.randn(len(initial)) for i in range(nwalkers)]
    backend = emcee.backends.HDFBackend("Chains/Final/wCDM/Lusso/"+priors[i][-1]+".h5")
    sampler, pos, prob, state = main.main(p0,nwalkers,100000,len(initial),lnprob,backend=backend) #The idial iteration number should be round 100,000
