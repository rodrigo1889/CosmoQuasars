import numpy as np
from multiprocess import Pool
import emcee
from cosmo import f2tp
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

#BAO SDSS DR12
from call_covariance import COV_BAO
bao_matrix = COV_BAO()
icov_bao = np.linalg.inv(bao_matrix)


#QSO
from call_data import QUASARS_FLUX, QUASARS_v2
z_qso, mu_qso, emu_qso = QUASARS_v2()
z_q, fx, fuv, efx, efuv = QUASARS_FLUX()



H0s = np.array([
    [73.3,1.04,"shoes"], #SH0ES
    [74.03,1.42,"gaia"], #GAIA
    [69.8,0.8,"trgb"], #TRGB
    [67.36,0.54,"planck"], #Planck TT+TE+EE+lowE+lensing
    [67.9,1.5,"act"], #ACT
    ])


def lnlikeH0(theta,H0c,sigmaH0):
    H0, m, M, b, beta = theta
    likk = -(H0-H0c)**2/(sigmaH0**2)
    return likk

def lnlike_cc(theta):
    H0, m, M, b, beta = theta
    delta = f2tp.H(z_cc,H0,m,b) - H_cc
    likk = -0.5*np.dot(delta, np.dot(icov_cc,delta))
    return likk

def lnlike_sn(theta):
    H0, m, M, b, beta = theta
    mb_teor = f2tp.distance_modulus(z_p,H0,m,b) + M
    delta = mb_teor - mb_p
    likk = -0.5*np.dot(delta,np.dot(icov_pantheon,delta)) - 0.5*np.log(S/(2*np.pi))
    return likk

def lnlike_6dF(theta):
    H0, m, M, b, beta = theta
    delta = f2tp.rdoverDV(0.106,H0,m,b) - 0.336
    return -0.5*(delta**2/0.015**2) - np.log(0.015**2)

def lnlike_SDSS7(theta):
    H0, m, M, b, beta = theta
    delta = f2tp.DVoverrd(0.15,H0,m,b,r_fid=148.69) - 664
    return -0.5*(delta**2/25**2) - np.log(25**2)

def lnlike_BOSSDR11(theta):
    H0, m, M, b, beta = theta
    delta = f2tp.D_M(2.4,H0,m,b)/f2tp.r_d(H0,m) - 36.6
    return -0.5*(delta**2/1.2**2) - np.log(1.2**2)

def lnlike_eBOSS(theta):
    H0, m, M, b, beta = theta
    delta = (f2tp.D_V(1.52,H0,m,b)*147.78)/f2tp.r_d(H0,m) -3843
    return -0.5*(delta**2/147**2) - np.log(147**2)

def lnlike_SDSSDR12(theta):
    H0, m, M, b, beta = theta
    rdfid = 147.78
    zs = np.array([0.38,0.51,0.61])
    Dmrs = np.array([1512.39,1975.22,2306.64])
    Hrs = np.array([81.2087,90.9029,98.9647])
    err = np.array([2.1,2.2,2.1])
    delta0 = f2tp.Hoverrd(zs,H0,m,b,r_fid = rdfid) - Hrs
    delta1 = f2tp.DMoverrd(zs,H0,m,b,r_fid = rdfid) - Dmrs
    delta = np.vstack((delta1,delta0)).ravel("F")
    #return -0.5*np.sum((delta0**2)/(err**2))
    return -0.5*np.dot(delta, np.dot(icov_bao,delta))

def lnlike_SDSSDR14(theta):
    H0, m, M, b, beta = theta
    zs = np.array([0.978,1.23,1.526,1.944])
    Hrs = np.array([113.72,131.44,148.11,172.63])
    err = np.array([14.63,12.42,12.75,14.79])
    delta0 = f2tp.Hoverrd(zs,H0,m,b,r_fid= 147.78)
    #return -0.5*np.dot(delta, np.dot(icov_bao_2,delta))
    return -0.5*np.sum((delta0**2)/(err**2))

def lnlike_BAO(theta):
    return lnlike_6dF(theta) + lnlike_SDSS7(theta) + lnlike_BOSSDR11(theta) + lnlike_SDSSDR12(theta)


def lnlike_qso(theta):
    H0, m, M, b, beta = theta
    mu_teor = f2tp.distance_modulus(z_qso,H0,m,b)
    delta = mu_teor - mu_qso
    likk = -0.5*np.sum((delta**2/emu_qso**2)+np.log(emu_qso**2))
    return likk


def lnlike_qso_flux(theta):
    H0, m, M, b, beta = theta
    gamma = 0.702
    logdl_teor = np.log10(f2tp.D_L(z_q,H0,m,b))
    Psi = 1/(2*(gamma-1))*(fx-gamma*fuv) - beta/(2*(gamma-1))
    si = fx**2 + 0.21**2
    likk = -0.5*np.sum((fx-Psi)**2/si + np.log(si))
    return likk

def lnlike_light(theta):
    return lnlike_cc(theta)+ lnlike_sn(theta)

def lnlike_base(theta):
    return lnlike_cc(theta)+ lnlike_sn(theta) + lnlike_BAO(theta)

def lnprior(theta):
    H0, m, M, b, beta = theta
    #if 65 < H0 < 80 and 0.1 <= m <= 0.9 and -30.0 < M < 0.0:
    if 50 < H0 < 80 and 0.0 < m <= 0.9 and -30.0 < M < 0.0 and 0 < b <= 1 and -20 < beta < 20:
        return 0.0
    return -np.inf
initial = np.array([70,0.3,0.01,-19.0,3.0])
nwalkers= 20

for i in range(len(H0s)):
    print("Calculating "+H0s[i][-1])
    def lnlike(theta):
        return lnlike_light(theta)+ lnlike_qso_flux(theta) + lnlikeH0(theta,float(H0s[i][0]),float(H0s[i][1]))
    def lnprob(theta):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta)
    p0 = [np.array(initial) + 1e-5 * np.random.randn(len(initial)) for i in range(nwalkers)]
    backend = emcee.backends.HDFBackend("Chains/f2Lusso"+H0s[i][-1]+".h5")
    sampler = main.main(p0,nwalkers,100000,len(initial),lnprob,backend=backend)
