import pandas as pd
import numpy as np

### FULL DATA SETS

def CC():
    data = pd.DataFrame(pd.read_csv("Full_data/cc.csv"))
    z = data["z"].to_numpy(); H = data["H(z)"].to_numpy(); Herr = data["s_H"].to_numpy()
    return z,H,Herr

def PANTHEON():
    data = pd.DataFrame(pd.read_csv("Full_data/Pantheon.txt",header=None,sep=" "))
    data = data.sort_values(by=1)
    zcmb = data[1].to_numpy(); mb = data[4].to_numpy(); emb = data[5].to_numpy()
    return zcmb,mb,emb

def QUASARS():
    data = pd.DataFrame(pd.read_csv("Full_data/Qsort.csv"))
    z = data["z"].to_numpy(); mu = data["mu"].to_numpy(); emu = data["emu"].to_numpy()
    return z,mu,emu

def QUASARS_FLUX():
	data = pd.DataFrame(pd.read_csv("Full_data/Quasars.csv",sep=",")) 
	data = data.sort_values(by="z")
	#data = data[data["z"]>0.7]
	sub = data[data["gammax"]>2.2]
	#sub = data
	#data.columns = ["z","Fx","Fuv","eFx","eFuv"]
	#z = data["z"].to_numpy(); fx = data["Fx"].to_numpy(); fuv = data["Fuv"].to_numpy(); efx = data["eFx"].to_numpy(); efuv = data["eFuv"].to_numpy()
	z = sub["z"].to_numpy(); fx = sub["logFx"].to_numpy();efx = sub["e_logFx"].to_numpy();fuv= sub["logFuv"].to_numpy(); efuv = sub["e_logFuv"].to_numpy(); dl = sub["logdl"].to_numpy(); edl = sub["elogdl"].to_numpy()
	return z, fx, fuv, efx, efuv,dl,edl
	#return z

def QUASARS_v2():
    data = pd.DataFrame(pd.read_csv("Full_data/Marzianiv2.csv"))
    z = data["z"].to_numpy(); mu = data["mu"].to_numpy(); emu = data["emu"].to_numpy()
    return z,mu,emu

def BAO_DM(): #Arxiv 2111.02420
    data = pd.DataFrame(pd.read_csv("Full_data/Baodm.txt",header=None))
    z = data[0].to_numpy(); DM = data[1].to_numpy()
    return z,DM

def BAO_6dF(): # rd/Dv
    return  0.106,0.336, 0.015

def BAO_SDSSDR7(): #Dv*(rd/rdfid)
    return 0.15, 664, 25

def BAO_H():
    data = pd.DataFrame(pd.read_csv("Full_data/BaoH.txt",header=None))
    z = data[0].to_numpy(); H = data[1].to_numpy()
    return z,H

def GRB():
    data = pd.DataFrame(pd.read_csv("Full_data/grb.csv"))
    z = data["z"].to_numpy(); Ep = data["Ep"].to_numpy(); e_Ep = data["e_Ep"].to_numpy(); Eiso = data["E_iso"].to_numpy(); e_Eiso = data["e_Eiso"].to_numpy()
    return z,Ep,e_Ep,Eiso,e_Eiso

def PANTHEON_PLUS():
    data = pd.DataFrame(pd.read_csv("Full_data/Pantheonplus.txt"))
    data = data.sort_values(by="zCMB")
    zcmb = data["zCMB"].to_numpy(); mb = data["m_b_corr"].to_numpy(); emb = data["m_b_corr_err_DIAG"].to_numpy(); is_calibrator = data["IS_CALIBRATOR"].to_numpy(); d_ceph = data["CEPH_DIST"].to_numpy()
    return zcmb,mb, emb, is_calibrator, d_ceph

def VHZ():
    vhz = pd.DataFrame(pd.read_csv("Full_data/VHZ_mock.csv"))
    zvhz = vhz["z1"]; mbvhz = vhz["mb_corr"]; embvhz = vhz["emb"]
    return zvhz, mbvhz, embvhz

def RVM():
    data = pd.DataFrame(pd.read_csv("Full_data/rvmapping.csv"))
    z_rvm = data["z"].to_numpy(); DLrvm = data["D_L"].to_numpy(); e_DLrvm = data["eD_L"].to_numpy()
    return z_rvm, DLrvm, e_DLrvm

def HII():
    data = pd.DataFrame(pd.read_csv("Full_data/hii.csv"))
    data = data.sort_values(by="redshift")
    z = data["redshift"].to_numpy();logsigma=data["log sigma_corr"].to_numpy();err_logsigma = data["err_log sigma_corr"].to_numpy();logfhb = data["log fHb_corr"].to_numpy(); errlogfhb = data["err_log fHb_corr"].to_numpy()
    return z,logsigma,err_logsigma,logfhb,errlogfhb

def QSO_ANG():
    data = pd.DataFrame(pd.read_csv("FUll_data/Qso-angular.csv"))
    z = data["z"].to_numpy(); theta = data["theta"].to_numpy(); s_theta = data["s_theta"].to_numpy()
    return z, theta, s_theta

def fs8_eBOSS():
    data = pd.DataFrame(pd.read_csv("Full_data/rsd_eBoss.csv"))
    data = data.sort_values(by="z")
    z = data["z"].to_numpy(); fs8 = data["fs8"].to_numpy(); efs8 = data["efs8"].to_numpy()
    return z, fs8, efs8

def fs8_Wigg():
    data = pd.DataFrame(pd.read_csv("Full_data/rsd_Wigg.csv"))
    #data = data.sort_values(by="z")
    z = data["z"].to_numpy(); fs8 = data["fs8"].to_numpy(); efs8 = data["efs8"].to_numpy()
    return z, fs8, efs8

def fs8_1():
    data = pd.DataFrame(pd.read_csv("Full_data/rsd.csv"))
    data = data.sort_values(by="z")
    z = data["z"].to_numpy(); fs8 = data["fs8"].to_numpy(); efs8 = data["efs8"].to_numpy()
    return z, fs8, efs8

def fs8_2():
    data = pd.DataFrame(pd.read_csv("Full_data/rsd2.csv"))
    data = data.sort_values(by="z")
    z = data["z"].to_numpy(); fs8 = data["fs8"].to_numpy(); efs8 = data["efs8"].to_numpy()
    return z, fs8, efs8

def fs8_3():
    data = pd.DataFrame(pd.read_csv("Full_data/rsd3.csv"))
    data = data.sort_values(by="z")
    z = data["z"].to_numpy(); fs8 = data["fs8"].to_numpy(); efs8 = data["efs8"].to_numpy()
    return z, fs8, efs8

### BINNED DATA SETS

def PANTHEON_BINNED():
    data = pd.DataFrame(pd.read_csv("Binned_data/Pantheon.txt",header=None,sep=" "))
    data = data.sort_values(by=1)
    zcmb = data[1].to_numpy(); mb = data[4].to_numpy(); emb = data[5].to_numpy()
    return zcmb,mb,emb

def QUASARS_BINNED():
    data = pd.DataFrame(pd.read_csv("Binned_data/Quasars.txt",header=None))
    data = data.transpose()
    z = data[0].to_numpy(); mu = data[1].to_numpy(); emu = data[2].to_numpy()
    return z,mu,emu

def QUASARS_BINNED2():
    data = np.genfromtxt("Binned_data/Quasars_2.txt")
    z = data[0]; dm = data[1]; edm = data[2]
    return z, dm, edm

def QUASARS_BINNED3():
    data = pd.DataFrame(pd.read_csv("Binned_data/Quasars3.csv"))
    #return data["z"].to_numpy(), data["fx"].to_numpy(), data["e_fx"].to_numpy(), data["fuv"].to_numpy(),
    return data["z"].to_numpy(), data["fx"].to_numpy(), data["e_fx"].to_numpy(), data["fuv"].to_numpy(), data["e_fuv"].to_numpy(),data["dl"].to_numpy(),data["err_mean"].to_numpy()
