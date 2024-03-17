import pandas as pd
import numpy as np

def COV_CC():
    data = pd.read_csv("Covariance/matrix.csv",header=None)
    data = data.to_numpy()
    return data

def COV_PANTHEON():
    data = np.genfromtxt("Covariance/pantheon.txt")
    matrix = data[1:].reshape((1048,1048))
    return matrix

def COV_BAO():
    data = np.genfromtxt("Covariance/bao.txt")
    return data

def COV_BAO2():
    data = np.genfromtxt("Covariance/M2.txt")
    return data

def COV_PANTHEON_PLUS():
    data = np.genfromtxt("Covariance/STAS+SYS.txt")
    mat = data[1:].reshape((1701,1701))
    return mat

def COV_CMB_DISTANCE():
    data = pd.DataFrame(pd.read_csv("Covariance/cmb_distance.csv"))
    return data.to_numpy()

def COV_S8():
    data = pd.DataFrame(pd.read_csv("Covariance/rsd.txt",header=None,sep=" "))
    return data.to_numpy()

#BINNED DATA
def COV_PANTHEON_BINNED():
    data = np.genfromtxt("Covariance/pantheon_binned.txt")
    matrix = data[1:].reshape((40,40))
    return matrix
