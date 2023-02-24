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

def COV_PANTHEON_PLUS():
    data = np.genfromtxt("Covariance/pantheon_plus.txt")
    mat = data.reshape((1701,1701))
    return mat

#BINNED DATA
def COV_PANTHEON_BINNED():
    data = np.genfromtxt("Covariance/pantheon_binned.txt")
    matrix = data[1:].reshape((40,40))
    return matrix
