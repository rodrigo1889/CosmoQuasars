import numpy as np 
import pandas as pd


df = pd.DataFrame(pd.read_csv("Marzianiv2.csv"))
np.savetxt("qso_full_long.txt",df.values,delimiter=" ")

tabla = np.genfromtxt("qso_full_long.txt")
print(tabla)
