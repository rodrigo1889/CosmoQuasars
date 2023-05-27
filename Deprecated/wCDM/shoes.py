import numpy as np
#from getdist import MCSamples, plots
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import emcee

names = ["$\Omega_m$","$w$"]
#Shoes
prior = input("Prior:")
base = emcee.backends.HDFBackend("Base/"+prior+".h5").get_chain(flat=True)
dultzin = emcee.backends.HDFBackend("Dultzin/"+prior+".h5").get_chain(flat=True)
lusso = emcee.backends.HDFBackend("Lusso/"+prior+".h5").get_chain(flat=True)
j = ChainConsumer()
j.add_chain(base,parameters=names,name="Base")
j.add_chain(dultzin,parameters=names,name="Base+Dultzin")
j.add_chain(lusso,parameters=names,name="Base+Lusso")
j.configure(shade=True,shade_alpha=0.3)
#fig3 = j.plotter.plot(extents=[[0.25, 0.4], [-3.0,0.0]])
fig3 = j.plotter.plot()
fig3.set_size_inches(5 + fig3.get_size_inches())
#plt.title("Base: CC+Pantheon+BAO")
fig3.savefig(prior+".pdf",dpi=100)
