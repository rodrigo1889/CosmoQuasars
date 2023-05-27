import numpy as np
#from getdist import MCSamples, plots
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import emcee

names = ["$\Omega_m$","$w$","$n_1$","$n_2$"]
#Shoes
prior = input("Prior:")
base = emcee.backends.HDFBackend("Base/"+prior+".h5").get_chain(flat=True)
dultzin = emcee.backends.HDFBackend("Dultzin/"+prior+".h5").get_chain(flat=True)
lusso = emcee.backends.HDFBackend("Lusso/"+prior+".h5").get_chain(flat=True)
j = ChainConsumer()
j.add_chain(base,parameters=names[:2],name="Base")
j.add_chain(dultzin,parameters=names[:2],name="Base+xA")
j.add_chain(lusso,parameters=names,name="Base+nUVX")
j.configure(shade=True,shade_alpha=0.5,colors=["#011638","#23CE6B","#F08080","#1B9AAA","#F7D488","#136F63"])
#fig3 = j.plotter.plot(extents=[[0.25, 0.4], [-3.0,0.0]])
fig3 = j.plotter.plot(parameters=names[:2],extents=[[0.2,0.5],[-1.5,-0.9]])
fig3.set_size_inches(5 + fig3.get_size_inches())
#plt.title("Base: CC+Pantheon+BAO")
fig3.savefig(prior+".pdf",dpi=100)
