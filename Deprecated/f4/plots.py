import numpy as np
#from getdist import MCSamples, plots
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import emcee



names = ["$H_0$","$\Omega_m$","$M$"]
base = emcee.backends.HDFBackend("Base/noprior.h5").get_chain(flat=True)
xA = emcee.backends.HDFBackend("Dultzin/noprior.h5").get_chain(flat=True)

c = ChainConsumer()
c.add_chain(base,parameters=names,name="Base")
c.add_chain(xA,parameters=names,name="Base + xA")
c.configure(shade=True,shade_alpha=0.5,colors=["#011638","#23CE6B","#F08080"])
fig = c.plotter.plot(extents=[[62,67],[0.23,0.33],[-19.3,-19.6]])

fig.set_size_inches(5 + fig.get_size_inches())
plt.savefig("nopriors.pdf",dpi=100)
