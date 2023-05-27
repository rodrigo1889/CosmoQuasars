import numpy as np
#from getdist import MCSamples, plots
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import emcee



names = ["$\Omega_m$","$b_1$"]
names2 = ["$H_0$","$\Omega_m$","$b_1$"]
shoes = emcee.backends.HDFBackend("Base/shoes.h5").get_chain(flat=True)
planck = emcee.backends.HDFBackend("Base/planck.h5").get_chain(flat=True)
holicow = emcee.backends.HDFBackend("Base/holicow.h5").get_chain(flat=True)
gaia = emcee.backends.HDFBackend("Base/gaia.h5").get_chain(flat=True)
trgb = emcee.backends.HDFBackend("Base/tgrb.h5").get_chain(flat=True)
act = emcee.backends.HDFBackend("Base/act.h5").get_chain(flat=True)
wmap = emcee.backends.HDFBackend("Base/wmap.h5").get_chain(flat=True)
spt = emcee.backends.HDFBackend("Base/spt.h5").get_chain(flat=True)
gw = emcee.backends.HDFBackend("Base/gw.h5").get_chain(flat=True)
lista = [shoes,planck,holicow,gaia,trgb,act,wmap,spt,gw]
nombres = ["shoes","planck","holicow","gaia","trgb","act","wmap","spt","gw"]
#for i in range(len(lista)):
#    y = ChainConsumer()
#    y.add_chain(lista[i],parameters=names,name=str(i))
#    fig = y.plotter.plot()
#    fig.set_size_inches(5 + fig.get_size_inches())
#    fig.savefig(nombres[i]+".pdf",dpi=100)

#def configure():
#    for i in range(len(lista)):
#        h = ChainConsumer()
#        h.add_chain(lista[i],parameters=names,name=str(i))
#        return h
#h =  configure()
#h.configure(shade=True,shade_alpha=0.2)
#fig2 = h.plotter.plot()
#fig2.set_size_inches(5 + fig2.get_size_inches())
#fig2.savefig("joint.pdf",dpi=100)

j = ChainConsumer()
j.add_chain(shoes,parameters=names,name="S$H0$ES")
j.add_chain(planck,parameters=names,name="Planck")
j.add_chain(holicow,parameters=names,name="$H_0$LiCow")
j.add_chain(gaia,parameters=names,name="GAIA")
j.add_chain(trgb,parameters=names,name="TRGB")
j.add_chain(act,parameters=names,name="ACT")
j.add_chain(wmap,parameters=names,name="WMAP")
j.add_chain(spt,parameters=names,name="SPT")
j.add_chain(gw,parameters=names,name="GW")
j.configure(shade=True,shade_alpha=0.2)
#fig3 = j.plotter.plot(extents=[[0.25, 0.4], [-3.0,0.0]])
fig3 = j.plotter.plot()
fig3.set_size_inches(5 + fig3.get_size_inches())
plt.title("Base: CC+Pantheon+BAO")
fig3.savefig("final.pdf",dpi=100)
