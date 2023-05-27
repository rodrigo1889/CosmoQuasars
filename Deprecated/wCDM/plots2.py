import emcee
import matplotlib.pyplot as plt
import numpy as np
from chainconsumer import ChainConsumer


sample = "Lusso/"
shoes = emcee.backends.HDFBackend(sample+"shoes.h5").get_chain(flat=True) #Previous shoes version
planck = emcee.backends.HDFBackend(sample+"planck.h5").get_chain(flat=True)
holicow = emcee.backends.HDFBackend(sample+"holicow.h5").get_chain(flat=True)
gaia = emcee.backends.HDFBackend(sample+"gaia.h5").get_chain(flat=True)
trgb = emcee.backends.HDFBackend(sample+"trgb.h5").get_chain(flat=True)
act = emcee.backends.HDFBackend(sample+"act.h5").get_chain(flat=True)
wmap = emcee.backends.HDFBackend(sample+"wmap.h5").get_chain(flat=True)
spt = emcee.backends.HDFBackend(sample+"spt.h5").get_chain(flat=True)
gw = emcee.backends.HDFBackend(sample+"gw.h5").get_chain(flat=True)




parameters = ["\Omega_m","w","\\alpha","\\beta"]
lista = [shoes,planck,holicow,gaia,trgb,act,wmap,spt,gw]
nombres = ["S$H0$ES","Planck","$H0$LiCOW","GAIA","TRGB","ACT","WMAP","SPT","GW"]
def ensamble():
    c = ChainConsumer()
    for i in range(len(nombres)):
        c.add_chain(lista[i],parameters=parameters,name=nombres[i])
    return c

c = ensamble()
c.configure(legend_color_text=False)
c.configure_truth(ls=":",color="purple")
fig= c.plotter.plot_summary(errorbar=True,
                            truth=[[0.3153-0.0073,0.3153,0.3147+0.0073],[-1.028-0.031,-1.028,-1.028+0.031]],parameters = parameters[:2],
                            extents=[[0.29,0.38],[-1.3,-0.9]])
fig.set_size_inches(5 + fig.get_size_inches())
#fig.savefig("whisker_nUVX.pdf",dpi=100)

print(spt)
