import emcee
import matplotlib.pyplot as plt
import numpy as np
from chainconsumer import ChainConsumer

shoes = emcee.backends.HDFBackend("Base/shoes.h5").get_chain(flat=True) #Previous shoes version
planck = emcee.backends.HDFBackend("Base/planck.h5").get_chain(flat=True)
holicow = emcee.backends.HDFBackend("Base/holicow.h5").get_chain(flat=True)
gaia = emcee.backends.HDFBackend("Base/gaia.h5").get_chain(flat=True)
trgb = emcee.backends.HDFBackend("Base/trgb.h5").get_chain(flat=True)
act = emcee.backends.HDFBackend("Base/act.h5").get_chain(flat=True)
wmap = emcee.backends.HDFBackend("Base/wmap.h5").get_chain(flat=True)
spt = emcee.backends.HDFBackend("Base/spt.h5").get_chain(flat=True)
gw = emcee.backends.HDFBackend("Base/gw.h5").get_chain(flat=True)

parameters = ["\Omega_m","1/b_3"]
lista = [shoes,planck,holicow,gaia,trgb,act,wmap,spt,gw]
nombres = ["S$H0$ES","Planck","$H0$LiCOW","GAIA","TRGB","ACT","WMAP 9","SPT","GW170717"]
def ensamble():
    c = ChainConsumer()
    for i in range(len(nombres)):
        c.add_chain(lista[i],parameters=parameters,name=nombres[i])
    return c

c = ensamble()
c.configure(legend_color_text=False)
c.configure_truth(ls=":",color="purple")
fig= c.plotter.plot_summary(errorbar=True,truth=[[0.3153-0.0073,0.3153,0.3147+0.0073],[0.0]])
fig.set_size_inches(5 + fig.get_size_inches())
fig.savefig("whisker_f3.pdf",dpi=100)
