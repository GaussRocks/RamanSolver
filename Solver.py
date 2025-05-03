from solver_model import AdamBashforth_2th, AdamBashforth_3th, AdamBashforth_4th, Avg_power
from utils.ChannelUtils import ChannelTool as ct
from solver_model.ShootingFunctions import Shooting
from file_management.FileFunctions import savetxt
from matplotlib import pyplot as plt
from math_model import Parameters
from tqdm import tqdm
import numpy as np 
import sys

dbm = lambda x: ct.from_watt_to_dbm(np.array(x))
watt = lambda x: ct.from_dbm_to_watt(np.array(x))

_max=None; _min=watt(-300)

Nit = Parameters.getNit()
Pa0, Pa1, Pb0 = Parameters.getBoundaryConditions()
z, f, dz = Parameters.getDomain()

if __name__ == "__main__":
    
    it = list()
    pa = [Pa0, Pa1]; pb = [Pb0]

    for itn in range(1, Nit+1):
        with tqdm(total=z.size-1, file=sys.stdout) as pbar:
            
            p = [pa[-1]]; F=list()
            
            p1, F0 = Avg_power(z, p, dz)
            p.append(p1); F.append(F0)
            
            p2, F1 = AdamBashforth_2th(z, p, F, dz)
            p.append(p2); F.append(F1)
            
            p3, F2 =  AdamBashforth_3th(z, p, F, dz)
            p.append(p3); F.append(F2)
            
            savetxt(p, itn)
            pbar.update(3)
            for n in range(z.size-4):
                pj, Fj = AdamBashforth_4th(z, p, F, n, dz)
                p.append(pj); F.append(Fj)
                savetxt([p[-1]], itn)
                pbar.update(1)
        
        # pb.append(np.clip(p[-1], _min, _max))
        # pa.append(np.clip(
        # Shooting(pa[-2:], pb[-2:], [Pb0], itn), _min, _max))
        p = np.array(p)
        it.append(p.T)
        
    plt.semilogy(z, it[-1][-1][0])