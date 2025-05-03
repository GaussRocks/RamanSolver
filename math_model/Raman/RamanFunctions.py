import numpy as np
from scipy.constants import Planck as hp
from scipy.constants import Boltzmann as Kb
from math_model.Raman.gR import gRintegral as gR
from math_model.Raman.BuildConditions import Parameters
from PadUtils.Lab.ChannelUtils import ChannelTool as ct
from multiprocessing import Pool

dbm = lambda x: ct.from_watt_to_dbm(np.array(x))
watt = lambda x: ct.from_dbm_to_watt(np.array(x))

Aeff, eta, Gamma, gRmax, T, dv, alpha = Parameters.getFiberParam()
z, f, dz = Parameters.getDomain()
Ncpu = Parameters.getNcpu()

def _theta(df):
    return gR(df, gRmax)/(Gamma*Aeff)

def _phi(df):
    return  1 + (np.exp(hp*(df)/(Kb*T)) - 1)**-1

def _sum(px, py, ii):
    S_0_i = 0
    for jj in range(0, ii):
        df = f[ii] - f[jj]
        p = px[jj] + py[jj]
        S_0_i = S_0_i + _theta(-df)*p*(1 + (hp*f[ii]/px[ii])*_phi(-df)*dv)
        
    S_i_n = 0
    for jj in range(ii+1, f.size):
        df = f[ii] - f[jj]
        p = px[jj] + py[jj]
        S_i_n = S_i_n + (f[ii]/f[jj])*_theta(df)*(p + 2*hp*f[ii]*_phi(df)*dv)
        
    return S_0_i, S_i_n

def _funcQ(arg = tuple):
    
    Qf = list(); Qb = list()
    pf, pb = arg[1]
    for ii in arg[0]:
    
        S_0_i_f, S_i_n_f = _sum(pf, pb, ii)
        S_0_i_b, S_i_n_b = _sum(pb, pf, ii)
        
        Qf.append(-alpha[ii] + eta*(pb[ii]/pf[ii]) - S_i_n_f + S_0_i_f)
        Qb.append( alpha[ii] - eta*(pf[ii]/pb[ii]) + S_i_n_b - S_0_i_b)
        
    return [Qf, Qb]

#Parallel
def F_func(zj: float, pj: list):
    
    index = np.arange(0, f.size, 1)
    space = np.array_split(index, Ncpu)
    
    with Pool(Ncpu) as cpu:
        result = cpu.map(_funcQ, [(n, pj) for n in space])
    
    Qf = list(); Qb = list()
    for _res in result: 
        Qf.extend(list(_res[0]))
        Qb.extend(list(_res[1]))
        
    return np.array([np.array(Qf), np.array(Qb)])

#Serial
# def F_func(zj: float, pj: list):
#     index = np.arange(0, f.size, 1)  
#     Qf, Qb = funcQ((index, pj))
#     return np.array([np.array(Qf), np.array(Qb)])