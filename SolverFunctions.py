import numpy as np
from scipy.constants import Planck as hp
from scipy.constants import Boltzmann as Kb
from gR import gRintegral as gR
from multiprocessing import Pool
from BuildConditions import getDomain, getFiberParam
import yaml

from PadUtils.Lab.ChannelUtils import ChannelTool as ct
dbm = lambda x: ct.from_watt_to_dbm(np.array(x))
watt = lambda x: ct.from_dbm_to_watt(np.array(x))

_max=None; _min=watt(-300)

with open('Parameters.yaml') as file:
    param = yaml.safe_load(file)
    
Ncpu = param['Computation']['Ncpu']
path = param['Computation']['path']
fname = param['Computation']['file']
Aeff, eta, Gamma, gRmax, T, dv, alpha = getFiberParam()
z, f, dz = getDomain()

def savetxt(power, itn):
    power =  np.transpose(power, (1, 0, 2))
    N = np.shape(power)[-1]
    # print(N)
    with open(f'{path}/{itn}th_{fname}_fwd.txt', 'ab') as file:
        np.savetxt(file, power[0], fmt=['%.2e']*N, delimiter=', ')
    with open(f'{path}/{itn}th_{fname}_bwd.txt', 'ab') as file:
        np.savetxt(file, power[1], fmt=['%.2e']*N, delimiter=', ')

def MergePower(arr):
    merged = list()
    for n in arr:
        merged.append(
        np.array(list(n[0]) + list(n[1])))
    return merged

def UnmergePower(arr):
    unmerged = list()
    for n in arr:
        unmerged.append(
            np.array_split(n, 2))
    return unmerged

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
    
def Shooting(pa:list, pb:list, pb0, n:int):
    pa = MergePower(pa); pb = MergePower(pb); pb0 = MergePower(pb0)
    _bdiff = np.clip(pb[1]-pb[0],_min,None)
    guess = [
    [(pa[1]*(np.logical_not(pb[0]/pb[1]!=0)+pb[0]/pb[1]))],
    pa[0]+(pa[1]-pa[0])*(pb0-pb[0])/(_bdiff)
    ][n>1]
    return UnmergePower(guess)[0]

def funcQ(arg = tuple):
    
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
        result = cpu.map(funcQ, [(n, pj) for n in space])
    
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

def Avg_power(z, p) -> list:
    F0 = F_func(z[0], p[0])
    return np.clip(np.array(p[0])*np.exp(F0*dz),_min,_max), F0

def AdamBashforth_2th(z, p, F) -> list:
    F1 = F_func(z[1], p[1])
    theta = (3*F1 - F[0])*(dz/2)
    p2_bar = np.clip(np.array(p[1])*np.exp(theta),_min,_max)
    F2_bar = F_func(z[2], p2_bar)
    theta = (F2_bar + F1)*(dz/2)
    return np.clip(np.array(p[1])*np.exp(theta),_min,_max), F1

def AdamBashforth_3th(z, p, F):
    F2 = F_func(z[2], p[2])
    theta = (23*F2 - 16*F[1] + 5*F[0])*(dz/12) 
    p3_bar = np.clip(np.array(p[2])*np.exp(theta),_min,_max)
    F3_bar = F_func(z[3], p3_bar)
    theta = (5*F3_bar + 8*F2 - F[1])*(dz/12)
    return np.clip(np.array(p[2])*np.exp(theta),_min,_max), F2

def AdamBashforth_4th(z, p, F, jj):
    jj = jj+3
    Fjj = F_func(z[jj], p[jj])
    theta = (55*Fjj - 59*F[jj-1] + 37*F[jj-2] - 9*F[jj-3])*(dz/24)
    pjj_1_bar = np.clip(np.array(p[jj])*np.exp(theta),_min,_max)
    Fjj_1_bar = F_func(z[jj+1], pjj_1_bar)
    theta = (9*Fjj_1_bar + 19*Fjj - 5*F[jj-1] + F[jj-2])*(dz/24)
    return np.clip(np.array(p[jj])*np.exp(theta),_min,_max), Fjj