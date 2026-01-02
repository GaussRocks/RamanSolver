from utils.ChannelUtils import ChannelTool as ct
from math_model import F_func
import numpy as np

dbm = lambda x: ct.from_watt_to_dbm(np.array(x))
watt = lambda x: ct.from_dbm_to_watt(np.array(x))

_max=None; _min=watt(-300)

def Avg_power(z, p, dz) -> list:
    F0 = F_func(z[0], p[0])
    return np.clip(np.array(p[0])*np.exp(F0*dz),_min,_max), F0

def AdamBashforth_2th(z, p, F, dz) -> list:
    F1 = F_func(z[1], p[1])
    theta = (3*F1 - F[0])*(dz/2)
    return np.clip(np.array(p[1])*np.exp(theta),_min,_max), F1

def AdamBashforth_3th(z, p, F, dz):
    F2 = F_func(z[2], p[2])
    theta = (23*F2 - 16*F[1] + 5*F[0])*(dz/12) 
    return np.clip(np.array(p[2])*np.exp(theta),_min,_max), F2

def AdamBashforth_4th(z, p, F, jj, dz):
    jj = jj+3
    Fjj = F_func(z[jj], p[jj])
    theta = (55*Fjj - 59*F[jj-1] + 37*F[jj-2] - 9*F[jj-3])*(dz/24)
    return np.clip(np.array(p[jj])*np.exp(theta),_min,_max), Fjj