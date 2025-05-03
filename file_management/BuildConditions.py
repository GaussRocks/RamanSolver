from PadUtils.Lab.ChannelUtils import ChannelTool as ct
import numpy as np
import yaml

watt = lambda x: ct.from_dbm_to_watt(np.array(x))
dbm = lambda x: ct.from_watt_to_dbm(np.array(x))
conv = ct.unitConverter

with open('Parameters.yaml') as f:
    param = yaml.safe_load(f)

Npump = len(param['Pump']['Wavelen'])
Nsig = param['Signal']['Nsig']

def getBoundaryConditions():
    #BOUNDARY CONDITIONS
    Pump_b_zL = watt(param['Pump']['boundary']['backward']['Power_L'][::-1])
    Sig_b_zL = watt(param['Signal']['boundary']['backward']['Power_L'])
    Pb_zL_fp = np.array(list(Pump_b_zL) + [0]*Nsig)
    Pb_zL_fs = np.array([0]*Npump + [Sig_b_zL]*Nsig)
    
    Pump_f_z0 = watt(param['Pump']['boundary']['forward']['Power_0'])
    Sig_f_z0 = watt(param['Signal']['boundary']['forward']['Power_0'])
    Pf_z0_fp = np.array(list(Pump_f_z0) + [0]*Nsig)
    Pf_z0_fs = np.array([0]*Npump + [Sig_f_z0]*Nsig)
    
    #GUESSING FOR Z=L to Z=0 INTEGRATION
    Pump_f_zL = watt(param['Pump']['boundary']['forward']['Power_L'])
    Sig_f_zL = watt(param['Signal']['boundary']['forward']['Power_L'])
    Pf_zL_fp = np.array(list(Pump_f_zL) + [0]*Nsig)
    Pf_zL_fs = np.array([0]*Npump + [Sig_f_zL]*Nsig)
    
    #GUESSING FOR Z=0 to Z=L INTEGRATION
    Pump_b_z0 = watt(param['Pump']['boundary']['backward']['Power_0'])
    Sig_b_z0 = watt(param['Signal']['boundary']['backward']['Power_0'])
    Pb_z0_fp = np.array(list(Pump_b_z0) + [0]*Nsig)
    Pb_z0_fs = np.array([0]*Npump + [Sig_b_z0]*Nsig)
    
    #COND Z=L to L=0
    PfzL = Pf_zL_fp + Pf_zL_fs #GUESS
    PbzL = Pb_zL_fp + Pb_zL_fs 
    
    #COND Z=0 to Z=L
    Pfz0 = Pf_z0_fp + Pf_z0_fs 
    Pbz0 = Pb_z0_fp + Pb_z0_fs #GUESS
    
    #Z=L to Z=0
    PzL = [PfzL, PbzL]
    #Z=0 to Z=L
    Pz0 = [Pfz0, Pbz0]
    
    Pz00 = [Pf_z0_fp + Pf_z0_fs, np.zeros(Npump+Nsig)]
    PzLL = [np.zeros(Npump+Nsig), Pb_zL_fp + Pb_zL_fs]
    
    match param['Computation']['Direction'].upper():
        case 'FORWARD': return Pz00, Pz0, PzLL  #Pa0, Pa1, Pb0 
        case 'BACKWARD': return PzLL, PzL, Pz00 #Pa0, Pa1, Pb0 
    
def getDomain():
    dz = param['Fiber']['dz']
    f = np.linspace(param['Signal']['start'], param['Signal']['stop'], Nsig)[::-1]
    f = np.array(list(conv(param['Pump']['Wavelen'])*1e12) + list(f))
    z = np.arange(param['Fiber']['zL'], 0-dz, -dz)
    
    match param['Computation']['Direction'].upper():
        case 'FORWARD': return z[::-1], f, dz 
        case 'BACKWARD': return z, f, -dz
        
def getFiberParam():
    alpha = np.array(param['Pump']['alpha'][::-1] + [param['Signal']['alpha']]*Nsig)/(4.343*1000)
    par = [param['Fiber'][n] for n in list(param['Fiber'])[2:]]
    par.append(alpha)
    return par