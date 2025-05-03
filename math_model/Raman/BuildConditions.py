from file_management.FileFunctions import LoadYamlFile
from utils.ChannelUtils import ChannelTool as ct
import numpy as np

watt = lambda x: ct.from_dbm_to_watt(np.array(x))
dbm = lambda x: ct.from_watt_to_dbm(np.array(x))
conv = ct.unitConverter

def init(cls):
    cls.LoadYamlConfig()
    return cls

@init
class Parameters():
    
    @classmethod
    def LoadYamlConfig(cls):
        cls.param = LoadYamlFile('./config/RamanParameters.yaml')
        cls.Npump = len(cls.param['Pump']['Wavelen']) 
        cls.Nsig = cls.param['Signal']['Nsig']

    @classmethod
    def getNit(cls):
        return cls.param['Computation']['Nit']
    
    @classmethod
    def getNcpu(cls):
        return cls.param['Computation']['Ncpu']

    @classmethod
    def getBoundaryConditions(cls):
        #BOUNDARY CONDITIONS
        Pump_b_zL = watt(cls.param['Pump']['boundary']['backward']['Power_L'][::-1])
        Sig_b_zL = watt(cls.param['Signal']['boundary']['backward']['Power_L'])
        Pb_zL_fp = np.array(list(Pump_b_zL) + [0]*cls.Nsig)
        Pb_zL_fs = np.array([0]*cls.Npump + [Sig_b_zL]*cls.Nsig)
        
        Pump_f_z0 = watt(cls.param['Pump']['boundary']['forward']['Power_0'])
        Sig_f_z0 = watt(cls.param['Signal']['boundary']['forward']['Power_0'])
        Pf_z0_fp = np.array(list(Pump_f_z0) + [0]*cls.Nsig)
        Pf_z0_fs = np.array([0]*cls.Npump + [Sig_f_z0]*cls.Nsig)
        
        #GUESSING FOR Z=L to Z=0 INTEGRATION
        Pump_f_zL = watt(cls.param['Pump']['boundary']['forward']['Power_L'])
        Sig_f_zL = watt(cls.param['Signal']['boundary']['forward']['Power_L'])
        Pf_zL_fp = np.array(list(Pump_f_zL) + [0]*cls.Nsig)
        Pf_zL_fs = np.array([0]*cls.Npump + [Sig_f_zL]*cls.Nsig)
        
        #GUESSING FOR Z=0 to Z=L INTEGRATION
        Pump_b_z0 = watt(cls.param['Pump']['boundary']['backward']['Power_0'])
        Sig_b_z0 = watt(cls.param['Signal']['boundary']['backward']['Power_0'])
        Pb_z0_fp = np.array(list(Pump_b_z0) + [0]*cls.Nsig)
        Pb_z0_fs = np.array([0]*cls.Npump + [Sig_b_z0]*cls.Nsig)
        
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
        
        Pz00 = [Pf_z0_fp + Pf_z0_fs, np.zeros(cls.Npump+cls.Nsig)]
        PzLL = [np.zeros(cls.Npump+cls.Nsig), Pb_zL_fp + Pb_zL_fs]
        
        match cls.param['Computation']['Direction'].upper():
            case 'FORWARD': return Pz00, Pz0, PzLL  #Pa0, Pa1, Pb0 
            case 'BACKWARD': return PzLL, PzL, Pz00 #Pa0, Pa1, Pb0 
    
    @classmethod   
    def getDomain(cls):
        #DOMAIN
        dz = cls.param['Fiber']['dz']
        f = np.linspace(cls.param['Signal']['start'], cls.param['Signal']['stop'], cls.Nsig)[::-1]
        f = np.array(list(conv(cls.param['Pump']['Wavelen'])*1e12) + list(f))
        z = np.arange(cls.param['Fiber']['zL'], 0-dz, -dz)
        match cls.param['Computation']['Direction'].upper():
            case 'FORWARD': return z[::-1], f, dz 
            case 'BACKWARD': return z, f, -dz
    
    @classmethod        
    def getFiberParam(cls):
        #FIBER PARAMETERS
        alpha = np.array(cls.param['Pump']['alpha'][::-1] + [cls.param['Signal']['alpha']]*cls.Nsig)/(4.343*1000)
        par = [cls.param['Fiber'][n] for n in list(cls.param['Fiber'])[2:]]
        par.append(alpha)
        return par