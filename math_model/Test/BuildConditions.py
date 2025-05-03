from file_management.FileFunctions import LoadYamlFile
from PadUtils.Lab.ChannelUtils import ChannelTool as ct
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
        cls.param = LoadYamlFile('./config/TestParameters.yaml')

    @classmethod
    def getBoundaryConditions(cls):
        #BOUNDARY CONDITIONS
        Pf_z0 = np.array([cls.param['Boundary']['Power_0']])
        Pb_z0 = np.array([cls.param['Boundary']['Power_0']])
        
        #COND Z=0 to Z=L
        Pfz0 = Pf_z0 
        Pbz0 = Pb_z0
        
        #Z=0 to Z=L
        Pz0 = [Pfz0, Pbz0]
        
        Pz00 = [Pf_z0, np.zeros([0])]
        PzLL = None
        
        return Pz00, Pz0, PzLL  #Pa0, Pa1, Pb0 

    @classmethod
    def getNit(cls):
        return cls.param['Computation']['Nit']
    
    @classmethod   
    def getDomain(cls):
        #DOMAIN
        dz = cls.param['Domain']['dz']
        z = np.arange(cls.param['Domain']['z0'], cls.param['Domain']['zL']+dz, dz)
        return z, None, dz