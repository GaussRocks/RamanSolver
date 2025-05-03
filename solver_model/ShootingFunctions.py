from PadUtils.Lab.ChannelUtils import ChannelTool as ct
import numpy as np

dbm = lambda x: ct.from_watt_to_dbm(np.array(x))
watt = lambda x: ct.from_dbm_to_watt(np.array(x))

_max=None; _min=watt(-300)

def _MergePower(arr):
    merged = list()
    for n in arr:
        merged.append(
        np.array(list(n[0]) + list(n[1])))
    return merged

def _UnmergePower(arr):
    unmerged = list()
    for n in arr:
        unmerged.append(
            np.array_split(n, 2))
    return unmerged
    
def Shooting(pa:list, pb:list, pb0, n:int):
    pa = _MergePower(pa); pb = _MergePower(pb); pb0 = _MergePower(pb0)
    _bdiff = np.clip(pb[1]-pb[0],_min,None)
    guess = [
    [(pa[1]*(np.logical_not(pb[0]/pb[1]!=0)+pb[0]/pb[1]))],
    pa[0]+(pa[1]-pa[0])*(pb0-pb[0])/(_bdiff)
    ][n>1]
    return _UnmergePower(guess)[0]
