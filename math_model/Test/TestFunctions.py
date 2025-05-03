"""
Solve:
dy/dz = y*ln(y)*(z**2 + z + 1)
y(0) = 2

Exact Solution: 
y(z) = exp{exp[1/3*z**3 + 1/2*z**2 + z + ln(ln(2))]}
"""

import numpy as np
from math_model.Test.BuildConditions import Parameters

z, _, dz = Parameters.getDomain()

def F_func(zj: float, pj: list):
    pfj, pbj = pj
    return np.array([np.log(pfj)*(zj**2 + zj + 1), np.log(pbj)*(zj**2 + zj + 1)])