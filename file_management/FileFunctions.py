import numpy as np
import yaml

with open('./Parameters.yaml') as file:
    param = yaml.safe_load(file)
    
path = param['Computation']['path']
fname = param['Computation']['file']

def savetxt(power, itn):
    power =  np.transpose(power, (1, 0, 2))
    N = np.shape(power)[-1]
    with open(f'{path}/{itn}th_{fname}_fwd.txt', 'ab') as file:
        np.savetxt(file, power[0], fmt=['%.2e']*N, delimiter=', ')
    with open(f'{path}/{itn}th_{fname}_bwd.txt', 'ab') as file:
        np.savetxt(file, power[1], fmt=['%.2e']*N, delimiter=', ')