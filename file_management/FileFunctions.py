import numpy as np
import yaml

def LoadYamlFile(filename):
    with open(filename) as f:
        param = yaml.safe_load(f)
    return param

def savetxt(power, itn):
    param = LoadYamlFile('./config/config.yaml')
    path = param['output_file_path']
    fname = param['ouput_file_name']
    power =  np.transpose(power, (1, 0, 2))
    N = np.shape(power)[-1]
    with open(f'{path}/{itn}th_{fname}_fwd.txt', 'ab') as file:
        np.savetxt(file, power[0], fmt=['%.2e']*N, delimiter=', ')
    with open(f'{path}/{itn}th_{fname}_bwd.txt', 'ab') as file:
        np.savetxt(file, power[1], fmt=['%.2e']*N, delimiter=', ')