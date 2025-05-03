from file_management.FileFunctions import LoadYamlFile
param = LoadYamlFile('./config/config.yaml')

match param['Solver']: 
    case 'PCM': from solver_model.PCM import AdamBashforth_2th, AdamBashforth_3th, AdamBashforth_4th, Avg_power
    case 'AdamBashforth': from solver_model.AdamBashforth import AdamBashforth_2th, AdamBashforth_3th, AdamBashforth_4th, Avg_power