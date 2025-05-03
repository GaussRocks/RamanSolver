from file_management.FileFunctions import LoadYamlFile
param = LoadYamlFile('./config/config.yaml')

match param['Model']: 

    case 'Raman': 
        from math_model.Raman.RamanFunctions import F_func
        from math_model.Raman.BuildConditions import Parameters
        
    case 'Test': 
        from math_model.Test.TestFunctions import F_func
        from math_model.Test.BuildConditions import Parameters