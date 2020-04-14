import evolutionalg as evolution
import cec17.functions as cec_functions
import cec17.functions_info as cec_info
from functools import partial
import numpy as np

def analyze_algorithm(function_name, mutation, max_iterations=1000, population_size = 50, dims=2, runs=51, verbosity = 1, range_limits = [-50, 50]):

    if function_name not in cec_info.available_functions:
        raise Exception('Wrong function name')
    
    results = []    
    maxFES = max_iterations * dims    
    function_min = cec_info.F_min[function_name]
    function = cec_functions.generate_modal_function(function_name, dims, range_limits[1])

    e = evolution.EvolutionAlg()
    e.eps = 10e-4
    e.range_limits = range_limits
    e.function_min =  function_min    

    for i in range(runs):
        population = np.random.rand(population_size,dims)
        population = population * (range_limits[1] - range_limits[0]) + range_limits[0]

        pop, progress_fun = e.run(population, function, maxFES, population_size, mutation)
        res = abs(function(pop) - function_min)
        res_min = min(res)
        results.append(res_min)

        if(verbosity > 0):
            print('epoch: {}\t result: {}'.format(i+1, res_min))

    np_results = np.array(results)

    return np.min(np_results),  np.max(np_results), np.mean(np_results), np.median(np_results), np.std(np_results), progress_fun