import evolutionalg as evolution
import cec17.functions as cec_functions
import cec17.functions_info as cec_info
from functools import partial
import numpy as np
import time
import sys

def analyze_algorithm(function_name, mutation, max_iterations=1000, population_size = 50, dims=2, runs=51, verbosity = 1, range_limits = [-50, 50], print_out=[sys.stdout]):
    """
    Runs algorithm, handling analyze with CEC2017 guidelines

    Parameters:
    function_name - name of function to analyze (from CEC list)
    mutation - type of mutation applied: 'normal' / 'cauchy'
    max_iterations - max number of iterations (if not found good enough solution)
    population_size - size of population in each iteration
    dims - mutation distribution: 'normal' or 'cauchy'
    runs - value of standard deviation for a normal distribution
    verbosity - 0 - no log, 1 - log after each epoch
    range_limits - number in range <0,1>. Part of children
                            generated from crossing parents
    """
    if function_name not in cec_info.available_functions:
        raise Exception('Wrong function name')
    
    results = []
    progress = []
    populations = []
    maxFES = max_iterations * dims
    function_min = cec_info.F_min[function_name]
    function = cec_functions.generate_modal_function(function_name, dims, range_limits[1])

    e = evolution.EvolutionAlg()
    e.eps = 10e-8
    e.range_limits = range_limits
    e.function_min =  function_min    

    for i in range(runs):
        start_time = time.time()
        population = np.random.rand(population_size,dims)
        population = population * (range_limits[1] - range_limits[0]) + range_limits[0]

        pop, progress_fun = e.run(population, function, maxFES, population_size, mutation)
        f_pop = function(pop)
        populations.append(pop[np.argmin(f_pop)])
        res_min = min(abs(f_pop - function_min))        
        results.append(res_min)
        prog_fun = np.array(progress_fun)
        prog_fun.resize(max_iterations)
        progress.append(prog_fun)

        if(verbosity > 0):
            for out in print_out:
                print('epoch: {}\t result: {}\t duration: {:.2f}s'.format(i+1, res_min, time.time() - start_time), file=out)

    return np.array(results), np.array(progress), np.array(populations)