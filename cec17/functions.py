"""Implementation of basic modal functions generation from CEC2017 data.
"""
__author__ = "Kostrzewa Lukasz"

import numpy as np
import cec17.basic_functions as bf
import cec17.functions_info as fi

functions = {
    'bent_cigar': bf.bent_cigar_function,
    'zakharov': bf.zakharov_function,
    'rosenbrock': bf.rosenbrock_function,
    'rastrigin': bf.rastrigin_function,
    'expanded_shaffer_f6': bf.expanded_schaffer_f6_function,
    'levy': bf.levy_function,
    'schwefel': bf.schwefel_function
}

def get_rotation_matrix(num, dims):
    '''
    Loads rotation matrix from data directory. Data comes from CEC 2017 files.
    '''
    fname = 'cec17/data/M_{}_D{}.txt'.format(num, dims) 
    return np.loadtxt(fname)

def get_shift_matrix(num, dims):
    '''
    Loads shift vector from data directory. Data comes from CEC 2017 files.    
    '''
    fname = 'cec17/data/shift_data_{}.txt'.format(num)
    return np.loadtxt(fname)[:dims]

def modal_function(function_name, X, modify = True, random_modification = True):
    ''' Returns modal function value for the given X. 
    '''

    if function_name not in fi.available_functions:
        raise ValueError('Wrong function_name')

    if modify:        
        if random_modification:
            num = np.random.randint(1,11)
        else:
            num = fi.function_number[function_name]
        M = get_rotation_matrix(num, X.shape[1])
        o = get_shift_matrix(num, X.shape[1])

        if function_name == 'rosenbrock':
            X_modified = M.dot(2.048e-2*(X - o).T).T + 1
        elif function_name == 'schwefel':
            X_modified = M.dot(10*(X - o).T).T
        else:
            X_modified = M.dot((X - o).T).T 

    else:
        X_modified = X

    F = fi.F_min[function_name]
    
    return functions[function_name](X_modified) + F

def generate_modal_function(function_name, dims, range_limit):
    """
    Generates modal function from CEC 2017 

    Parameters:
    function_name
    dims
    range_limit - needed to ensure that the global minimum after
                  function shift is inside this range
    Returns:
    Shifted and rotated modal function.
    """
    num = fi.function_number[function_name]
    M = get_rotation_matrix(num, dims)
    o = get_shift_matrix(num, dims)
    o = o * range_limit / 100

    F = fi.F_min[function_name]
    base_fun = functions[function_name]

    k = 1.0

    if function_name == 'rosenbrock':
        k = 2.048e-2
    elif function_name == 'schwefel':
        k = 10.0
    elif function_name == 'levy':
        k = 0.0512
    
    def final_function(X):
        X_modified = M.dot(k*(X - o).T).T 
        return base_fun(X_modified) + F

    return final_function