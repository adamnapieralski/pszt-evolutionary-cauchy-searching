import numpy as np
import cec17.basic_functions as bf
import cec17.functions_info as fi
from functools import partial

# def bent_cigar(X, M, shift, F):
#     '''
#     1) Shifted and Rotated Bent Cigar
#     '''
#     return bf.bent_cigar_function(M*(X-shift)) + F

# def zakharov(X, M, shift, F):
#     '''
#     2) Shifted and Rotated Zakharov Function
#     '''
#     return bf.zakharov_function(M*(X-shift)) + F

# def rastrigin(X, M, shift, F):
#     '''
#     3) Shifted and Rotated Rastrigins's Function
#     '''
#     return bf.rastrigin_function(M*(X-shift)) + F

# def rosenbrock(X, M, shift, F):
#     '''
#     4) Shifted and Rotated Rosenbrock's Function
#     '''
#     return bf.rastrigin_function(M*(X-shift)) + F


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
    Loads rotation matrix from  data directory.    
    '''
    fname = 'cec17/data/M_{}_D{}.txt'.format(num, dims) 
    return np.loadtxt(fname)

def get_shift_matrix(num, dims):
    '''
    Loads shift vector from  data directory.    
    '''
    fname = 'cec17/data/shift_data_{}.txt'.format(num)
    return np.loadtxt(fname)[:dims]

def modal_function(function_name, X, modify = True, random_modification = True):
    '''
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
        else:
            X_modified = M.dot((X - o).T).T 

    else:
        X_modified = X

    F = fi.F_min[function_name]

    return functions[function_name](X_modified) + F

def generate_modal_function(function_name, dims):
    num = fi.function_number[function_name]
    M = get_rotation_matrix(num, dims)
    o = get_shift_matrix(num, dims)

    F = fi.F_min[function_name]
    base_fun = functions[function_name]

    if function_name == 'rosenbrock':
        def final_function(X):
            X_modified = M.dot(2.048e-2*(X - o).T).T 
            return base_fun(X_modified) + F
    else:
        def final_function(X):
            X_modified = M.dot((X - o).T).T 
            return base_fun(X_modified) + F

    return final_function


