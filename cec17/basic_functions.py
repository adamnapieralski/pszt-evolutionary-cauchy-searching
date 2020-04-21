"""Implementations of basics functions from CEC2017.
"""
__author__ = "Kostrzewa Lukasz, Napieralski Adam"

import numpy as np

'''
All functions take n x m np.array as an argument, where
m - number of single point dimensions,  
n - number of points
'''

def bent_cigar_function(X):
    '''
    1) Bent Cigar Function modified to give smaller values. 10e6 changed to 10e3
    '''    
    return X[:,0]**2 + 10e3*np.sum(X[:,1:]**2, axis=1)


def zakharov_function(X):
    '''
    2) Zakharov Function    
    '''
    return np.sum(X**2, axis=1) + np.sum(0.5*X, axis=1)**2 + np.sum(0.5*X, axis=1)**4

def rosenbrock_function(X):
    '''
    3) Rosenbrock's Function    
    '''
    results = np.empty(X.shape[0])
    for k, x in enumerate(X):
        sum = 0
        for i in range(x.shape[0]-1):
            sum += 100*(x[i]**2 - x[i+1])**2 + (x[i] - 1)**2
        results[k] = sum
    return results

def rastrigin_function(X):
    '''
    4) Rastrigin's Function with 1e-2 * X**2 to allow bigger area 
    '''
    return np.sum(X**2 - 10*np.cos(2*np.pi*X) + 10, axis=1)

def expanded_schaffer_f6_function(X):
    '''
    5) Expanded Schaffer's F6 Function
    '''
    def schaffer(x, y):
        return 0.5+((np.sin(np.sqrt(x**2+y**2)))**2-0.5) / (1+0.001*(x**2+y**2))**2
    
    results = np.empty(X.shape[0])
    for k, x in enumerate(X):
        sum = 0
        for i in range(x.shape[0]-1):
            sum += schaffer(x[i-1], x[i])
        results[k] = sum
    return results

def levy_function(X):
    '''
    8) Levy Function
    '''
    w = 1 + (X-1) / 4
    return (np.sin(np.pi*w[:,0]))**2 + np.sum((w-1)**2*(1+10*(np.sin(np.pi*w + 1)**2)), axis=1) \
           - (w[:,-1]-1)**2*(1+10*(np.sin(np.pi*w[:,-1] + 1)**2)) \
           + (w[:,-1]-1)**2*(1+(np.sin(2*np.pi*w[:,-1])**2))

def schwefel_function(X):
    '''
    9) Modified Schwefel's Function
    '''
    Z = X + 4.209687462275036e2
    results = np.empty(X.shape[0])
    for i, z in enumerate(Z):
        sum = 418.9829*X.shape[1]
        for v in z:
            if np.abs(v) <= 500:
                sum -= v*np.sin(np.sqrt(np.abs(v)))
            elif v > 500:
                sum -= (500-np.mod(v,500))*np.sin(np.sqrt(np.abs(500-np.mod(v,500))))
                -(v-500)**2/(10000*X.shape[1])
            elif v < -500:
                sum -= (np.mod(np.abs(v),500)-500)*np.sin(np.sqrt(np.abs(np.mod(np.abs(v),500)-500)))
                -(v+500)**2/(10000*X.shape[1])
        results[i] = sum
    return results