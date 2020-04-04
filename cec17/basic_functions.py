import numpy as np

'''
All functions take n x m np.array as an argument, where
m - number of single point dimensions,  
n - number of points
'''

def bent_cigar_function(X):
    '''
    1) Bent Cigar Function
    '''
    print('bent cigar ', X.shape)
    return X[:,0]**2 + 10e6*np.sum(X[:,1:]**2, axis=1)


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
    4) Rastrigin's Function
    '''
    return np.sum(X**2 - 10*np.cos(2*np.pi*X) + 10, axis=1)