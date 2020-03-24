import numpy as np

def select(population, fitness_function, n):
    """
    Selects individuals from the population. Indivduals are selected
    with probability depending on the results from fitness_function

    prameters:
    poulation - 2D np.array with population
    fitness_function - informs how good each individual is. It must 
                       accept 2D np.array as the input and return
                       1D array with non-negative values
    n - number of individuals to select
    """
    #number of individuals
    count = population.shape[0]   

    f = fitness_function(population)
    sum = np.sum(f)

    # p - probability array
    if sum != 0:
        p = f/sum
    else:
        p = np.ones(count)      
        
    # select n individuals from parents with the given probability
    return population[np.random.choice(count, n, replace=False, p=p)]


class EvolutionAlg:
    
    def fun():
        print(":)")
