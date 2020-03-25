import numpy as np

def select(population, fitness_function, n):
    """
    Selects individuals from the population. Indivduals are selected
    with probability depending on the results from fitness_function

    parameters:
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


def mutate_normal(individual, std_deviation=1.0):
    """
    Generates new individual from the surrounding of the given one
    using normal distribution.

    Parameters:
    individual: 1D np.array with individual's values 
    std_deviation: value of standard deviation for a distribution

    Returns:
    new_individual
    """

    new_individual = np.copy(individual) + np.random.normal(0, std_deviation, individual.shape[0])
    
    return new_individual

def mutate_cauchy(individual):
    """
    Generates new individual from the surrounding of the given one using
    cauchy standard distribution.

    Parameters:
    individual: 1D np.array with individual's values 

    Returns:
    new_individual
    """

    new_individual = np.copy(individual) + np.random.standard_cauchy(individual.shape[0])
    
    return new_individual

def replace(population, new_individuals, fitness_function):
    """
    Replaces old population with new one containing individuals
    selected from the previous one + newly generated individuals.
    Indivduals are selected with probability depending on the results from fitness_function.

    Parameters:
    poulation: 2D np.array with population
    new_individuals: 2D np.array with newly generated individuals
    fitness_function: informs how good each individual is. It must 
                       accept 2D np.array as the input and return
                       1D array with non-negative values
    """
    all_population = np.row_stack((population, new_individuals))
    population = select(all_population, fitness_function, population.shape[0])


class EvolutionAlg:
    
    def fun():
        print(":)")
