"""Evolution Algorithm class module.

Implements all steps of standard evolution algorithm including
individuals selection, mutation, crossover and replacement in population.
"""
__author__ = "Kostrzewa Lukasz, Napieralski Adam"

import numpy as np

class EvolutionAlg:

    def __init__(self):
        self.mutation = 'normal'
        self.crossover_method = 'arithmetic'        
        self.mutation_std = 1
        self.fitness_function = lambda x : np.ones(x.shape[0])        
        self.tournament_size=2
        self.function_min = None
        self.eps = 10e-8
        self.range_limits = [-100, 100]

    def setup(self,  mutation='normal', mutation_std=1,
              crossover_method='arithmetic'):
        self.mutation = mutation
        self.crossover_method = crossover_method        
        self.mutation_std = mutation_std

    def run(self, population, fitness_function, iterations, children_num,
            mutation='normal', mutation_std=1,
            crossover_method='arithmetic', crossover_threshold=0.5, verbosity = 0,
            tournament_size=2):
        """
        Runs evolution algorithm

        Parameters:
        poulation - 2D np.array with population
        fitness_function - informs how good each individual is. It must 
                           accept 2D np.array as the input and return
                           1D array with non-negative values
        iterations - number of iterations
        children_num - number of children generated in each iteration
        mutation - mutation distribution: 'normal' or 'cauchy'
        mutation_std - value of standard deviation for a normal distribution
        crossover_method - String 'arithmetic' or 'binary'. For binary
                           weights are 0 or 1. In arithmetic they are
                           in range <0,1>.
        crossover_threshold - number in range <0,1>. Part of children
                              generated from crossing parents
        verbosity - 0 - no log, 1 - log after each epoch        
        tournament_size - tournament group size
        """
        self.mutation = mutation
        self.crossover_method = crossover_method
        self.fitness_function = fitness_function
        self.mutation_std = mutation_std        
        self.tournament_size = tournament_size

        error = []
      
        for it in range(iterations):
            cross_children_num = len([np.random.rand(children_num) > crossover_threshold])
            non_cross_children_num = children_num - cross_children_num
            
            #children without crossing
            selected = self.select(population, non_cross_children_num)
            children = np.array([self.mutate(x) for x in selected])                           

            #children with crossing
            for _ in range(cross_children_num):
                parents = self.select(population, 2)
                child = self.crossover(parents)
                mutated = self.mutate(child)

                children = np.row_stack((children, mutated))
            
            population = self.replace(population, children)

            if verbosity == 1:
                print('population after ', it+1, 'step')
                print(population)

            if self.function_min is not None:
                err = min(abs(self.fitness_function(population) - self.function_min))
                error.append(err)
                if(err < self.eps):
                    print('Evolution ends in ', it+1, ' epoch reaching error ', err)
                    break

        return population, error

    def select(self, population, n):
        """
        Selects individuals from the population. Indivduals are selected
        using tournament method.

        parameters:
        poulation - 2D np.array with population        
        n - number of individuals to select
        """
        #number of individuals
        count = population.shape[0]

        new_population = []
        for _ in range(n):
            group = population[np.random.choice(count, self.tournament_size, replace=True)]                
            min_ind = np.argmin(self.fitness_function(group))
            new_population.append(group[min_ind])
        return np.array(new_population)
  
    def mutate(self, individual):
        """
        Generates new individual from the surrounding of the given one
        using normal distribution or cauchy standard distribution. Values
        after mutation must be inside the range limits.

        Parameters:
        individual - 1D np.array with individual's values         

        Returns:
        new_individual
        """

        if self.mutation == 'normal':
            mutated = np.copy(individual) + np.random.normal(0, self.mutation_std, individual.shape[0])
        
        if self.mutation == 'cauchy':
            mutated = np.copy(individual) + np.random.standard_cauchy(individual.shape[0])

        return np.clip(mutated, self.range_limits[0], self.range_limits[1])

    def crossover(self, parents):
        """
        Generates a new individual from parents.
        Parameters:
        parents - 2D np.array 2 x n representing parents. 
        
        Returns:
        A new individual.
        """

        n = parents.shape[1]

        if self.crossover_method == 'binary':
            w = np.random.randint(2, size=n)
        else:
            if self.crossover_method != 'arithmetic':
                print('Wrong method. Only arithmetic and binary methods are accepted.'
                    ' Using arithmetic.')
            
            w = np.random.rand(n)

        return w*parents[0] + (1-w)*parents[1]

    
    def replace(self, population, new_individuals):
        """
        Replaces old population with new one containing individuals
        selected from the previous one + newly generated individuals.
        Indivduals are selected with probability depending on the results from fitness_function.

        Parameters:
        poulation - 2D np.array with population
        new_individuals - 2D np.array with newly generated individuals
        """
        f_old, f_new = self.fitness_function(population), self.fitness_function(new_individuals)

        if min(f_old) < min(f_new):
            new_individuals[np.argmax(f_new)] = population[np.argmin(f_old)]

        return new_individuals