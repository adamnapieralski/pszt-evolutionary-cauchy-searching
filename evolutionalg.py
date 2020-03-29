import numpy as np

class EvolutionAlg:

    def __init__(self):
        self.mutation = 'normal'
        self.crossover_method = 'arithmetic'        
        self.mutation_std = 1
        self.fitness_function = lambda x : np.ones(x.shape[0])

    def set_fitness_function(self, fun):
        self.fitness_function = fun

    def setup(self,  mutation='normal', mutation_std=1,
              crossover_method='arithmetic'):
        self.mutation = mutation
        self.crossover_method = crossover_method        
        self.mutation_std = mutation_std

    def run(self, population, fitness_function, iterations, children_num,
            mutation='normal', mutation_std=1,
            crossover_method='arithmetic', crossover_threshold=0.5):
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
        """
        self.mutation = mutation
        self.crossover_method = crossover_method
        self.fitness_function = fitness_function
        self.mutation_std = mutation_std
      
        for it in range(iterations):
            cross_children_num = len([np.random.rand(children_num) > crossover_threshold])
            # print("crossover_children_num", cross_children_num)
            non_cross_children_num = children_num - cross_children_num
            
            #children without crossing
            # print('children without crossing')
            selected = self.select(population, non_cross_children_num)
            # print('selected', selected)            
            children = np.array([self.mutate(x) for x in selected])                           
            # print('children', children)            

            #children with crossing
            for _ in range(cross_children_num):
                # print('iter', i)
                parents = self.select(population, 2)
                # print('parents', parents)
                child = self.crossover(parents)
                # print('child', child)
                mutated = self.mutate(child)
                # print('mutated child', mutated)
                np.append(children, mutated)

            population = self.replace(population, children)

            print('population after ', it, 'step')
            print(population)

    def select(self, population, n):
        """
        Selects individuals from the population. Indivduals are selected
        with probability depending on the results from fitness_function

        parameters:
        poulation - 2D np.array with population        
        n - number of individuals to select
        """
        #number of individuals
        count = population.shape[0]   

        f = self.fitness_function(population)
        # f_univ = [ x - min(f) for x in f ]

        sum = np.sum(f)

        if sum != 0:
            p = f / sum
        else:
            p = np.ones(count)

        # print("Population: ", population)
        # print("Probability: ", p)

        # select n individuals from parents with the given probability        
        return population[np.random.choice(count, n, replace=False, p=p)]


    def mutate(self, individual):
        """
        Generates new individual from the surrounding of the given one
        using normal distribution or cauchy standard distribution.

        Parameters:
        individual: 1D np.array with individual's values         

        Returns:
        new_individual
        """

        if self.mutation == 'normal':
            return np.copy(individual) + np.random.normal(0, self.mutation_std, individual.shape[0])
        
        if self.mutation == 'cauchy':
            return np.copy(individual) + np.random.standard_cauchy(individual.shape[0])
        
        print('wrong mutation method')


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
        poulation: 2D np.array with population
        new_individuals: 2D np.array with newly generated individuals
        """
        all_population = np.row_stack((population, new_individuals))
        return self.select(all_population, population.shape[0])

