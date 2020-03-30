import numpy as numpy
from evolutionalg import *

def fitness(population):
    return [np.abs((x[0] + x[1])) for x in population]

def fitness2(population):
    return [-x[0] * (x[0] - 1) * (x[0] - 2) * (x[0] - 3) * (x[0] - 4) for x in population]

def fitness3(population):
    return [max(-x[0] * (x[0] - 1) * (x[0] - 2) * (x[0] - 3) * (x[0] - 4), 0) for x in population]


def normal_distribution(x, mean, dev):
    return 1. / (dev * np.sqrt(2*np.pi)) * np.power(np.e, -0.5 * ((x - mean) / dev)**2)

def two_gauss(x):
    return normal_distribution(x, 10, 2) + normal_distribution(x, -2, 3.5)

def fitness4(population):
    return [two_gauss(x[0]) for x in population]

e = EvolutionAlg()

# population = np.array([[1,2], [3,4], [3,1], [4,2], [5,3], [6,2]])
# population2 = np.array([[2], [4], [1], [1.5], [4], [2.5], [3], [1.7] ])
population3 = np.array([[0.596], [1.067], [1.184], [1.431], [1.641],
[1.792], [2.068], [2.495], [3.169], [3.785]])

population4 = np.linspace(-20, 20, 20).reshape((20, 1))

population_normal = e.run(population4, fitness4, 25, 4, mutation='normal', mutation_std=0.25)

population_cauchy = e.run(population4, fitness4, 25, 4, mutation='cauchy')

print('population normal')
print(population_normal)

print('population cauchy')
print(population_cauchy)