import numpy as numpy
from evolutionalg import *

def fitness(population):
    return [np.abs((x[0] + x[1])) for x in population]

def fitness2(population):
    return [-x[0] * (x[0] - 1) * (x[0] - 2) * (x[0] - 3) * (x[0] - 4) for x in population]

def fitness3(population):
    return [max(-x[0] * (x[0] - 1) * (x[0] - 2) * (x[0] - 3) * (x[0] - 4), 0) for x in population]

e = EvolutionAlg()

# population = np.array([[1,2], [3,4], [3,1], [4,2], [5,3], [6,2]])
# population2 = np.array([[2], [4], [1], [1.5], [4], [2.5], [3], [1.7] ])
population3 = np.array([[0.596], [1.067], [1.184], [1.431], [1.641],
[1.792], [2.068], [2.495], [3.169], [3.785]])

# f = fitness2(population2)
# print(f)

e.run(population3, fitness3, 25, 4, mutation='normal', mutation_std=0.25)