import unittest
from evolutionalg import *


def binary_fun(p):
    return [1 if x > 0 else 0 for x in p]

def exp_fun(p):
    return np.array([(np.exp(x[0]) + np.exp(x[1]))/1000 for x in p])


class TestSelect(unittest.TestCase):      

    def test_binary(self):

        population = np.array([4,5,-3,6,7,-4,8,-1,-2, 3, 1, -4])
        population = population.reshape(population.shape[0], 1)

        n = 5
        selected = select(population, binary_fun, n)

        print('test binary')
        print('population', population.T)
        print('selected', selected.T)

        self.assertEqual(selected.shape[0], n)
        for i in selected:
            self.assertTrue(i > 0)


    def test_exp(self):

        population = np.array([[2, 2], [4, 4], [8, 8], [10,10], [16, 16]])
                
        n = 3
        selected = select(population, exp_fun, n)

        print('test exp')
        print('population\n', population)
        print('selected\n', selected)
        self.assertEqual(selected.shape[0], n)
        
        k = 10
        for i in range(k):
            selected = np.concatenate((selected, select(population, exp_fun, n)), axis=0)            
        
        k += 1

        p1 = selected[np.mean(selected, axis=1) >= 16].shape[0]  #16 was selected p1 times
        p2 = selected[np.mean(selected, axis=1) >= 10].shape[0] -p1 #10 was selected p2 times
        p3 = selected[np.mean(selected, axis=1) >= 6].shape[0] - p1 - p2 #8 was selected p3 times
        
        print(p1, p2, p3)
        self.assertTrue(p1 > 0.9*k)
        self.assertTrue(p2 > 0.75*k)
        self.assertTrue(p3 > 0.5*k)


class TestCrossover(unittest.TestCase):
    def test_binary(self):
        parents = np.array([[0,0,0,0], [1,1,1,1]])
        individual = crossover(parents, 'binary')
        print('test binary')
        print(individual)
        for i in individual:
            self.assertTrue(i == 0 or i == 1)

    def test_aritmetic(self):
        parents = np.array([[0,0,0,0], [1,1,1,1]])
        individual = crossover(parents, 'aritmetic')
        print('test aritmetic')
        print(individual)
        for i in individual:
            self.assertTrue(i >= 0 and i <= 1)

if __name__ == '__main__':
    unittest.main()
