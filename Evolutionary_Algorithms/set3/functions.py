import math
import numpy as np
from functools import reduce, partial


def eval_obj_function(objective_func, population):
    chromosome_length = population.shape[1]
    size = population.shape[0]
    objective_values = np.zeros(size)
    for i in range(size):
        objective_values[i] = objective_func(chromosome_length, population[i, :])
    return objective_values


def compose(*funcs):
    return reduce(lambda f, g: lambda args: f(g(args)), funcs, lambda args: args)


def bohachevsky_function(n, x):
    # n is always 2
    return x[0]**2+2*x[1]**2-0.3*math.cos(3*math.pi*x[0])-0.4*math.cos(4*math.pi*x[1])+0.7


def bohachevsky(X):
    return eval_obj_function(bohachevsky_function, X)


def dixon_and_price_function(n, x):    
    s = 0
    for j in range(1, n):
        s += j*(2*x[j]**2-x[j-1])**2
    return s+(x[0]-1)**2;


def dixon_and_price(X):
    return eval_obj_function(dixon_and_price_function, X)


def rastrigin(X):
    n = X.shape[1]
    s = (np.power(X, 2) - 10*np.cos(2*math.pi*X)).sum(axis=1)
    return 10*n+s


def griewank(X):
    fr = 4000
    s = np.power(X, 2).sum(axis=1) / fr
    p = np.cos(
        X / np.sqrt(
            np.arange(1, X.shape[1]+1)
        )
    ).prod(axis=1)
    return s-p+1


def schwefel(X):
    n = X.shape[1]
    s = (X * compose(np.sin, np.sqrt, np.abs)(X)).sum(axis=1)
    return 418.9829*n-s
