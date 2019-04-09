import math
import numpy as np
from functools import reduce, partial


def compose(*funcs):
    return reduce(lambda f, g: lambda args: f(g(args)), funcs, lambda args: args)


def bohachevsky_function(n, x):
    # n is always 2
    return x[0]**2+2*x[1]**2-0.3*math.cos(3*math.pi*x[0])-0.4*math.cos(4*math.pi*x[1])+0.7


def dixon_and_price_function(n, x):
    s = 0
    for j in range(1, n):
        s += j*(2*x[j]**2-x[j-1])**2
    return s+(x[0]-1)**2;


def rastrigin_function(n, x):
    s = (np.power(x, 2) - 10*np.cos(2*math.pi*x)).sum()
    return 10*n+s


def griewank_function(n, x):
    fr = 4000
    s = (np.power(x, 2) / fr).sum()
    p = np.cos(x / np.sqrt(np.arange(1, x.shape[0]+1))).prod()
    return s-p+1


def schwefel_function(n, x):
    s = (x * compose(np.sin, np.sqrt, np.abs)(x)).sum()
    return 418.9829*n-s
