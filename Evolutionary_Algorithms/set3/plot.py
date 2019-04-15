import numpy as np
import matplotlib.pyplot as plt
import time

problem_name = ''

def relative_error(x, x0):
    return np.abs(x - x0) / x


def absolute_error(x, x0):
    return np.abs(x - x0)


def plot_es_results(problem_name, **data):
    figsize = (12, 4)
    costs = data['costs']
    sigmas = data['sigmas']
    sigmas_means = map(lambda s: s.mean(), sigmas)
    min_cost = costs.min()

    print('Results for {}'.format(problem_name))
    print('Min cost: {}'.format(min_cost))

    print('costs mean: {}'.format(costs.mean()))
    print('costs std: {}'.format(costs.std()))

    plt.figure(figsize=figsize, dpi=80)
    plt.title('Results')
    plt.plot(costs)
    
    plt.figure(figsize=figsize, dpi=80)
    plt.title('Sigmas')
    plt.plot(sigmas)
    