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
    min_cost = costs.min()

    print('Results for {}'.format(problem_name))
    print('Min cost: {}'.format(min_cost))
#     print('Relative error: {}%'.format(round(relative_error(opts[problem_name], min_cost)*100, 2)))
#     print('Absolute error: {}'.format(round(absolute_error(opts[problem_name], min_cost), 2)))

    print('costs mean: {}'.format(costs.mean()))
    print('costs std: {}'.format(costs.std()))

    plt.figure(figsize=figsize, dpi=80)
    plt.title('Results')
    plt.plot(costs)
    