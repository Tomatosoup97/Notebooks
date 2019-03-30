import numpy as np
import matplotlib.pyplot as plt
import time

problem_name = ''

opts = {
    'berlin52': 7542,
    'bayg29': 1610,
    'bays29': 2020,
    'kroA100': 21282,
    'kroA150': 26524,
    'kroA200': 29368,
}

def relative_error(x, x0):
    return np.abs(x - x0) / x


def absolute_error(x, x0):
    return np.abs(x - x0)


def plot_sga_results(problem_name, **data):
    figsize = (12, 4)
    costs = data['costs']
    min_cost = data['best_value']

    print('Min cost: {}'.format(min_cost))
    print('OPT cost: {}'.format(opts[problem_name]))
    print('Relative error: {}%'.format(round(relative_error(opts[problem_name], min_cost)*100, 2)))
    print('Absolute error: {}'.format(round(absolute_error(opts[problem_name], min_cost), 2)))

    print('costs mean: {}'.format(costs.mean()))
    print('costs std: {}'.format(costs.std()))

    plt.figure(figsize=figsize, dpi=80)
    plt.title('Results')
    plt.plot(costs)
