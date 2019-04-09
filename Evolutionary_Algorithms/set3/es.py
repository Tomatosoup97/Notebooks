import numpy as np
import matplotlib.pyplot as plt
import time
import math
from functools import partial, reduce


def compose(*funcs):
    return reduce(lambda f, g: lambda args: f(g(args)), funcs, lambda args: args)


def mu_plus_lambda_replacement(p, children_p, p_values, children_p_values, p_size):
    objective_values = np.hstack([p_values, children_p_values])
    population = np.vstack([p, children_p])

    I = np.argsort(objective_values)
    population = population[I[:p_size], :]
    objective_values = objective_values[I[:p_size]]
    return population, objective_values


def mutation(population, t, t0, chromosome_len, size):
    t0_sq = np.power(t0, 2)
    t_sq = np.power(t, 2)
    for i in range(size):
        xs, sigmas = population[i, :]
        eps0 = np.random.normal(0,  t0_sq)
        epsi = np.random.normal(0, t_sq, chromosome_len)
        sigmas_disturbance = np.exp(eps0 + epsi)
        sigmas *= sigmas_disturbance
        variances = np.power(sigmas, 2)
#        print(sigmas)
        epsi = np.random.normal(0, variances, chromosome_len)
        xs =  xs + epsi
        population[i, :] = xs, sigmas


def mutation(P, Tau, Tau0, chromosome_len, size):
    # copy_P = P.copy()
    ind_size = (P.shape[-1] // 2)
    xs = P[:, 0]
    sigmas = P[:, 1]

    epsilons_i = np.random.randn(*sigmas.shape) * (Tau ** 2)
    epsilons_0 = np.random.randn(sigmas.shape[0], 1) * (Tau0 ** 2)
    sigmas *= np.exp(epsilons_0 + epsilons_i)

    xs += np.random.randn(*xs.shape) * (sigmas ** 2)



def eval_obj_function(objective_func, population, size, chromosome_length):
    objective_values = np.zeros(size)
    for i in range(size):
        objective_values[i] = objective_func(chromosome_length, population[i, :][0])  # first projection?
    return objective_values


def roulette_wheel_parent_indicies(objective_values, population_size, number_of_offspring):
    fitness_values = objective_values.max() - objective_values
    if fitness_values.sum() > 0:
        fitness_values = fitness_values / fitness_values.sum()
    else:
        fitness_values = np.ones(population_size) / population_size
    return np.random.choice(population_size, number_of_offspring, True, fitness_values).astype(np.int64)


def random_population(mu, chromosome_len, domain):
    xs = np.random.uniform(domain[0], domain[1], (mu, chromosome_len))
    sigmas = np.abs(np.random.normal(0, 1, (mu, chromosome_len)))
    return np.array(list(zip(xs, sigmas)))


def es_mu_plus_lambda(mu, lambd, chromosome_len, T, K, F, domain=(0, 1)):
    # mu - population size
    # lambd - number of created children
    # t, t0 - auto-adaptation params
    # K - used to calculate t, t0
    # F - objective function

    costs = np.zeros(T)
    time0 = time.time()
    std_zero_count = 0
    
    t = K / math.sqrt(2*chromosome_len)
    t0 = K / math.sqrt(2*math.sqrt(chromosome_len))

    population = random_population(mu, chromosome_len, domain)
    objective_values = eval_obj_function(F, population, mu, chromosome_len)

    for i in range(T):
        parent_indices = roulette_wheel_parent_indicies(objective_values, mu, lambd)
        children = population[list(parent_indices), :].copy()

        mutation(children, t, t0, chromosome_len, lambd)
        children_objective_values = eval_obj_function(F, children, lambd, chromosome_len)

        population, objective_values = mu_plus_lambda_replacement(population,
                                                                  children,
                                                                  objective_values,
                                                                  children_objective_values,
                                                                  mu)

        # Benchmarks only

        costs[i] = objective_values[0]

        if not i % (T/10) or i == T-1:
            print('%3d %14.8f min: %12.8f mean: %12.8f max: %12.8f std: %12.8f\r' % (
                    i, time.time() - time0,
                    objective_values.min(), objective_values.mean(),
                    objective_values.max(), objective_values.std())
                 )
        
        if objective_values.std() == 0:
            std_zero_count += 1

        if std_zero_count > 50:
            break
    
    return {
        'costs': costs,
    }
