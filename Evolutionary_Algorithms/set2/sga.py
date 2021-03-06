import numpy as np
import time
from functools import partial, reduce
from collections import Counter

from plot import *
from operators import *
from mutations import *

# TODO: hardcoded 2's for Q3AP in gen_initial_population, create_children_population: (size, 2, chromosome_len)

def tsp_objective_function(n, A, perm):
    n = len(perm)
    s = 0.0
    for i in range(n):
        s += A[perm[i-1], perm[i]]
    return s


def gen_initial_population(size, chromosome_len, gen_random_sample):
    current_population = np.zeros((size, 2, chromosome_len), dtype=np.int64)
    for i in range(size):
        current_population[i, :] = gen_random_sample(chromosome_len)
    return current_population


def eval_obj_function(A, chromosome_length, objective_func, population, size):
    objective_values = np.zeros(size)
    for i in range(size):
        objective_values[i] = objective_func(chromosome_length, A, population[i, :])
    return objective_values


def mutation(mutation_func, population, size, probability):
    for i in range(size):
        if np.random.random() < probability:
            population[i, :] = mutation_func(population[i, :])


def mu_plus_lambda_replacement(population, children_population,
                               objective_vals, children_objective_vals, population_size):
    objective_vals = np.hstack([objective_vals, children_objective_vals])
    population = np.vstack([population, children_population])

    I = np.argsort(objective_vals)
    population = population[I[:population_size], :]
    objective_vals = objective_vals[I[:population_size]]
    return population, objective_vals


def roulette_wheel_parent_indicies(objective_values, population_size, number_of_offspring):
    fitness_values = objective_values.max() - objective_values
    if fitness_values.sum() > 0:
        fitness_values = fitness_values / fitness_values.sum()
    else:
        fitness_values = np.ones(population_size) / population_size
    return np.random.choice(population_size, number_of_offspring, True, fitness_values).astype(np.int64)


def create_children_population(population, parent_indices, number_of_offspring,
                               chromosome_length, crossover_operator, crossover_probability):
    children_population = np.zeros((number_of_offspring, 2, chromosome_length), dtype=np.int64)
    for i in range(int(number_of_offspring/2)):
        if np.random.random() < crossover_probability:
            children_population[2*i, :], children_population[2*i+1, :] = \
                crossover_operator(population[parent_indices[2*i], :].copy(),
                                   population[parent_indices[2*i+1], :].copy())
        else:
            children_population[2*i, :], children_population[2*i+1, :] = \
                population[parent_indices[2*i], :].copy(), population[parent_indices[2*i+1]].copy()
    if np.mod(number_of_offspring, 2) == 1:
        children_population[-1, :] = population[parent_indices[-1], :]
    return children_population

crossover_operator = pmx_crossover
mutation_operator = reverse_sequence_mutation


def gen_initial_sample(chromosome_length):
    return np.zeros((1, chromosome_length))


def gen_double_initial_sample(chromosome_length):
    return gen_initial_sample(chromosome_length), gen_initial_sample(chromosome_length)


def gen_random_sample(chromosome_length):
    return np.random.permutation(chromosome_length)


def gen_random_double_sample(chromosome_length):
    return gen_random_sample(chromosome_length), gen_random_sample(chromosome_length)


def sga(T,
        distance_matrix,
        crossover_operator,
        mutation_operator,
        population_size,
        chromosome_length,
        crossover_probability,
        mutation_probability,
        objective_func=tsp_objective_function,
        local_search_probability=0,
        local_search_k=2,
        gen_initial_sample=gen_initial_sample,
        gen_random_sample=gen_random_sample):

    costs = np.zeros(T)
    
    number_of_offspring = population_size
    best_objective_value = np.Inf
    best_chromosome = gen_initial_sample(chromosome_length)

    current_population = gen_initial_population(population_size, chromosome_length, gen_random_sample)

    eval_sga_obj_func = partial(eval_obj_function, distance_matrix, chromosome_length, objective_func)

    objective_values = eval_sga_obj_func(current_population, population_size)

    time0 = time.time()
    
    std_zero_count = 0

    for t in range(T):
        parent_indices = roulette_wheel_parent_indicies(objective_values, population_size, number_of_offspring)

        children_population = create_children_population(current_population, parent_indices,
                                                         number_of_offspring, chromosome_length,
                                                         crossover_operator, crossover_probability)

        mutation(mutation_operator, children_population, number_of_offspring, mutation_probability)

        local_search_operator = partial(local_search_mutation,
            partial(objective_func, chromosome_length, distance_matrix), 
            local_search_k,
        )
        if (T * 2/10) < t:
            mutation(local_search_operator, children_population, number_of_offspring, local_search_probability)

        children_objective_values = eval_sga_obj_func(children_population, number_of_offspring)

        current_population, objective_values = mu_plus_lambda_replacement(current_population, children_population,
                                                                          objective_values,
                                                                          children_objective_values,
                                                                          population_size)
        if objective_values[0] < best_objective_value:
            best_objective_value = objective_values[0]
            best_chromosome = current_population[0, :]

        costs[t] = objective_values[0]

        if not t % (T/10) or t == T-1:
            print('%3d %14.8f min: %12.8f mean: %12.8f max: %12.8f std: %12.8f' % (
                    t, time.time() - time0,
                    objective_values.min(), objective_values.mean(),
                    objective_values.max(), objective_values.std())
                 )
        
        if objective_values.std() == 0:
            std_zero_count += 1
           
        if std_zero_count > 20:
            break

            
    return {
        'costs': costs,
        'best_value': best_objective_value,
        'best_chromosome': best_chromosome,
    }
