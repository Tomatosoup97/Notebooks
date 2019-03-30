import numpy as np
import time
from functools import partial, reduce


def reverse_sequence_mutation(p):
    a = np.random.choice(len(p), 2, False)
    i, j = a.min(), a.max()
    q = p.copy()
    q[i:j+1] = q[i:j+1][::-1]
    return q


def throas_mutation(k, p):
    i = np.random.choice(len(p)-k)
    q = p.copy()
    q[i:k+1] = q[i:k+1][::-1]
    return q


def scramble_mutation(k, p):
    q = p.copy()
    ss = np.random.choice(len(p), k, replace=False)

    if k == 2:
        q[ss[0]], q[ss[1]] = p[ss[1]], p[ss[0]]
        return q

    new_pos = np.random.choice(ss, k, replace=False)
    for s, pos in zip(ss, new_pos):
        q[pos] = p[s]
    return q


def local_search_mutation(objective_func, k, p):
    positions = np.random.choice(len(p), k, False)
    best_sample_val = objective_func(p)
    best_sample = p
    
    rounds = np.array([0]*len(p))
    
    def inc_round(rs):
        for i in positions:
            if rs[i] >= len(p)-1:
                rs[i] = 0
            else:
                rs[i] += 1
                return True
        return False

    while inc_round(rounds):
        candidate = (p + rounds) % len(p)
        new_cand_val = objective_func(candidate)
        if new_cand_val > best_sample_val:
            best_sample = candidate
            best_sample_val = new_cand_val
    
    return best_sample


two_scramble_mutation = partial(scramble_mutation, 2)