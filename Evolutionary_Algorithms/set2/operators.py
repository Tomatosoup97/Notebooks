import numpy as np
import time
from functools import partial, reduce
from collections import Counter


def _pmx(p1, p2, i, j):
    n = len(p1)

    c = np.full(n, -1)
    c[i:j+1] = p1[i:j+1]

    def find_place(k):
        if k < i or k > j:
            return k
        candidate, = np.where(p2 == p1[k])
        return find_place(candidate)

    for k, el in enumerate(p2[i:j+1], i):
        if el not in c:
            c[find_place(k)] = el

    for k, el in enumerate(c):
        if el == -1:
            c[k] = p2[k]

    return c


def _ox(p1, p2, i, j):
    n = len(p1)

    c = np.full(n, -1)
    c[i:j+1] = p1[i:j+1]

    parent_values = list(filter(lambda x: x not in c[i:j+1], np.append(p2[j+1:], p2[:j+1])))
    end_count = (n-(j+1))
    c[j+1:] = parent_values[:end_count]
    c[:i] = parent_values[end_count:]

    return c


def _ex(p1, p2):
    n = len(p1)
    c = np.full(n, -1)
    edges_count = lambda i, p: Counter({int(p[i-1]): 1, int(p[(i+1)%n]): 1})
    edge_table = {
        x: dict(edges_count(i, p1) + edges_count(np.where(p2 == x)[0], p2))
        for i, x in enumerate(p1)
    }

    c[0] = np.random.choice(p1)
    for i, w in enumerate(c[:n-1]):
        es = edge_table.pop(w)
        edge_table = {k: [v.pop(w,0), v][1] for k, v in edge_table.items()}
        r = {0 if v == 2 else len(edge_table[k]): k for k, v in es.items()}
        new_el = r[min(r)] if r else np.random.choice(list(edge_table.keys()))
        c[i+1] = new_el

    return c


def random_partition(n):
    a = np.random.choice(n, 2, False)
    return a.min(), a.max()  # partition range


def pmx_crossover(p1, p2):
    # Partially-mapped crossover
    i, j = random_partition(len(p1))
    return _pmx(p1, p2, i, j), _pmx(p2, p1, i, j)


def double_pmx_crossover(pq1, pq2):
    p1, q1 = pq1
    p2, q2 = pq2
    return pmx_crossover(p1, q1), pmx_crossover(p2, q2)


def ox_crossover(p1, p2):
    # Order Crossover
    i, j = random_partition(len(p1))
    return _ox(p1, p2, i, j), _ox(p2, p1, i, j)


def ex_crossover(p1, p2):
    # Edge recombination operator
    return _ex(p1, p2), _ex(p1, p2)
