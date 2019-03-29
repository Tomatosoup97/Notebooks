import numpy as np
import math


def romberg(f, a, b, n, R):
    h = b - a
    R[0][0] = 0.5 * h * (f(a) + f(b))
    print(R[0][0])

    for i in range(n):
        h *= 0.5
        cur_sum = 0

        # cur_sum = sum(map(lambda k: f(a + k*h), range(1, 2**i-1, 2)))
        for k in range(1, 2**i-1, 2):
            cur_sum += f(a + k*h)

        R[i][0] = 0.5 * R[i-1][0] + cur_sum * h
        print(R[i][0], end=' ')

        for j in range(1, i):
            R[i][j] = R[i][j-1] + (R[i][j-1] - R[i-1][j-1]) / (4**j-1.)
            print(R[i][j], end=' ')

        print('')


def main():
    n = 15
    R = np.zeros((n+1, n+1))

    f = lambda x: 2018*x**5 + 2017*x**4 - 2016*x**3 + 2015*x
    g = lambda x: 1 / (1+x**2)
    h = lambda x: math.cos(2*x) / x

    print('a)')
    romberg(f, -2, 3, n, R)

    print('\nb)')
    romberg(g, -5, 5, n, R)

    print('\nc)')
    romberg(h, math.pi, 15, n, R)

    print('\n')


if __name__ == '__main__':
    main()

