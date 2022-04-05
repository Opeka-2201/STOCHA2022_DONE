import numpy as np
from sympy import false, true
import matplotlib as plt
from scipy.stats import binom
K = 10
p = 0.3
r = 0.1


def q_distrib(y, x):
    if x == 0 and y == 0:
        return r
    elif x == K and y == K:
        return (1-r)
    elif 0 < x <= K and y == (x-1):
        return r
    elif 0 < x <= K and y == (x+1):
        return (1-r)
    else:
        return 0


def bin(k):
    x = binom(K, p).pmf(k)
    return x


def alphaCalc(x0, y):
    alpha = (bin(y)/bin(x0))*(q_distrib(x0, y)/q_distrib(y, x0))
    print(alpha)
    return min(alpha, 1)


def y_gen(x0):
    x = np.random.randint(0, 11)
    return q_distrib(x,x-1)


def mhNextStep(x0):
    y = y_gen(x0)
    alpha = alphaCalc(x0, y)
    u = np.random.random()
    if (u < alpha):
        x1 = y
        a = true
    else:
        x1 = x0
        a = false
    return x1, a


def mhAll(x_start):
    result = list()
    result.append((x_start, true))
    x0 = x_start
    t = 0
    tolérance  = 10**2
    while t < tolérance:
        result.append(mhNextStep(x0))
        x0 = result[-1][0]
        t += 1
    return result

x_start = 5

result = y_gen(x_start)
print(result)
