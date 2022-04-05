import numpy as np
from sympy import false, true

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
    return np.random.binomial(K, p, k)


def alphaCalc(x0, y):
    alpha = (bin(y)/bin(x0))*(q_distrib(x0, y)/q_distrib(y, x0))
    return min(alpha, 1)


def y_gen(x0):
    return np.random.random()  # TODO


def convCheck():
    return false  # TODO


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


def mhAll(x0):
    result = list()
    result.append((x0, true))
    conv = true
    while not conv:
        result.append(mhNextStep(x0))
        x0 = result[-1][0]
        conv = convCheck()
    return result
