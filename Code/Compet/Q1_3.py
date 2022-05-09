from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

K = 10
p = 0.3
r = 0.1
tolerance  = 600

def q_distrib(y, x):
    if x == 0 and y == 0:
        return r
    elif x == K and y == K:
        return (1-r)
    elif 0 < x <= K and y == (x-1):
        return r
    elif 0 <= x < K and y == (x+1):
        return (1-r)
    else:
        return 0

def y_gen(x0):
    u = np.random.random()
    if x0 == 0 and u <= r:
        return 0
    elif x0 == 0:
        return 1
    elif x0 == K and u <= (1-r):
        return K
    elif x0 == K:
        return K - 1
    elif 0 < x0 < K and u <= r:
        return x0 - 1
    else:
        return x0 + 1

def bin(k):
    x = binom(K, p).pmf(k)
    return x

def alphaCalc(x0, y):
    alpha = (bin(y)/bin(x0))# *(q_distrib(x0, y)/q_distrib(y, x0))
    # print(alpha)
    return min(alpha, 1)

def mhNextStep(x0):
    y = y_gen(x0)
    alpha = alphaCalc(x0, y)
    u = np.random.random()
    if (u < alpha):
        x1 = y
    else:
        x1 = x0
    return x1

def mhAll(x_start):
    result = list()
    result.append(x_start)
    mean = list()
    std = list()
    x0 = x_start
    t = 0
    while t < tolerance:
        result.append(mhNextStep(x0))
        curr_mean = sum(result)/len(result)
        mean.append(curr_mean)
        std.append(sum((x-curr_mean)**2 for x in result) / len(result))
        x0 = result[-1]
        t += 1
    return [result, mean, std]

x_start = 0

# ----- MOYENNE
results = mhAll(x_start)
r = 0.1
plt.plot(results[1])

r = 0.5
results = mhAll(x_start)
plt.plot(results[1], color = 'orange')

plt.legend(['r = 0.1', 'r = 0.5'])
plt.title('Convergence de la moyenne en fonction de r')
plt.show()
# plt.savefig('/Users/arthurlouis/Documents/ULiège/Bachelier/Bloc 3/Q2/Processus Stochastiques/STOCHA2022/Rapport/figs/convergence_mean.png')

#  -----  VARIANCE
tolerance = 2000
results = mhAll(x_start)
r = 0.1
plt.plot(results[2])

r = 0.5
results = mhAll(x_start)
plt.plot(results[2], color = 'orange')

plt.legend(['r = 0.1', 'r = 0.5'])
plt.title('Convergence de la variance en fonction de r')
plt.show()
# plt.savefig('/Users/arthurlouis/Documents/ULiège/Bachelier/Bloc 3/Q2/Processus Stochastiques/STOCHA2022/Rapport/figs/convergence_var.png')

# ----- HISTO
labels, counts = np.unique(results[0], return_counts=True)
plt.bar(labels, np.divide(counts,tolerance), align='center')
plt.gca().set_xticks(labels)
plt.plot(binom(K,p).pmf(range(K+1)), color = 'orange')
plt.title('Histogramme des réalisations')
plt.legend(['Théorique','Pratique'])
plt.show()
# plt.savefig('/Users/arthurlouis/Documents/ULiège/Bachelier/Bloc 3/Q2/Processus Stochastiques/STOCHA2022/Rapport/figs/histo.png')