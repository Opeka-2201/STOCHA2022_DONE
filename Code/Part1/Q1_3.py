from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

K = 10
p = 0.3
r = 0.1
tolerance  = 600


def y_gen(x0): #making a y candidate from the q distribution given in the statement
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

def bin(k): #binomial distribution calculation
    x = binom(K, p).pmf(k)
    return x

def alphaCalc(x0, y):
    alpha = (bin(y)/bin(x0))# Because of the symetric distribution q, the fraction (q_distrib(x0, y)/q_distrib(y, x0)) is equal to 1 and does not change alpha
    return min(alpha, 1)

def mhNextStep(x0): #An itteration of the MH algorithm literllay like given in the statement too
    y = y_gen(x0)
    alpha = alphaCalc(x0, y)
    u = np.random.random()
    if (u < alpha):
        x1 = y
        a = True
    else:
        x1 = x0
        a = False
    return x1

def mhAll(x_start):
    result = list()
    result.append(x_start)
    mean = list()
    std = list()
    x0 = x_start
    t = 0
    while t < tolerance: # doing multiple iterations (in fonction of the tolerance) with the help of last function mhNextStep
        result.append(mhNextStep(x0))
        curr_mean = sum(result)/len(result) #calculating mean
        mean.append(curr_mean)
        std.append(sum((x-curr_mean)**2 for x in result) / len(result)) #calculating standard deviation
        x0 = result[-1]
        t += 1
    return [result, mean, std]

x_start = 0

# ----- MOYENNE
plt.figure()
results = mhAll(x_start)
r = 0.1
plt.plot(results[1])

r = 0.5
results = mhAll(x_start)
plt.plot(results[1], color = 'orange')

plt.legend(['r = 0.1', 'r = 0.5'])
plt.title('Convergence de la moyenne en fonction de r')
# plt.savefig('Report/figs/convergence_mean.png')
# plt.show()

#  -----  VARIANCE
plt.figure()
tolerance = 2000
results = mhAll(x_start)
r = 0.1
plt.plot(results[2])

r = 0.5
results = mhAll(x_start)
plt.plot(results[2], color = 'orange')

plt.legend(['r = 0.1', 'r = 0.5'])
plt.title('Convergence de la variance en fonction de r')
# plt.savefig('Report/figs/convergence_var.png')
# plt.show()

# ----- HISTO1
plt.figure()
r = 0.1
results = mhAll(x_start)
labels, counts = np.unique(results[0], return_counts=True)
plt.bar(labels, np.divide(counts,tolerance), align='center')
plt.gca().set_xticks(labels)
plt.plot(binom(K,p).pmf(range(K+1)), color = 'orange')
plt.title('Histogramme des réalisations')
plt.legend(['Théorique','Pratique $r=0.1$'])
# plt.savefig('Report/figs/histo1.png')
# plt.show()

# ----- HISTO2
plt.figure()
r = 0.5
results = mhAll(x_start)
labels, counts = np.unique(results[0], return_counts=True)
plt.bar(labels, np.divide(counts,tolerance), align='center')
plt.gca().set_xticks(labels)
plt.plot(binom(K,p).pmf(range(K+1)), color = 'orange')
plt.title('Histogramme des réalisations')
plt.legend(['Théorique','Pratique $r=0.5$'])
# plt.savefig('Report/figs/histo2.png')
# plt.show()