import numpy as np
import matplotlib.pyplot as plt

def q_distrib(Y, sachX, r, K):
    if Y == 0 and sachX == 0:
        return r
    elif sachX == K and Y == K:
        return 1-r
    elif 0 < sachX <= K and Y == sachX-1:
        return r
    elif 0 < sachX <= K and Y == sachX+1:
        return 1-r
    else:
        return 0

def metropolis_hastings_etape(xt_1,r):
    K=10
    yt = q_distrib(np.random.binomial(K,0.3),xt_1,r,K)
    alpha = np.min(1, binomial(yt) * q_distrib(xt_1, yt, r, K) / (binomial(xt_1) * q_distrib(yt,xt_1, r, K)))
    u = np.random.uniform(0,1)
    if(u < alpha):
        x1 = yt
    else:
        x1 = xt_1
    return x1
x = 10
for i in range(100):
    x = metropolis_hastings_etape(x,0.1)

def binomial(k):
    return np.random.binomial(10,0.3,k)
