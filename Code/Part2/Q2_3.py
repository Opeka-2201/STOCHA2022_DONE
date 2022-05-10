from math import exp, log
import random
import sys
import numpy as np


def computeN(x, g):
    """
    function to compute the number of link between communities. 
    There are only 3 possibilities
    """
    N = np.zeros(3)
    N[0] = np.sum(g[x == 1][:, x == 2]) #N[0] -> 1-2, N[1] -> 1-1, N[2]->2-2
    N[1] = np.sum(g[x == 1][:, x == 1])/2
    N[2] = np.sum(g[x == 2][:, x == 2])/2
    return N


def computeOm(x):
    """
    function to compute numbers of vertices in each community
    We only need this functionnality for 2 communities
    """
    Om = np.zeros(3) #it's a size 3 vector to be more intuitive in the use
    Om[1] = len(x[x == 1])
    Om[2] = len(x[x == 2])
    return Om


def computeNc(Om, N):
    """
    function to compute the number of non-link between communities. 
    There are only 3 possibilities
    """
    Nc = np.zeros(3)  # Nc[0] -> 1-2, Nc[1] -> 1-1, Nc[2]->2-2
    Nc[0] = Om[1]*Om[2] - N[0]
    Nc[1] = (Om[1]*(Om[1]-1)/2) - N[1]
    Nc[2] = (Om[2]*(Om[2]-1)/2) - N[2]
    return Nc


def genX(n):
    """
    function to create a random vector of communities. 
    We only need this functionnality for 2 communities and with uniform probability
    """
    x = np.zeros(n)
    for i in range(n):
        x[i] = random.choice([1, 2])
    return x


def newY(x, g, N, Om):
    """
    function to change the community of a random vertice and 
    adjust the N and Om variables to improve computation time
    """
    i = random.randrange(0, len(x))
    newN = np.copy(N)
    newX = np.copy(x)
    newOm = np.copy(Om)
    temp = np.sum(g[i, x[i] == x])  # reassigne every edges linked with the node i 
    newN[int(x[i])] -= temp
    newN[0] += temp
    temp = np.sum(g[i][x[i] != x])
    newN[0] -= temp
    newN[1 if x[i] == 2 else 2] += temp
    newX[i] = 1 if x[i] == 2 else 2 # change the vertice i community
    newOm[int(x[i])] -= 1 # update the count of vertices in each community
    newOm[int(newX[i])] += 1
    return newX, newN, newOm


def computePGX(N, Nc, Om, A, B):
    """
    function to compute the logarithm of P(G|x)
    """
    lg = N[0]*log(B) + Nc[0]*log(1-B)
    lg += (N[1] * log(A)) + (Nc[1] * log(1-A))
    lg += (N[2] * log(A)) + (Nc[2] * log(1-A))
    return lg


def computePXPGX(N, Om, A, B):
    """
    function to compute the logarithm of (P(G|x) * P(x))
    P(x) is a constant and so not interessant for the maximization
    This not usefull to compute it 
    """
    Nc = computeNc(Om, N)
    pgx = computePGX(N, Nc, Om, A, B)
    return pgx


def mhNextStep(x, g, Nx, Omx, A, B, pxpgx):
    """
    function to compute an iteration of the Metropolis-Hastingss algorithm
    """
    y, Ny, Omy = newY(x, g, Nx, Omx)
    pypgy = computePXPGX(Ny, Omy, A, B)
    if pypgy > pxpgx:  #accept if the new vector have better probabilty
        return y, pypgy, Ny, Omy
    else: 
        u = random.random()
        if log(u) < (pypgy-pxpgx): #accept with the acceptance alpha
            return y, pypgy, Ny, Omy
        else:  #reject the new vector
            return x, pxpgx, Nx, Omx


def mhAll(g, A, B, nbTests, nbIt):
    """
    function to compute nbTests times the Metropolis-Hastings algorithm with nbIt itération
    return the best vector of all tests and all itérations
    """
    n = g.shape[0]
    maxPXPGX = float("-inf")
    bestVector = np.zeros(n)
    for _ in range(nbTests): #To avoid local minimum
        x = genX(n)
        Nx = computeN(x, g)
        Omx = computeOm(x)
        pxpgx = computePXPGX(Nx, Omx, A, B)
        for _ in range(nbIt): #classical algorithm 
            x, pxpgx, Nx, Omx = mhNextStep(x, g, Nx, Omx, A, B, pxpgx)
            if pxpgx > maxPXPGX:
                bestVector = x
                maxPXPGX = pxpgx
    return bestVector



file = sys.argv[1]  # paramètres initiaux
nbTests = int(sys.argv[2])
nbIt = int(sys.argv[3])
g = np.load(file)
n = g.shape[0]
p = [0.5, 0.5]
a = 39.76
b = 3.29
A = a/n
B = b/n

x = mhAll(g, A, B, nbTests, nbIt)
np.save('Code/Part2/x.npy', np.array(x, dtype=np.int8))
