from math import exp, log
import random
import sys
import numpy as np


def computeN(x, g):
    N = np.zeros(3)
    N[0] = np.sum(g[x == 1][:, x == 2])
    N[1] = np.sum(g[x == 1][:, x == 1])/2
    N[2] = np.sum(g[x == 2][:, x == 2])/2
    return N


def computeOm(x):
    Om = np.zeros(3)
    Om[1] = len(x[x == 1])
    Om[2] = len(x[x == 2])
    return Om


def computeNc(Om, N):
    Nc = np.zeros(3)
    Nc[0] = Om[1]*Om[2] - N[0]
    Nc[1] = (Om[1]*(Om[1]-1)/2) - N[1]
    Nc[2] = (Om[2]*(Om[2]-1)/2) - N[2]
    return Nc


def genX(n):
    x = np.zeros(n)
    for i in range(n):
        x[i] = random.choice([1, 2])
    return x


def newY(x, g, N, Om):
    i = random.randrange(0, len(x))
    newN = np.copy(N)
    newX = np.copy(x)
    newOm = np.copy(Om)
    temp = np.sum(g[i, x[i] == x])
    newN[int(x[i])] -= temp
    newN[0] += temp
    temp = np.sum(g[i][x[i] != x])
    newN[0] -= temp
    newN[1 if x[i] == 2 else 2] += temp
    newX[i] = 1 if x[i] == 2 else 2
    newOm[int(x[i])] -= 1
    newOm[int(newX[i])] += 1
    return newX, newN, newOm


def computePGX(N, Nc, Om, A, B):
    lg = N[0]*log(B) + Nc[0]*log(1-B)
    lg += (N[1] * log(A)) + (Nc[1] * log(1-A))
    lg += (N[2] * log(A)) + (Nc[2] * log(1-A))
    return lg


def computePXPGX( N, Om, A, B):
    Nc = computeNc(Om, N)
    pgx = computePGX(N, Nc, Om, A, B)
    return pgx


def mhNextStep(x, g, Nx, Omx, A, B, pxpgx):
    y, Ny, Omy = newY(x, g, Nx, Omx)
    pypgy = computePXPGX(Ny, Omy, A, B)
    if pypgy > pxpgx:
        return y, pypgy, Ny, Omy
    else:
        u = random.random()
        if log(u) < (pypgy-pxpgx):
            return y, pypgy, Ny, Omy
        else:
            return x, pxpgx, Nx, Omx


def mhAll(g, A, B, nbTests, nbIt):
    n = g.shape[0]
    maxPXPGX = float("-inf")
    bestVector = np.zeros(n)
    for i in range(nbTests):
        print("Test n°", i)
        x = genX(n)
        Nx = computeN(x, g)
        Omx = computeOm(x)
        pxpgx = computePXPGX(Nx, Omx, A, B)
        for j in range(nbIt):
            if j % 10000 == 0:
                print("It n°", j)
            y, pypgy, Ny, Omy = mhNextStep(x, g, Nx, Omx, A, B, pxpgx)
            if pypgy > maxPXPGX:
                bestVector = np.copy(y)
                maxPXPGX = pypgy
            Nx = np.copy(Ny)
            Omx = np.copy(Omy)
            pxpgx = pypgy
            x = np.copy(y)
        print(maxPXPGX)
    return bestVector



file = sys.argv[1]  # paramètres initiaux
nbTests = int(sys.argv[2])
nbIt = int(sys.argv[3])
g = np.load("G.npy")
n = g.shape[0]
p = [0.5, 0.5]
a = 39.76
b = 3.29
A = a/n
B = b/n

x = mhAll(g, A, B, nbTests, nbIt)
np.save('x.npy', np.array(x, dtype=np.int8))
