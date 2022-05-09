from math import exp, log
import sys
import numpy as np
import random
import time

file = sys.argv[1]


G = np.load(file)
n = G.shape[0]
p = [0.5, 0.5]
a = 39.76
b = 3.29
A = a/n
B = b/n


def genX(n, p):
    x = np.zeros(n)
    for i in range(n):
        x[i] = random.choices([1, 2], p)[0]
    return x


def computeOm(x):
    Om = np.zeros(2)
    Om[0] = np.sum(x[x == 1])
    Om[1] = np.sum(x[x == 2])
    return Om


def computepx(Om, p):
    return (Om[0]*log(p[0]))+(Om[1]*log(p[1]))


def computeN(x, G, n):
    N = np.zeros(3)  # N[0] == 1-2 N[1] == 1-1 N[2] == 2-2
    N[0] = np.sum(G[x == 1][:, x == 2])
    N[1] = np.sum(G[x == 1][:, x == 1])
    N[2] = np.sum(G[x == 2][:, x == 2])
    return N


def computeNc(Om, N):
    Nc = np.zeros(3)  # N[0] == 1-2 N[1] == 1-1 N[2] == 2-2
    Nc[0] = Om[0]*Om[1] - N[0]
    Nc[1] = Om[0]*((Om[0]-1)/2) - N[1]
    Nc[2] = Om[1]*((Om[1]-1)/2) - N[2]
    return Nc


def computepgx(N, Nc, A, B):
    lg = N[0]*log(B)+(Nc[0]*log(1-B))
    lg += (N[1]*log(A))+(Nc[1]*log(1-A))
    lg += N[2]*log(A)+(Nc[2]*log(1-A))
    return lg


def computepxpgx(x, p, A, B, N):
    Om = computeOm(x)
    px = computepx(Om, p)
    Nc = computeNc(Om, N)
    pgx = computepgx(N, Nc, A, B)
    return px+pgx


def genNextX(x, N):
    rnd = random.randrange(0,len(x))
    N[0] -= np.sum(G[rnd][x!=x[rnd]])
    N[int(x[rnd])] -= np.sum(G[rnd][x == x[rnd]])
    x[rnd] = 1 if x[rnd] == 2 else 2
    N[0] += np.sum(G[rnd][x!=x[rnd]])
    N[int(x[rnd])] += np.sum(G[rnd][x == x[rnd]])
    return x, N


def mhNextStep(x, G, A, B, pxpgx, Nx):
    y ,Ny = genNextX(x,Nx)
    pypgy = computepxpgx(y, p, A, B, Ny)

    alpha = pypgy-pxpgx
    u = np.random.random()
    accept = True
    if (u > exp(alpha)):
        y = x
        pypgy = pxpgx
        accept = False
        Ny = Nx
    return y, pypgy, accept, Ny


def mhAll(G, A, B, p):
    n = G.shape[0]
    maxPxpgx = float("-inf")
    bestVector = np.zeros(n)
    for i in range(5):
        print("max =", maxPxpgx)
        changes = 0
        x = genX(n, p)
        Nx = computeN(x,G,n)
        pxpgx = computepxpgx(x, p, A, B, Nx)
        if maxPxpgx<pxpgx:
            maxPxpgx = pxpgx
            bestVector = x
        while changes < 10000:
            #if changes%1000 == 0:    
            y, pypgy, accept, Ny = mhNextStep(x, G, A, B, pxpgx,Nx)
            changes += 1
            if accept:
                
                if pypgy > maxPxpgx:
                    maxPxpgx = pypgy
                    bestVector = y
            x = y
            pxpgx = pypgy
            Nx = Ny
    return bestVector


