from math import exp, log
import sys
import numpy as np
import random
import time

from sqlalchemy import false, true

file = sys.argv[1]  # paramètres initiaux
nbTests = int(sys.argv[2])
nbIt = int(sys.argv[3])
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


def computeN(x, G):
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


def computepx(Om, p):
    lg = Om[0]*log(p[0])
    lg += Om[1]*log(p[1])
    return lg


def computepgx(N, Nc, A, B):
    lg = N[0]*log(B)+Nc[0]*log(1-B)
    lg += (N[1]+N[2]) * log(A) + (Nc[1]+Nc[2])*log(1-A)
    return lg


def computepxpgx(x, p, A, B, N, Om):
    px = computepx(Om, p)
    Nc = computeNc(Om, N)
    pgx = computepgx(N, Nc, A, B)
    return px + pgx


def genNextX(x, N, G, Om):
    rnd = random.randrange(0, len(x))
    tmp = np.sum(G[rnd][int(x[rnd]) == x]) #mise a jour de N
    N[int(x[rnd])] -= tmp
    N[0] += tmp
    tmp = np.sum(G[rnd][int(x[rnd]) != x])
    N[0] -= tmp
    N[1 if int(x[rnd]) == 2 else 2] += tmp
    x[rnd] = 1 if int(x[rnd]) == 2 else 2 #swap dans le vecteur x
    if x[rnd] == 1:
        Om[0]+=1
        Om[1]-=1
    else:
        Om[0]-=1
        Om[1]+=1
    return x, N, Om


def mhNextStep(x ,G ,Nx,Omx ,p,A,B,pxpgx):
    y ,Ny ,Omy= genNextX(x, Nx, G, Omx)
    pypgy = computepxpgx(y, p, A, B, Ny, Omy)
    alpha = pypgy-pxpgx
    alpha = exp(alpha)
    u = random.random()
    if (u < alpha):
        y = x
        pypgy = pxpgx
        Ny = Nx 
        Omy = Omx       
    return y , pypgy, Ny, Omy

def mhAll(G, A, B, p, nbTests, nbIt):
    n = G.shape[0]
    maxPxpgx = float("-inf")
    bestVector = np.zeros(n)
    for i in range(nbTests):
        print("TestNb : ",i)
        x = genX(n,p)
        Nx = computeN(x,G)
        Omx = computeOm(x)
        pxpgx = computepxpgx(x,p,A,B,Nx,Omx)
        for j in range(nbIt):
            if j%10000 == 0:
                print("ItNb : ",j)
            y, pypgy, Ny, Omy = mhNextStep(x,G,Nx,Omx,p,A,B,pxpgx)
            if pypgy > maxPxpgx:
                maxPxpgx = pypgy
                bestVector = y
            x = y
            pxpgx = pypgy
            Nx = Ny
            Omx = Omy
    return bestVector


x = mhAll(G, A, B, p, nbTests, nbIt)
np.save('x.npy', np.array(x, dtype=np.int8))
