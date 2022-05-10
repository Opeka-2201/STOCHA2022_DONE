from cProfile import label
from math import exp, log
from matplotlib import pyplot as plt
import numpy as np
import random

def generate_SBM(N, K, p, a, b):
    A = a / N
    B = b / N
    if (len(p) != K):
        raise ValueError("p vector must have a size = K")
    x = np.zeros(N)
    for i in range(len(x)):
        x[i] = random.choices(list(range(1, K+1)), p)[0]

    G = np.zeros([N, N])
    for i in range(N):
        for j in range(i+1, N):
            rand = random.random()
            if x[i] == x[j] and rand <= A:
                G[i][j] = 1
                G[j][i] = 1
            elif rand <= B:
                G[i][j] = 1
                G[j][i] = 1
    return x, G


def concordance(x, y):
    count1 = 0
    count2 = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            count1 += 1
        else:
            count2 += 1
    max = count1 if count1 > count2 else count2
    return max / len(x)


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


def computePXPGX(N, Om, A, B):
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
        x = genX(n)
        Nx = computeN(x, g)
        Omx = computeOm(x)
        pxpgx = computePXPGX(Nx, Omx, A, B)
        for j in range(nbIt):
            y, pypgy, Ny, Omy = mhNextStep(x, g, Nx, Omx, A, B, pxpgx)
            if pypgy > maxPXPGX:
                bestVector = np.copy(y)
                maxPXPGX = pypgy
            Nx = np.copy(Ny)
            Omx = np.copy(Omy)
            pxpgx = pypgy
            x = np.copy(y)
    return bestVector

def graphB_A():
    p = [0.5,0.5]
    a = 5.97
    n = 500
    result = np.zeros((99,2))
    for i in range(99):
        print("i:",i)
        b = 6-a
        x, g = generate_SBM(n, 2, p, a, b)
        conc = 0
        for _ in range(10):
            y = mhAll(g,a/n,b/n,1,10000)
            conc += concordance(x,y)/10
        result[i] = (b/a,conc)
        a -= 0.03


    plt.figure()
    plt.plot(result[:,0],result[:,1])
    plt.title("Evolution de la concordance en fonction du rapport b/a")
    plt.xlabel("b/a")
    plt.ylabel("Concordance")
    plt.savefig("concordance_a_b.svg")
    plt.show()


def graphCommunity():
    p = [0.5, 0.5]
    a = 5.97
    n = 500
    result = np.zeros((99, 4))
    for i in range(99):
        b = 6-a
        print("i:", i, "  b/a:", (b/a))
        temp = [0,0,0,0]
        for _ in range(10):
            x, g = generate_SBM(n, 2, p, a, b)
            N = computeN(x,g)
            temp[0] += N[0]/10
            temp[1] += N[1]/10
            temp[2] += N[2]/10
        result[i] = (b/a, temp[0], temp[1], temp[2])
        a -= 0.03
    plt.figure()
    plt.plot(result[:, 0], result[:, 1], label = "$N_{1,2}$")
    plt.plot(result[:, 0], result[:, 2], label = "$N_{1,1}$")
    plt.plot(result[:, 0], result[:, 3], label= " $N_{2,2}$")
    plt.legend()

    plt.title("Evolution des des liens entre communeautés en fonction du rapport b/a")
    plt.xlabel("b/a")
    plt.ylabel("$N_{ij}$")
    plt.savefig("/nij.png")
    plt.show()

graphCommunity()
