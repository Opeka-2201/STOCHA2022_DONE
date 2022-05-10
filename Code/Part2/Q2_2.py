from cProfile import label
from math import exp, log
from matplotlib import pyplot as plt
import numpy as np
import random


def generate_SBM(N, K, p, a, b):
    """
    Function to generate a vector of community and an adjacency matrix 
    following a SBM distribution
    Args : N number of vertices
           K number of communities
           p vector of probabilty for communities
           a parametre for the probability of a link between 2 vertices from same community
           b parametre for the probability of a link between 2 vertices from different communities

    return : x a vector of size N with community of each vertices
             G a matrix of size NxN with link between each vertices
    """
    A = a / N
    B = b / N
    if (len(p) != K):
        raise ValueError("p vector must have a size = K")
    x = np.zeros(N)  # creation of vector x
    for i in range(len(x)): 
        x[i] = random.choices(list(range(1, K+1)), p)[0] #assignement of communities for each vertices

    G = np.zeros([N, N])  # creation of matrix G
    for i in range(N):
        for j in range(i+1, N):  # pyramidal link assignement  for every possible couple
            rand = random.random()
            if x[i] == x[j] and rand <= A:
                G[i][j] = 1
                G[j][i] = 1
            elif rand <= B:
                G[i][j] = 1
                G[j][i] = 1
    return x, G


def concordance(x, y):
    """
    Compute concordance between 2 vectors of communities
    To use this function, vectors must have exactly 2 different communites
    return pourcentage of concordance
    """
    count1 = 0
    count2 = 0
    for i in range(len(x)):
        if x[i] == y[i]: #this if statement stand for the permutation
            count1 += 1
        else:
            count2 += 1
    max = count1 if count1 > count2 else count2
    return max / len(x)


def computeN(x, g):
    """
    function to compute the number of link between communities. 
    There are only 3 possibilities
    """
    N = np.zeros(3) #N[0] -> 1-2, N[1] -> 1-1, N[2]->2-2
    N[0] = np.sum(g[x == 1][:, x == 2])
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
    plt.savefig("Report/figs/concordance_a_b.png")
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
    plt.savefig("Report/figs/nij.png")
    plt.show()

graphCommunity()
