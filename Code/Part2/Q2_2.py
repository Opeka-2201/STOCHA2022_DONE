import numpy as np
import random

p = [0.5, 0.5]


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


def generate_p_unif(K):
    p = np.zeros(K)
    p += 1/K
    return p


def rand_permutation(y):
    i = random.choice(list(range(len(y))))
    j = random.choice(list(range(len(y))))
    tmp = y[i]
    y[i] = y[j]
    y[j] = tmp
    return y


def change_community(x, K):
    i = random.choices(list(range(K)))
    K_l = list(range(1, K+1))
    K_l.remove(x[i])
    x[i] = random.choices(K_l)[0]
    return x


def concordance(x, y):
    max = 0
    count = 0
    for _ in range(len(x)):
        y = random.shuffle(x)
        for i in range(len(x)):
            if x[i] == y[i]:
                count += 1
        if count > max:
            max = count
        count = 0
    return max / len(x)


N = 10
K = 2
p = generate_p_unif(K)
x, G = generate_SBM(N, K, p, 4, 2)

print(x)
print(G)
