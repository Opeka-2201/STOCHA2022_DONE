from inspect import Parameter
from sqlite3 import paramstyle
from statistics import variance
from turtle import pos
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import math
import statistics
from scipy.stats import binom

x = np.load('X_1000.npy')
G = np.load('G_1000.npy')


def generate_y(x):  # distrib qs

    # communities = range(1, K)
    # random_index = x[random.randint(0, len(x)-1)] #noeud choisi au hasard
    # possible_choices = [v for v in communities if v != x[random_index]]
    # choice = np.random.multinomial(1, possible_choices,size=None)
    # chosen_y = np.where(choice == 1)
    # y = chosen_y[0] + 1
    random_index = random.randrange(0, len(x))
    if x[random_index] == 1:
        y = 2
    else:
        y = 1

    return y, random_index


def metropolis_hastings(x, convergence_time):
    time = 0
    p = np.array([0.5, 0.5])

    while time <= convergence_time:
        #print ('notre state liste :', state_list)
        y, rand_i = generate_y(x)

        #print('y=',y)
        ones_x = x[x == 1]
        nbr_one_x = len(ones_x)
        #print(nbr_one_x)

        twos_x = x[x == 2]
        nbr_two_x = len(twos_x)
        #print(nbr_two_x)

        new_nbr_one = nbr_one_x
        new_nbr_two = nbr_two_x

        if y == 1:
            new_nbr_one += 1
            new_nbr_two -= 1
        else:
            new_nbr_one -= 1
            new_nbr_two += 1

        #print(new_nbr_one)
        #print(new_nbr_two)

        # pas de 2e terme car la probabilité que le noeud change de communauté est tjrs de 1
        alpha = (p[0] ** (nbr_one_x) * p[1] ** (nbr_two_x)) / \
            (p[0] ** (new_nbr_one) * p[1] ** (new_nbr_two))
        #print(alpha)
        u = np.random.uniform()
        if u < alpha:
            x[rand_i] = y
        time += 1

    return x, concordance


def generate_graph(N, K, p, a, b):
    A = a/N
    B = b/N
    vertices_proba = np.zeros(K)
    communities = range(1, K+1)  # (1 2)
    G = np.zeros((N, N))
    commu_vector = np.zeros(N)
    for k in range(0, N):
        # x : vecteur donnant la commu de chaque noeud
        commu_vector[k] = random.choices(communities, p)[0]

    print(commu_vector)
    # for i in range(0, K):
    #     for j in range(0, K):
    #         if(i == j):
    #             W[i][j] = A
    #         else:
    #             W[i][j] = B
    vertices = range(0, 2)  # (0 1) : 0 si pas d'arrête et 1 si arrête
    for i in range(0, N):
        for j in range(0, N):
            if(x[i] == x[j]):  # noeuds i et j ont la même commu
                vertices_proba[0] = 1-A  # proba qu'il y ait pas d'arrête
                vertices_proba[1] = A  # proba qu'il y ait une arrête
                G[i][j] = 0
            else:
                vertices_proba[0] = 1-B
                vertices_proba[1] = B
                G[i][j] = random.choices(vertices, vertices_proba)[0]

    return commu_vector, G


# def concordance(x_star,x, N, K)

#     for pi in range Sk:
#         for i in range (1, N):

# proba = np.array([0.5, 0.5])
# print(generate_graph(10, 2, proba, 4, 2"')[0])

# metropolis_hastings(x, 100)

# for N_G in range :
#     generate_graph()
#     for T in range :
#         metropolis_hastings(x,T)
#         concor[T] = concordance()
#         # faire la moyenne de la concordance : moy = sum(concor)/T
#     # faire la moyenne de la moyenne : concordance_moyenne = sum(moy)/N_G
# return concordance_moyenne
