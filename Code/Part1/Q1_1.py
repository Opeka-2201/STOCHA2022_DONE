import numpy as np
import matplotlib.pyplot as plt
import random
print("Question 1 : Chaînes de Markov")

print("\n### Q1 ###")

time_limit = 20

Q = np.array([[0.0, 0.1, 0.1, 0.8],
              [1.0, 0.0, 0.0, 0.0],
              [0.6, 0.0, 0.1, 0.3],
              [0.4, 0.1, 0.5, 0.0]])  # transition matrix

p_1 = np.array([0.25, 0.25, 0.25, 0.25])  # uniformly distributed
p_2 = np.array([0.0, 0.0, 1.0, 0.0])  # base case always 3
p_1_temp = p_1
p_2_temp = p_2

P_1 = np.zeros([4, time_limit])
P_2 = np.zeros([4, time_limit])

for i in range(time_limit):
    for j in range(4):
        P_1[j, i] = p_1_temp[j]
        P_2[j, i] = p_2_temp[j]

    p_1_temp = np.matmul(p_1_temp, Q)
    p_2_temp = np.matmul(p_1_temp, Q)

plt.figure()
plt.plot(P_1[0, :])
plt.plot(P_1[1, :])
plt.plot(P_1[2, :])
plt.plot(P_1[3, :])
plt.title("Évolution de la probabilité de $P(X_t = x)$ dans un cas de base uniforme")
plt.legend(["$P(X_t = 1)$", "$P(X_t = 2)$", "$P(X_t = 3)$", "$P(X_t = 4)$"])
plt.xlabel("$t$")
plt.ylabel("Probability")
plt.savefig("Report/figs/evo_unif.png")
# plt.show()

plt.figure()
plt.plot(P_2[0, :])
plt.plot(P_2[1, :])
plt.plot(P_2[2, :])
plt.plot(P_2[3, :])
plt.title("Évolution de la probabilité de $P(X_t = x)$ dans un cas de base fixé sur 3")
plt.legend(["$P(X_t = 1)$", "$P(X_t = 2)$", "$P(X_t = 3)$", "$P(X_t = 4)$"])
plt.xlabel("$t$")
plt.ylabel("Probability")
plt.savefig("Report/figs/evo_fixed.png")
# plt.show()

print("\nUniformly distributed case :")
print(" P(X_t = 1) = %s" % (P_1[0, time_limit-1]))
print(" P(X_t = 2) = %s" % (P_1[1, time_limit-1]))
print(" P(X_t = 3) = %s" % (P_1[2, time_limit-1]))
print(" P(X_t = 4) = %s" % (P_1[3, time_limit-1]))

print("\nFixed case :")
print(" P(X_t = 1) = %s" % (P_2[0, time_limit-1]))
print(" P(X_t = 2) = %s" % (P_2[1, time_limit-1]))
print(" P(X_t = 3) = %s" % (P_2[2, time_limit-1]))
print(" P(X_t = 4) = %s" % (P_2[3, time_limit-1]))

print("\nQ^t =")
Q_inf = np.linalg.matrix_power(Q, time_limit)
print(Q_inf)


print("\n### Q2 ###")

print("1e condition", np.matmul(p_1, Q_inf))
print("2e condition", np.matmul(p_2, Q_inf))

print("\n### Q1_3 ###")
p = np.cumsum(p_1)

def nextState(p):
    rand = random.random()
    p = np.cumsum(p)
    if 0 < rand < p[0]:
        return 1
    elif p[0] < rand < p[1]:
        return 2
    elif p[1] < rand < p[2]:
        return 3
    elif p[2] < rand < p[3]:
        return 4

def makeRealisation(x0, T):
    real = list()
    real.append(x0)
    for _ in range(1, T):
        real.append(nextState(Q[real[-1]-1]))
    return real

def numberOfOccurences(real):
    occurence = np.zeros(4)
    for i in real:
        occurence[i - 1] += 1
    return occurence / len(real)

for nb_real in [100,1000,1000,10000,100000]:
  mk = makeRealisation(nextState([0.25,0.25,0.25,0.25]),nb_real)
  print(numberOfOccurences(mk))

sum = np.zeros([4,10000])
mk = np.zeros([4,10000])

state = nextState([0.25,0.25,0.25,0.25])

for i in range(10000):
  mk[state-1][i] = 1
  indices = list()
  for j in range(0,i+1):
    indices.append(j)

  sum[0][i] = mk[0][indices].sum() / (i+1)
  sum[1][i] = mk[1][indices].sum() / (i+1)
  sum[2][i] = mk[2][indices].sum() / (i+1)
  sum[3][i] = mk[3][indices].sum() / (i+1)

  state = nextState(Q[state-1])

plt.figure()
plt.plot(sum[0],'red')
plt.title("Évolution de la proportion d'états $P(X_t = 1)$")
plt.xlabel("$t$")
# plt.savefig('Report/figs/evo1.png')
# plt.show()

plt.figure()
plt.plot(sum[1],'blue')
plt.title("Évolution de la proportion d'états $P(X_t = 2)$")
plt.xlabel("$t$")
# plt.savefig('Report/figs/evo2.png')
# plt.show()

plt.figure()
plt.plot(sum[2],'orange')
plt.title("Évolution de la proportion d'états $P(X_t = 3)$")
plt.xlabel("$t$")
# plt.savefig('Report/figs/evo3.png')
# plt.show()

plt.figure()
plt.plot(sum[3],'green')
plt.title("Évolution de la proportion d'états $P(X_t = 4)$")
plt.xlabel("$t$")
# plt.savefig('Report/figs/evo4.png')
# plt.show()