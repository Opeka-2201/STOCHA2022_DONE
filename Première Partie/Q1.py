import numpy as np
import matplotlib.pyplot as plt
print("Question 1 : Chaînes de Markov")

print("\n### Q1 ###")

time_limit = 20

Q = np.array([[0.0,0.1,0.1,0.8],
              [1.0,0.0,0.0,0.0],
              [0.6,0.0,0.1,0.3],
              [0.4,0.1,0.5,0.0]]) # transition matrix

p_1 = np.array([0.25,0.25,0.25,0.25]) # uniformly distributed
p_2 = np.array([0.0, 0.0, 1.0, 0.0]) # base case always 3
p_1_temp = p_1
p_2_temp = p_2

P_1 = np.zeros([4,time_limit])
P_2 = np.zeros([4,time_limit])

for i in range(time_limit):
  for j in range(4):
    P_1[j,i] = p_1_temp[j]
    P_2[j,i] = p_2_temp[j]

  p_1_temp = np.matmul(p_1_temp,Q)
  p_2_temp = np.matmul(p_1_temp,Q)

plt.plot(P_1[0,:])
plt.plot(P_1[1,:])
plt.plot(P_1[2,:])
plt.plot(P_1[3,:])
plt.title("Évolution de la probabilité de P(X_t = x) dans un cas de base uniforme")
plt.legend(["P(X_t = 1)","P(X_t = 2)","P(X_t = 3)","P(X_t = 4)"])
plt.xlabel("t")
plt.ylabel("Probability")
plt.show()

plt.plot(P_2[0,:])
plt.plot(P_2[1,:])
plt.plot(P_2[2,:])
plt.plot(P_2[3,:])
plt.title("Évolution de la probabilité de P(X_t = x) dans un cas de base fixé sur 3")
plt.legend(["P(X_t = 1)","P(X_t = 2)","P(X_t = 3)","P(X_t = 4)"])
plt.xlabel("t")
plt.ylabel("Probability")
plt.show()

print("\nUniformly distributed case :")
print(" P(X_t = 1) = %s" %(P_1[0,time_limit-1]))
print(" P(X_t = 2) = %s" %(P_1[1,time_limit-1]))
print(" P(X_t = 3) = %s" %(P_1[2,time_limit-1]))
print(" P(X_t = 4) = %s" %(P_1[3,time_limit-1]))

print("\nFixed case :")
print(" P(X_t = 1) = %s" %(P_2[0,time_limit-1]))
print(" P(X_t = 2) = %s" %(P_2[1,time_limit-1]))
print(" P(X_t = 3) = %s" %(P_2[2,time_limit-1]))
print(" P(X_t = 4) = %s" %(P_2[3,time_limit-1]))

print("\nQ^t =")
print(np.linalg.matrix_power(Q,time_limit))

print("\n### Q2 ###")

