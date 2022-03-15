import numpy as np
import matplotlib as plt

Q = np.array([[0.0,0.1,0.1,0.8],
              [1.0,0.0,0.0,0.0],
              [0.6,0.0,0.1,0.3],
              [0.4,0.1,0.5,0.0]])

p1_1 = np.array([0.25,0.25,0.25,0.25])
p1_2 = np.array([0.0, 0.0, 1.0, 0.0])


def t_croissant(Q,p,t):
    Q_n = np.linalg.matrix_power(Q,t)
    result = np.zeros((50,4))
    for i in range(1,51):
        result[i-1,:]=t_croissant(Q,p1_1,i)
    return np.matmul(p,Q_n)

print(result)