import numpy as np
import matplotlib.pyplot as plt

# discount rate
gamma = 0.9

# expected rewards
r_sa = np.zeros((2, 2))
r_sa[1, 1] = 1.0

# transition dynamics
p_sasp = np.zeros((2, 2, 2))
p_sasp[0, 0, 0] = 1.0
p_sasp[0, 1, 1] = 1.0
p_sasp[1, 0, 0] = 1.0
p_sasp[1, 1, 1] = 1.0

# policy
policy = np.ones((2, 2)) * 0.5

# compute r_pi
r_pi = np.zeros((2, 1))
for s in range(2):
  r_pi[s] = np.dot(policy[s], r_sa[s])
print('r_pi:')
print(r_pi)

# compute P_pi
P_pi = np.zeros((2, 2))
for s in range(2):
  for sp in range(2):
    P_pi[s, sp] = np.dot(policy[s], p_sasp[s, :, sp])
print('P_pi:')
print(P_pi)

# A and b matrices
A = np.eye(2) - gamma * P_pi
b = r_pi

# v = A^(-1)b
v = np.matmul(np.linalg.inv(A), b)
print('v_pi')
print(v)