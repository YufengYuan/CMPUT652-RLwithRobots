import numpy as np
import matplotlib.pyplot as plt

# seed for reproducibility
np.random.seed(652)

# exp params
gamma = 0.9
n_runs = 1000
n_steps = 100

# to store stuff
values = np.zeros((2, n_runs + 1))
for s0 in range(2):
  for run in range(n_runs):
    # init state and return
    s = s0
    G = 0.0
    for step in range(n_steps):
      # choose action
      a = np.random.randint(2)
      # env dynamics
      if (s == 0 and a == 0) or (s == 1 and a == 0):
        sp, r = 0, 0
      elif (s == 0 and a == 1):
        sp, r = 1, 0
      else: #(s == 1 and a == 1):
        sp, r = 1, 1
      # accumulate return
      G += r * (gamma ** step)
      # next time step
      s = sp
    # average results
    values[s0, run + 1] = values[s0, run] + (1 / (run + 1)) * (G - values[s0, run])

# print final values for each state
print(values[:, -1].T)

# plot results
for s in range(2):
  plt.plot(np.arange(n_runs + 1), values[s], label='State {}'.format(s))
plt.axis([0 - 0.01 * n_runs, n_runs, 0, 1.01 * np.max(values)])
plt.xlabel('Number of Runs')
plt.ylabel('Estimated Return from Start State')
plt.legend()
plt.show()