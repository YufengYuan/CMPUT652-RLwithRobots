import matplotlib.pyplot as plt
import csv
import random
import numpy as np
from main import sliding_window

returns = []
with open('saved_returns.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        returns.append(list(map(float, line)))

'''
for i in range(10):
    lines = [random.choice(returns) for _ in range(3)]
    r = np.mean(lines, axis=0)
    plt.plot(sliding_window(r, 100))
plt.title("Episode Return")
plt.xlabel("Episode")
plt.ylabel("Average Return (Sliding Window 100)")
plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
plt.show()

# mean of 3 runs
mean_3_run = []
for _ in range(30):
    lines = [random.choice(returns) for _ in range(3)]
    mean_3_run.append(np.mean(lines, axis=0))
m = np.mean(mean_3_run, axis=0)
s = np.std(mean_3_run, axis=0)
plt.plot(sliding_window(m, 100))
plt.fill_between(np.arange(5000), sliding_window(m+s, 100), sliding_window(m-s,100), alpha=0.4)

mean_10_run = []
for _ in range(30):
    lines = [random.choice(returns) for _ in range(10)]
    mean_10_run.append(np.mean(lines, axis=0))
m = np.mean(mean_10_run, axis=0)
s = np.std(mean_10_run, axis=0)
plt.plot(sliding_window(m, 100))
plt.fill_between(np.arange(5000), sliding_window(m+s,100), sliding_window(m-s,100), alpha=0.4)

mean_30_run = []
m = np.mean(mean_10_run, axis=0)
s = np.std(mean_10_run, axis=0)
plt.plot(sliding_window(m, 100))
#plt.plot(m+s)
#plt.plot(m-s)
#plt.fill_between(np.arange(5000), sliding_window(m+s,100), sliding_window(m-s,100), alpha=0.6)

plt.legend(["3 runs", "10 runs", "30 runs"])
plt.title("Episode Return")
plt.xlabel("Episode")
plt.ylabel("Average Return (Sliding Window 100)")
'''

# gamma correction comparison

returns_ = []
with open('returns.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        returns_.append(list(map(float, line)))

m = np.mean(returns, axis=0)
s = np.std(returns, axis=0)
m_ = np.mean(returns_, axis=0)
s_ = np.std(returns_, axis=0)

plt.plot(sliding_window(m, 100))
plt.fill_between(np.arange(5000), sliding_window(m+s, 100), sliding_window(m-s, 100), alpha=0.4)
plt.plot(sliding_window(m_, 100))
plt.fill_between(np.arange(5000), sliding_window(m_+s_, 100), sliding_window(m_-s_, 100), alpha=0.4)

plt.legend(["Without Gamma Correction", "With Gamma Correction"])
plt.title("Episode Return")
plt.xlabel("Episode")
plt.ylabel("Average Return With or Without Correction on Gamma")


plt.show()











