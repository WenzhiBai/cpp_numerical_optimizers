import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("error_log.txt")
plt.figure(figsize=(16, 10))
plt.plot(data[:, 0], data[:, 1], color='blue', linewidth=1, alpha=0.8)
plt.scatter(data[:, 0], data[:, 1], color='red', s=10, label='Error per Iteration')
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("Optimization Error vs Iteration")
plt.grid(True)
plt.show()
·