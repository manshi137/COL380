import subprocess
import time
import matplotlib.pyplot as plt
serial = [1, 2, 3, 4, 5]
pthreads = [5, 4, 3, 2, 1]
n = [10, 100, 400, 700, 1400]
plt.plot(n, serial, label='Serial')
plt.plot(n, pthreads, label='Pthreads')
plt.xlabel('n')
plt.ylabel('Time')
plt.title('Serial vs Pthreads')
plt.legend()
plt.show()

# i want to plot serial time and parallel time v/s n using matplotlib
# i have the serial time and parallel time in a list
# i have the n values in another list





