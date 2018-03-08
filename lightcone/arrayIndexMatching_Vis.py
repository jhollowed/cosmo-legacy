import numpy as np
import matplotlib.pyplot as plt

num = [10, 100, 1000, 10000, 30000, 70000, 100000, 300000]
tbru = [2.25e-6, 8.43e-5, 0.00815, 0.16366, 1.332, 7.1576, 14.5901, 131.54]
tbin = [1.52e-5, 0.000207816, 0.00287, 0.006558, 0.02246, 0.0625, 0.0896, 0.311]

fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.loglog(num, tbru, 'o-', color='green', label='brute force')
ax.loglog(num, tbin, 's-', color='cyan', label='binary search')

ax.set_xlabel('N')
ax.set_ylabel('time (s)')

plt.grid()
plt.legend()
plt.show()
