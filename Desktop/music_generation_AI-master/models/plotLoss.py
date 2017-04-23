import matplotlib.pyplot as plt
import numpy as np

file = "ds:_1__arch:_200x200x__act:_uhh__loss_trend"

lossList = np.loadtxt(file)

plt.plot(lossList)
plt.show()