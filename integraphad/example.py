import numpy as np
import csdl_alpha as csdl
import matplotlib.pyplot as plt
from integraphad import Integrators as int

## Solving Lotka-Volterra coupled ODE system using numerical integration

def lotkavolterra(t, y, a, b, c, d):
    dy1dt = a * y[0] - b * y[0] * y[1]
    dy2dt = -c * y[1] + d * y[0] * y[1]
    return csdl.vstack([dy1dt, dy2dt]).flatten()

rec = csdl.Recorder(inline=False)
rec.start()

initCond = np.array([10, 5])
timeInt = np.array([0, 15])
a = 1.5
b = 1
c = 3
d = 1

t, y = int.solve(lotkavolterra, timeInt, initCond, a, b, c, d, num_steps=100, method='rk4') # Method can be changed to use other numerical integrators

rec.execute()

plt.plot(t.value, y.value[:, 0], color='blue', label='y1(t) - RK4')
plt.plot(t.value, y.value[:, 1], color='red', label='y2(t) - RK4')
plt.title('Lotka-Volterra Solution Using RK4 Integration')
plt.legend()
plt.show()
