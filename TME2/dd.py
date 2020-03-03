import numpy as np
import matplotlib.pyplot as plt
a = np.array([2,3,4])
print(np.zeros((2,3)).shape)
print(a.shape)
b = 0.5*a**2
c = 0.5*a.T*a
print(b)
print(c)
fac = (1/np.sqrt(2*np.pi))
print(fac)
print()
print(fac*np.exp(-0.5*1**2))
print(fac*np.exp(-0.5*0.7**2))
print(fac*np.exp(-0.5*0.5**2))
print(fac*np.exp(-0.5*0.1**2))
print(fac*np.exp(-0.5*0**2))

g = 0.01/0.015
print("g")
print(g)
h = fac*np.exp(-0.5*g**2)
print(h)
print(h/(500 * 0.015))


l = np.linspace(-10,10,50)
y = [fac*np.exp(-0.5*i**2) for i in l]
print(y)
plt.plot(l,y)
plt.show()