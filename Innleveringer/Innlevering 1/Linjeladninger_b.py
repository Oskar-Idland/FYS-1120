# b)

import numpy as np
import matplotlib.pyplot as plt

def efieldq(q0,r,r0):
    # Input: charge q in Coulomb
    #        r: position to find field (in 1,2 or 3 dimensions) in meters
    #        r0: position of charge q0 in meters
    # Output: electric field E at position r in N/C
    dr = r-r0
    drnorm = np.sqrt(dr.dot(dr))
    epsilon0 = 8.854187817e-12
    return q0/(2.0*np.pi*epsilon0*(drnorm)**3)*dr 

q = 1 # Simplified Q/L
L = 5
N = 30
x = np.linspace(-L,L,N)
y = np.linspace(-L + 3,L + 3,N)
rx,ry = np.meshgrid(x,y)
Ex = np.zeros((N,N),float)
Ey = np.zeros((N,N),float)
for i in range(len(rx.flat)):
    rx0 = np.array([rx.flat[i],0])
    rx5 = np.array([rx.flat[i],5])
    r = np.array([rx.flat[i],ry.flat[i]])
    Ex.flat[i],Ey.flat[i] = efieldq(-q,r,rx0) + efieldq(q,r,rx5) # Instead of making a new function for the y-axis we generalize the function and input a negatative charge instead
plt.quiver(rx,ry,Ex,Ey)
plt.hlines(5, -L, L, color = 'red', label = 'Positiv linjeladning')
plt.hlines(0,-L,L, color = 'blue', label = 'Negativ linjeladning')
plt.legend(loc = 'upper left')
plt.axis('equal')
plt.savefig('Linjeladninger_.b.pdf', format = 'pdf')