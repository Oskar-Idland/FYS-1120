import numpy as np 
import matplotlib.pyplot as plt 
from scipy.constants import epsilon_0
from numba import njit

@njit
def epot4lines(r, L, rho_l, N, a):
    V = 0
    dl = L/N
    for i in range(int(-N/2), int(N/2 + dl)):
        r1 = np.array([a,i*dl,0])
        r2 = np.array([-a,i*dl,0])
        r3 = np.array([i*dl,a,0])
        r4 = np.array([i*dl,-a,0])
        R1 = np.linalg.norm(r - r1)
        R2 = np.linalg.norm(r - r2)
        R3 = np.linalg.norm(r - r3)
        R4 = np.linalg.norm(r - r4)
        dV1 = dl*rho_l / (4*np.pi*epsilon_0*(R1))
        dV2 = dl*rho_l / (4*np.pi*epsilon_0*(R2))
        dV3 = dl*rho_l / (4*np.pi*epsilon_0*(R3))
        dV4 = dl*rho_l / (4*np.pi*epsilon_0*(R4))
        V += dV1 + dV2 + dV3 + dV4
    return V

@njit
def efield4lines(r, L, rho_l, N, a):
    E = np.array([0.0,0.0,0.0])
    dl = L/N
    for i in range(int(-N/2), int(N/2 + dl)):
        r1 = np.array([a,i*dl,0])
        r2 = np.array([-a,i*dl,0])
        r3 = np.array([i*dl,a,0])
        r4 = np.array([i*dl,-a,0])
        R1 = r - r1
        R2 = r - r2
        R3 = r - r3
        R4 = r - r4
        R1_norm = np.linalg.norm(R1)
        R2_norm = np.linalg.norm(R2)
        R3_norm = np.linalg.norm(R3)
        R4_norm = np.linalg.norm(R4)
        
        k = dl*rho_l / (4*np.pi*epsilon_0)
        dE1 = R1/R1_norm**3
        dE2 = R2/R2_norm**3
        dE3 = R3/R3_norm**3
        dE4 = R4/R4_norm**3
        E += k*(dE1 + dE2 + dE3 + dE4)
    
    return E

N = 50
L = 2
y = np.linspace(-L, L, N)
z = np.linspace(-L, L, N)
ry, rz = np.meshgrid(y,z)
rho_l = 1
a = 1
V = np.zeros((N,N))
Ey, Ez = np.zeros((N,N)), np.zeros((N,N))
for i in range(len(V.flat)):
    r = np.array([0, ry.flat[i], rz.flat[i]])
    field = efield4lines(r, L, rho_l, N, a)
    Ey.flat[i], Ez.flat[i] = field[1], field[2]
    V.flat[i] = epot4lines(r, L, rho_l, N, a)
    


plt.contourf(ry, rz, V, 10, cmap = 'plasma')
plt.streamplot(ry,rz,Ey,Ez, color = 'black', density = .8, broken_streamlines= False)
plt.xlabel('y-axis')
plt.ylabel('z-axis')
plt.show()


# d)

def AnalPot(z,a,Q):
    return Q/(4*np.pi*epsilon_0)* np.arcsinh(a/np.sqrt(a**2 + z**2))

Q = 4*rho_l*L
Vz = V[:,int(N/2)]
z = np.linspace(-L,L,N)
VzAnal = AnalPot(z,a,Q)
plt.plot(z,Vz, label = "Numerical Solution", linewidth = 3)
plt.plot(z,VzAnal, label = "Analytical Solution", linestyle = "--", color = 'red', linewidth = 3)
plt.xlabel('z-axis')
plt.ylabel(r'Potential $V(z)$')
plt.legend()
plt.savefig('Num_vs_Anal.pdf')
plt.show()