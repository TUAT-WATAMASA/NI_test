# -*- coding: utf-8 -*-
"""
Spyderエディタ

MCK system (Runge-Kutta)
"""

## Initialization

# import library
import numpy as np
import matplotlib.pyplot as plt
#simulation paramter

dt = 0.1;
N = 50000;
T = N*dt

# state variables
x1 = np.zeros(N)
x2 = np.zeros(N)
t = np.zeros(N)

E = np.zeros(N)
# state vectors
X_EE = np.array( [ x1, x2 ])
X_IE = np.array( [ x1, x2 ])
X_ERK = np.array( [ x1, x2 ])
X_AS = np.array( [ x1, x2 ])

# initial conditions
x10 = 0
x20 = 1
t0 = 0;
E0 = np.power(x10,2)/2+np.power(x20,2)/2

X_EE[0,0] = x10
X_EE[1,0] = x20

X_IE[0,0] = x10
X_IE[1,0] = x20

X_ERK[0,0] = x10
X_ERK[1,0] = x20

X_AS[0,0] = x10
X_AS[1,0] = x20

t[0] = t0;
E[0] = E0

# system matrix
A = np.array([ [0, 1],[-1, 0] ])
B = np.linalg.inv(np.eye(2)-A*dt)

# simulation

for i in range(N-1):
    # Explicit Euler method
    X_EE[:,i+1] = X_EE[:,i] + np.dot( A,X_EE[:,i] )*dt
    
    # Implicit Euler method 
    X_IE[:,i+1] = np.dot( B, X_IE[:,i] )
    
    # 4th Runge-Kutta method
    K = np.zeros( (2,4) )
    K[:,0] = np.dot( A, X_ERK[:,i] )
    K[:,1] = np.dot( A, ( X_ERK[:,i] + K[:,0]*dt/2 ) )
    K[:,2] = np.dot( A, ( X_ERK[:,i] + K[:,1]*dt/2 ))
    K[:,3] = np.dot( A, ( X_ERK[:,i]+K[:,2]*dt ) )
    X_ERK[:,i+1] = X_ERK[:,i] + (K[:,0] + 2*K[:,1] + 2*K[:,2] + K[:,3] )*dt/6
    
    # Analytical solution
    t[i+1] = t[i] + dt
    X_AS[:,i+1] = np.array( [np.sin(t[i+1]), np.cos(t[i+1])])
    
    # Energy
    E[i+1] = np.power(X_ERK[0,i+1],2)/2 + np.power(X_ERK[1,i+1],2)/2
plt.figure
plt.plot(t,X_AS[0,:],"k",label="Analytical solution")
plt.plot(t,X_EE[0,:],"b",label="Explicit Euler method")
plt.plot(t,X_IE[0,:],"r",label="Implicit Euler method")
plt.legend()
plt.savefig("Ex_Im_comp.png",format = 'png', dpi=300)
plt.close()

plt.figure
plt.plot(t,X_AS[0,:],"k",label="Analytical solution")
plt.plot(t,X_EE[0,:],"b",label="Euler method")
plt.plot(t,X_ERK[0,:],"r",label="Runge-Kutta method")
plt.legend()
plt.savefig("Euler_RK_comp.png",format = 'png', dpi=300)
plt.close()

plt.figure
plt.plot(t,E)