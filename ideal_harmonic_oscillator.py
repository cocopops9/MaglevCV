# This program simulates the behavior of the rotation angle in a damped harmonic oscillator, based on its parameters.

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import csv

# Realistic physical parameters
I = 1   # moment of inertia (kg·m²)
k_rot = 15  # torsional spring constant (N·m/rad)
c_rot = 1  # torsional damping coefficient (N·m·s/rad)

# Matrix A, with alpha and beta 
alpha = -k_rot / I
beta = -c_rot / I
A = np.array([[0, 1],
              [alpha, beta]])

# Matrix C (only the angle θ)
C = np.array([[1, 0]])

# Initial conditions x(0) = [theta_0, omega_0]
theta_0 = np.pi / 4  # initial angle (rad)
omega_0 = np.pi  # initial angular velocity (rad/s)
x0 = np.array([[theta_0], [omega_0]])

time_values = np.linspace(0, 10, 1000)  # time from 0 to 10 seconds with 1000 points

# Initialization of the response vector size
n = len(time_values)
theta = np.zeros(n)

# calculation of the time response
for i in range(n):
    t = time_values[i] 
    
    # Compute exp(A * t)
    exp_At = expm(A * t)
    
    # Compute C @ exp(A*t) 
    temp_result = np.dot(C, exp_At)  
    theta[i] = np.dot(temp_result, x0).item() 

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time_values, theta, label=r'$C \cdot e^{At} \cdot x_0$', color='b')
plt.title('Response of a Damped Rotational Harmonic Oscillator')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.grid(True)
plt.legend()
plt.show()