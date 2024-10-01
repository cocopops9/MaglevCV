# This program performs Kalman filtering of the angular position of a damped harmonic oscillator
# It receives data and parameters from 2 CSV files and generates a plot of the measurement and the Kalman filter estimate

import numpy as np
import matplotlib.pyplot as plt
import csv

# This function reads data from a csv file
def read_measurements(filename):
    time = []
    theta = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            time.append(float(row[0]))  # Time column
            theta.append(float(row[1]))  # Angle column
    return np.array(time), np.array(theta)

# Function to read parameters from a CSV file
def read_parameters(filename):
    parameters = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            parameters[row[0]] = float(row[1])
    return parameters

param_file = r'C:\Users\giuse\OneDrive\Desktop\personale\Triennale\Terzo anno\Secondo semestre\Tesi\myThesis\parametri.csv'
params = read_parameters(param_file)

 # save the estimated parameters
alpha = params['alpha']
beta = params['beta']
omega_0 = params['omega_0']
theta_0 = params['theta_0']

# The A matrix has been edited because the kalman filtering is a discrete time algorithm
A = np.array([[1, 0.0001],  # Sampling time 0.0001
              [alpha * 0.0001, 1 + beta * 0.0001]])

# Matrix C (we only observe the angle θ)
C = np.array([[1, 0]])

# Kalman filter initialization
B = np.array([[1e-5, 0],  # Process noise
              [0, 1e-5]])
D = np.array([[1e-2]])  # Measurement noise
P = np.array([[1, 0],  # Error covariance initialization
              [0, 1]]) 

theta_est = np.array([theta_0, omega_0])  # Initial estimated state

# Read the real model data
filename = r'C:\Users\giuse\OneDrive\Desktop\personale\Triennale\Terzo anno\Secondo semestre\Tesi\myThesis\risposta_rotazionale.csv'
time, theta_measured = read_measurements(filename)

n = len(time)

theta_estimated = np.zeros(n)

# Kalman filter
for i in range(n):
    # Predict the next state (system model)
    theta_pred = A @ theta_est  # State prediction
    P_pred = A @ P @ A.T + B  # Error covariance prediction

    # Update with the measurement
    z = theta_measured[i]  # Current measurement
    y = z - C @ theta_pred  # Innovation (residual)
    
    # Calculate the Kalman gain
    S = C @ P_pred @ C.T + D  # Innovation covariance
    K = P_pred @ C.T @ np.linalg.inv(S)  # Kalman gain
    
    # Update the state and error covariance
    theta_est = theta_pred + np.dot(K, y)  # Apply the Kalman gain to the residual
    P = (np.array([[1, 0],  # Error covariance initialization
              [0, 1]])  - K @ C) @ P_pred  # Update the error covariance

    # Save the estimated angle
    theta_estimated[i] = theta_est[0]

# Plot the results: noisy data vs Kalman filter estimate
plt.figure(figsize=(10, 6))
plt.plot(time, theta_measured, label='Measurement', color='r', linewidth=0.95)
plt.plot(time, theta_estimated, label='Kalman Filter Estimation', color='purple', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Kalman Filter - Rotation Angle Estimation')
plt.legend()
plt.grid(True)
plt.show()