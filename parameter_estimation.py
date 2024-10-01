# This program is used to derive the parameters of a damped harmonic oscillator, using the curve_fit function from scipy.optimize
# The data is read from a CSV file
# The output prints and saves the parameters: alpha (-k_rot / I), beta (-c_rot / I), theta0 (initial angle), and omega0 (initial angular velocity)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv

# Function to read data from the CSV file
def read_measurements(filename):
    time = []
    theta = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            time.append(float(row[0]))  # Time column
            theta.append(float(row[1]))  # Angle column
    return np.array(time), np.array(theta) 

# Function for the model of the underdamped harmonic oscillator model
def damped_harmonic_oscillator(t, A, gamma, omega_d, phi):
    return A * np.exp(-gamma * t) * np.cos(omega_d * t + phi)

# Read data from the CSV file
filename = r'C:\Users\giuse\OneDrive\Desktop\personale\Triennale\Terzo anno\Secondo semestre\Tesi\myThesis\risposta_rotazionale.csv'
time, theta = read_measurements(filename)

# Perform parameter fitting
popt, pcov = curve_fit(damped_harmonic_oscillator, time, theta)

# Obtain the optimal parameters
A_opt, gamma_opt, omega_d_opt, phi_opt = popt
# Calculate alpha, beta, theta_0, and omega_0 from the fit parameters
beta = -2 * gamma_opt
alpha = -(omega_d_opt**2 + gamma_opt**2)
theta_0 = A_opt * np.cos(phi_opt)
omega_0 = -A_opt * gamma_opt * np.cos(phi_opt) - A_opt * omega_d_opt * np.sin(phi_opt)

# Print calculated parameters
print(f"Calculated parameters: omega_0 = {omega_0}, alpha = {alpha}, beta = {beta}, theta_0 = {theta_0}")

# Save the calculated parameters in a CSV file
output_filename = r'C:\Users\giuse\OneDrive\Desktop\personale\Triennale\Terzo anno\Secondo semestre\Tesi\myThesis\parametri.csv'
with open(output_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['alpha', alpha])
    writer.writerow(['beta', beta])
    writer.writerow(['omega_0', omega_0])
    writer.writerow(['theta_0', theta_0])

# Calculate the difference between the CSV data and the fit
fit_error = theta - damped_harmonic_oscillator(time, A_opt, gamma_opt, omega_d_opt, phi_opt)

# Plot the original data and the fit
plt.figure(figsize=(10, 6))
plt.plot(time, damped_harmonic_oscillator(time, A_opt, gamma_opt, omega_d_opt, phi_opt), 'r-', label='Fit')
plt.xlabel('Time (s)')
plt.ylabel('Theta (rad)')
plt.legend()
plt.grid(True)
plt.show()

# Plot the residual (difference between data and fit)
plt.figure(figsize=(10, 6))
plt.plot(time, fit_error, 'g-', label='Fitting error (Data - Fit)')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Theta (rad)')
plt.title('Difference between data and fitting')
plt.legend()
plt.grid(True)
plt.show()