#This script makes a naive estimation of the model
#it's naive, because it doesn't even makes an initial guess, and in facts the fit for real data is unacceptable

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv

# Function to read measurements from the CSV file
def read_measurements(filename):
    time = []
    theta = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            time.append(float(row[0]))
            theta.append(float(row[1]))
    return np.array(time), np.deg2rad(np.array(theta))

# Damped harmonic oscillator equation of motion
def damped_harmonic_oscillator(t, A, beta, omega, phi):
    return A * np.exp(-beta * t) * np.cos(omega * t + phi)

# Read data
filename = r'C:\Users\giuse\OneDrive\Desktop\personale\Triennale\Terzo anno\Secondo semestre\Tesi\myThesis\csv\measured_rotation.csv'
time, theta = read_measurements(filename)

# Perform curve fitting using the damped harmonic oscillator model
popt, pcov = curve_fit(damped_harmonic_oscillator, time, theta, maxfev=100000)

# Extract optimal parameters
A_opt, beta_opt, omega_opt, phi_opt = popt

print(f"Fitted parameters: A = {A_opt}, beta = {beta_opt}, omega = {omega_opt}, phi = {phi_opt}")

# Save the parameters to a CSV file
output_filename = r'C:\Users\giuse\OneDrive\Desktop\personale\Triennale\Terzo anno\Secondo semestre\Tesi\myThesis\parameters.csv'
with open(output_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['A', A_opt])
    writer.writerow(['beta', beta_opt])
    writer.writerow(['omega', omega_opt])
    writer.writerow(['phi', phi_opt])

# Calculate the fitted values using the optimized parameters
fit_values = damped_harmonic_oscillator(time, A_opt, beta_opt, omega_opt, phi_opt)

# Convert both the original data and fitted values to degrees for plotting
theta_deg = np.rad2deg(theta)
fit_values_deg = np.rad2deg(fit_values)

# Calculate the residual (difference between data and fit) in degrees
fit_error_deg = theta_deg - fit_values_deg

# Plot the original data and the fit in degrees
plt.figure(figsize=(10, 6))
plt.plot(time, theta_deg, 'b-', label='Original Data (deg)')
plt.plot(time, fit_values_deg, 'orange', label='Fitted Curve (deg)')
plt.xlabel('Time (s)')
plt.ylabel('Theta (deg)')
plt.legend()
plt.grid(True)
plt.title('Damped Harmonic Oscillator: Data vs Fitted Curve (in Degrees)')
plt.show()

# Plot the residual (difference between data and fit) in degrees
plt.figure(figsize=(10, 6))
plt.plot(time, fit_error_deg, 'darkgreen', label='Fitting Error (Data - Fit) (deg)')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Residual (Theta in deg)')
plt.title('Difference Between Data and Fitting (in Degrees)')
plt.legend()
plt.grid(True)
plt.show()
