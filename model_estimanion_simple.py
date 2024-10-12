# This script computes an estimation of the parameters of the harmonic oscillator based on the measurements
# It's simple, because we consider an oscillator with constant parameters to model the system

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Function to read data from the CSV file
def read_measurements(filename):
    time = []
    theta = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            time.append(float(row[0]))
            theta.append(float(row[1]))
    return np.array(time), np.array(theta)

# Damped harmonic oscillator model with constant parameters
def damped_oscillator(t, A, gamma, omega, phi, offset):
    # gamma and omega are constants
    return A * np.exp(-gamma * t) * np.cos(omega * t + phi) + offset

# Function to compute the initial guess
def compute_initial_guess(time, theta):
    # Find the peaks in the data to estimate initial parameters
    peaks_indices, _ = find_peaks(theta)
    peaks_time = time[peaks_indices]
    peaks_theta = theta[peaks_indices]

    # Estimate gamma (damping coefficient) using logarithmic decrement
    A1 = peaks_theta[0]
    A2 = peaks_theta[1]
    delta = np.log(np.abs(A1 / A2))
    T = peaks_time[1] - peaks_time[0]  # Period of oscillation
    gamma_guess = delta / T

    # Estimate omega (angular frequency)
    omega_guess = 2 * np.pi / T

    # Estimate the offset (assumed to be zero if not specified)
    offset_guess = np.mean(theta)

    # Estimate the amplitude
    A_guess = (np.max(theta) - np.min(theta)) / 2

    # Estimate the initial phase
    if A_guess != 0:
        phi_guess = np.arccos((theta[0] - offset_guess) / A_guess)
    else:
        phi_guess = 0

    # Initial guess array
    initial_guess = [A_guess, gamma_guess, omega_guess, phi_guess, offset_guess]
    return initial_guess

# Read the original dataset
filename = r'C:\Users\giuse\OneDrive\Desktop\personale\Triennale\Terzo anno\Secondo semestre\Tesi\myThesis\csv\measured_rotation.csv'  
time, theta_deg = read_measurements(filename)

# Convert theta from degrees to radians
theta = np.deg2rad(theta_deg)

# Initial guess for the parameters
initial_guess = compute_initial_guess(time, theta)

# Perform the parameter fit using curve_fit
popt, pcov = curve_fit(
    damped_oscillator,
    time,
    theta,
    p0=initial_guess,
    maxfev=20000
)

# Extract optimal parameters
A_opt, gamma_opt, omega_opt, phi_opt, offset_opt = popt

# Calculate the fitted values
fit_values_rad = damped_oscillator(time, *popt)

# Save the parameters in a CSV file
param_filename = r'C:\Users\giuse\OneDrive\Desktop\personale\Triennale\Terzo anno\Secondo semestre\Tesi\myThesis\csv\parameters_constant.csv' 
with open(param_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['A', A_opt])
    writer.writerow(['gamma', gamma_opt])
    writer.writerow(['omega', omega_opt])
    writer.writerow(['phi', phi_opt])
    writer.writerow(['offset', offset_opt])

# Read the parameters from the CSV file (optional, for verification)
params = {}
with open(param_filename, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        params[row[0]] = float(row[1])

fit_values_deg = np.rad2deg(fit_values_rad)

# Calculate the residuals (difference between original data and fit)
residuals = theta_deg - fit_values_deg

# Calculate the sum of squared residuals (SSR)
SSR = np.sum(residuals ** 2)

# Print the sum of squared residuals
print(f'Sum of squared residuals (SSR): {SSR:.5f}')

# Print the optimal parameters
print('Optimal parameters:')
print(f'A = {A_opt}')
print(f'gamma = {gamma_opt}')
print(f'omega = {omega_opt}')
print(f'phi = {phi_opt}')
print(f'offset = {offset_opt}')

# Plot the original data and the fit
plt.figure(figsize=(12, 6))
plt.plot(time, theta_deg, label='Original Data')
plt.plot(time, fit_values_deg, label='Fit', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Theta (deg)')
plt.legend()
plt.title('Comparison between Original Data and Fit (Constant Parameters)')
plt.grid(True)
plt.show()

# Plot of the residuals with SSR in the title
plt.figure(figsize=(12, 6))
plt.plot(time, residuals, label='Residuals', color='darkgreen') 
plt.xlabel('Time (s)')
plt.ylabel('Residuals (deg)')
plt.title(f'Residuals between Original Data and Fit (SSR: {SSR:.2f})')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.grid(True)
plt.legend()
plt.show()








