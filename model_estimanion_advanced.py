# This script computes an estimation of the parameters of the harmonic oscillator, based on the measurements
# We take in account an oscillator with variable parameters, this has turned out to be a good strategy to improve the quality of fit

import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
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

# Damped oscillator model with time-varying parameters
def damped_oscillator_variable(t, A, gamma0, gamma1, omega0, omega1, phi, offset):
    gamma = gamma0 + gamma1 * t
    omega = omega0 + omega1 * t  # omega in rad/s
    omega_t = omega * t + phi    # omega * t + phi in rad
    return A * np.exp(-gamma * t) * np.cos(omega_t) + offset

# Function to compute the initial guess
def compute_initial_guess(time, theta):
    # At this point, we make an initial guess for the parameters, which is essential for a good fit
    # Try it and you'll see
    # Find the peaks in the data
    peaks_indices, peaks_options = find_peaks(theta)
    peaks_time = time[peaks_indices]
    peaks_theta = theta[peaks_indices]

    # Estimate gamma0_guess using logarithmic decrement
    A1 = peaks_theta[0]
    A2 = peaks_theta[1]
    delta = np.log(np.abs(A1 / A2))
    T_start = peaks_time[1] - peaks_time[0] 
    gamma0_guess = delta / T_start

    # Estimate omega0
    omega0_guess = 2 * np.pi / T_start

    # Estimate the offset
    offset_guess = np.mean(theta)
    # Estimate the amplitude
    A_guess = (np.max(theta) - np.min(theta)) / 2
    # Estimate the initial phase
    if A_guess != 0:
        phi_guess = np.arccos((theta[0] - offset_guess) / A_guess)
    else:
        phi_guess = 0

    # Estimate gamma1_guess and omega1_guess
    A_start = peaks_theta[0]
    A_end = peaks_theta[-1]
    delta_total = np.log(np.abs(A_start / A_end))
    T_total = peaks_time[-1] - peaks_time[0]  # Total time T
    gamma_mean = delta_total / T_total  # gamma_mean is the integral average of gamma
    gamma1_guess = 2 * (gamma_mean - gamma0_guess) / T_total  # Guess for gamma1, knowing that gamma_mean is the integral average

    T_end = peaks_time[-1] - peaks_time[-2]  # Period of the last oscillation
    omega_end = 2 * np.pi / T_end
    omega1_guess = (omega_end - omega0_guess) / T_total

    initial_guess = [A_guess, gamma0_guess, gamma1_guess, omega0_guess, omega1_guess, phi_guess, offset_guess]
    return initial_guess

# Read the original dataset
filename = r'C:\Users\giuse\OneDrive\Desktop\personale\Triennale\Terzo anno\Secondo semestre\Tesi\myThesis\csv\measured_rotation.csv'  
time, theta_deg = read_measurements(filename)

# Convert theta from degrees to radians
theta = np.deg2rad(theta_deg)

# Initial guess for the parameters
initial_guess = compute_initial_guess(time, theta)

# Perform the parameter fit using curve_fit
popt, pcov = curve_fit(damped_oscillator_variable, time, theta, p0=initial_guess, maxfev=20000)

# Extract optimal parameters
A_opt, gamma0_opt, gamma1_opt, omega0_opt, omega1_opt, phi_opt, offset_opt = popt

# Calculate the fitted values
fit_values_rad = damped_oscillator_variable(time, *popt)

# Save the parameters in a CSV file
param_filename = r'C:\Users\giuse\OneDrive\Desktop\personale\Triennale\Terzo anno\Secondo semestre\Tesi\myThesis\csv\parameters.csv' 
with open(param_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['A', A_opt])
    writer.writerow(['gamma0', gamma0_opt])
    writer.writerow(['gamma1', gamma1_opt])
    writer.writerow(['omega0', omega0_opt])
    writer.writerow(['omega1', omega1_opt])
    writer.writerow(['phi', phi_opt])
    writer.writerow(['offset', offset_opt])

# Read the parameters from the CSV file
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

# Plot the original data and the fit
plt.figure(figsize=(12, 6))
plt.plot(time, theta_deg, label='Original Data')
plt.plot(time, fit_values_deg, label='Fit', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Theta (deg)')
plt.legend()
plt.title('Comparison between Original Data and Fit')
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












