import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Funzioni per leggere i dati e calcolare offset
def read_measurements(filename):
    time = []
    theta = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            time.append(float(row[0]))
            theta.append(float(row[1]))
    return np.array(time), np.array(theta)

def compute_offset(theta):
    return np.mean(theta)

# Funzioni per identificare picchi e calcolare beta(t) e omega(t)
def find_peaks_and_troughs(theta):
    peaks_indices, _ = find_peaks(theta)
    troughs_indices, _ = find_peaks(-theta)
    return peaks_indices, troughs_indices

def calculate_beta(time, theta, extremum_indices):
    extremum_times = time[extremum_indices]
    extremum_amplitudes = np.abs(theta[extremum_indices])
    beta_times = []
    beta_values = []
    for i in range(len(extremum_amplitudes) - 1):
        A_i = extremum_amplitudes[i]
        A_ip1 = extremum_amplitudes[i + 1]
        t_i = extremum_times[i]
        t_ip1 = extremum_times[i + 1]
        delta_t = t_ip1 - t_i
        if A_ip1 == 0 or delta_t == 0:
            continue
        beta_i = (1 / delta_t) * np.log(A_i / A_ip1)
        beta_times.append(t_i + delta_t / 2)
        beta_values.append(beta_i)
    return np.array(beta_times), np.array(beta_values)

def calculate_omega(time, extremum_indices):
    extremum_times = time[extremum_indices]
    omega_times = []
    omega_values = []
    for i in range(len(extremum_times) - 1):
        T_i = extremum_times[i + 1] - extremum_times[i]
        omega_i = np.pi / T_i
        omega_times.append(extremum_times[i] + T_i / 2)
        omega_values.append(omega_i)
    return np.array(omega_times), np.array(omega_values)

# Modelli per beta(t) e omega(t)
def omega_model(t, A1, f1, phi1, omega0):
    return A1 * np.sin(f1 * t + phi1) + omega0

def beta_model(t, A2, f2, phi2, B):
    return A2 * np.sin(f2 * t + phi2) + B * t

# Funzione principale
def main():
    # Lettura dei dati
    filename = r'C:\Users\giuse\OneDrive\Desktop\personale\Triennale\Terzo anno\Secondo semestre\Tesi\myThesis\csv\measured_rotation.csv'# Sostituisci con il percorso corretto
    time, theta_deg = read_measurements(filename)
    theta = np.deg2rad(theta_deg)
    offset = compute_offset(theta)
    theta_no_offset = theta - offset

    # Identificazione dei picchi
    peaks_indices, troughs_indices = find_peaks_and_troughs(theta_no_offset)
    extremum_indices = np.sort(np.concatenate((peaks_indices, troughs_indices)))

    # Calcolo di beta(t) e omega(t)
    beta_times, beta_values = calculate_beta(time, theta_no_offset, extremum_indices)
    omega_times, omega_values = calculate_omega(time, extremum_indices)

    # Fitting di omega(t)
    A1_guess = (np.max(omega_values) - np.min(omega_values)) / 2
    omega0_guess = np.mean(omega_values)
    f1_guess = 0.1
    phi1_guess = 0
    initial_guess_omega = [A1_guess, f1_guess, phi1_guess, omega0_guess]
    omega_params_opt, _ = curve_fit(omega_model, omega_times, omega_values, p0=initial_guess_omega)

    # Fitting di beta(t)
    A2_guess = (np.max(beta_values) - np.min(beta_values)) / 2
    B_guess = (beta_values[-1] - beta_values[0]) / (beta_times[-1] - beta_times[0])
    f2_guess = 0.05
    phi2_guess = 0
    initial_guess_beta = [A2_guess, f2_guess, phi2_guess, B_guess]
    beta_params_opt, _ = curve_fit(beta_model, beta_times, beta_values, p0=initial_guess_beta)

    # Modello dell'oscillatore armonico aggiornato
    def theta_model(t, A, phi, offset, beta_params_opt, omega_params_opt):
        beta_t = beta_model(t, *beta_params_opt)
        exponent = -beta_t * t
        A1, f1, phi1, omega0 = omega_params_opt
        if f1 != 0:
            Psi_t = (-A1 / f1) * np.cos(f1 * t + phi1) + (A1 / f1) * np.cos(phi1) + omega0 * t + phi
        else:
            Psi_t = A1 * t * np.sin(phi1) + omega0 * t + phi
        theta_t = A * np.exp(exponent) * np.cos(Psi_t) + offset
        return theta_t

    # Fitting del modello dell'oscillatore armonico
    A_guess = (np.max(theta_no_offset) - np.min(theta_no_offset)) / 2
    phi_guess = 0
    offset_guess = 0
    initial_guess_theta = [A_guess, phi_guess, offset_guess]

    popt_theta, _ = curve_fit(
        lambda t, A, phi, offset: theta_model(t, A, phi, offset, beta_params_opt, omega_params_opt),
        time, theta_no_offset, p0=initial_guess_theta, maxfev=100000)

    A_opt, phi_opt, offset_opt = popt_theta

    # Calcolo dei valori fittati
    theta_fit = theta_model(time, *popt_theta, beta_params_opt, omega_params_opt)

    # Plot dei dati originali e del fit
    plt.figure(figsize=(12, 6))
    plt.plot(time, theta_no_offset, label='Dati Originali')
    plt.plot(time, theta_fit, label='Fit Modello Aggiornato', linestyle='--')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Theta (rad)')
    plt.legend()
    plt.title('Confronto tra Dati Originali e Fit con Modello Aggiornato')
    plt.grid(True)
    plt.show()

    # Calcolo dei residui
    residuals = theta_no_offset - theta_fit
    SSR = np.sum(residuals ** 2)
    print(f'Somma dei Residui al Quadrato (SSR): {SSR:.2f}')

    # Plot dei residui
    plt.figure(figsize=(12, 6))
    plt.plot(time, residuals, label='Residui', color='darkgreen')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Residui (rad)')
    plt.title(f'Residui tra Dati Originali e Fit (SSR: {SSR:.2f})')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Stampa dei parametri
    print("Parametri omega(t):", omega_params_opt)
    print("Parametri beta(t):", beta_params_opt)
    print("Parametri del modello dell'oscillatore:", popt_theta)

if __name__ == "__main__":
    main()




