import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
def generate_qam_symbols(u):
    """Generate (4u^2)-QAM constellation points"""
    real_parts = np.array([2 * k - 1 for k in range(-u + 1, u + 1, 2)])
    constellation = []
    for r in real_parts:
        for i in real_parts:
            constellation.append(r + 1j * i)
    return np.array(constellation)
def project_to_constellation(z, constellation):
    """Project a complex number to the nearest constellation point"""
    distances = np.abs(z - constellation)
    return constellation[np.argmin(distances)]
def generate_mimo_system(m, n, constellation, snr_db):
    """Generate MIMO system: H, x*, y according to model (1)"""
    # Generate channel matrix with i.i.d. complex Gaussian entries
    H = (np.random.randn(m, n) + 1j * np.random.randn(m, n)) / np.sqrt(2)
    # Generate random symbols from constellation
    x_true = np.random.choice(constellation, size=n)
    # Calculate noise variance based on SNR
    sigma_x_sq = np.mean(np.abs(constellation) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    sigma_v_sq = m * sigma_x_sq / snr_linear
    # Generate noise vector
    v = np.sqrt(sigma_v_sq / 2) * (np.random.randn(m) + 1j * np.random.randn(m))
    # Received signal
    y = H @ x_true + v
    return H, x_true, y, sigma_v_sq
def mmse_detector(H, y, sigma_v_sq, constellation):
    """MMSE detector implementation"""
    m, n = H.shape
    sigma_x_sq = np.mean(np.abs(constellation) ** 2)
    delta = sigma_v_sq / sigma_x_sq
    # MMSE solution
    x_mmse = np.linalg.inv(H.conj().T @ H + delta * np.eye(n)) @ H.conj().T @ y
    # Project to constellation
    x_est = np.array([project_to_constellation(z, constellation) for z in x_mmse])
    return x_est
def gpm_detector(H, y, constellation, max_iter=1, alpha=None, x_init=None):
    """Generalized Power Method detector implementation"""
    m, n = H.shape
    # Initialize with MMSE if no initial point provided
    if x_init is None:
        sigma_x_sq = np.mean(np.abs(constellation) ** 2)
        sigma_v_sq_estimate = np.var(y - H @ mmse_detector(H, y, 1, constellation))
        delta = sigma_v_sq_estimate / sigma_x_sq
        x_init = mmse_detector(H, y, sigma_v_sq_estimate, constellation)
    # Default step size (as suggested in Theorem 2)
    if alpha is None:
        alpha = 0.5/(1+max_iter)
    x_prev = x_init
    x_history = [x_prev.copy()]
    for k in range(max_iter):
        # Gradient calculation
        grad_F = 2 * H.conj().T @ (H @ x_prev - y)
        # Gradient step and projection
        z = x_prev - (alpha / m) * grad_F
        x_next = np.array([project_to_constellation(z_i, constellation) for z_i in z])
        # Check stopping criterion (no change in solution)
        if np.array_equal(x_next, x_prev):
            break
        x_prev = x_next
        x_history.append(x_prev.copy())
    return x_next, len(x_history)
def calculate_ser(x_true, x_est):
    """Calculate Symbol Error Rate"""
    return np.mean(x_true != x_est)
def verify_theorem_conditions(H, v, constellation, alpha):
    """Verify conditions from Theorem 1"""
    m, n = H.shape
    # Calculate c for QAM constellation
    min_dist = np.min([np.abs(a - b) for i, a in enumerate(constellation)
                       for j, b in enumerate(constellation) if i != j])
    c = 4 / min_dist
    # First condition: ||(2α/m)H*v||_∞ < 1/c
    term1 = np.abs((2 * alpha / m) * H.conj().T @ v)
    cond1 = np.max(term1) < (1 / c)
    # Second condition: ||I - (2α/m)H*H||_op ≤ β < 1/4
    op_term = np.eye(n) - (2 * alpha / m) * H.conj().T @ H
    op_norm = np.linalg.norm(op_term, 2)  # operator norm (largest singular value)
    cond2 = op_norm < 0.25
    return cond1, cond2, c, op_norm
def simulate_ser_vs_snr(m, n, constellation, snr_db_list, num_trials=1000):
    """Simulate SER vs SNR for GPM and MMSE detectors"""
    ser_gpm = []
    ser_mmse = []
    cond1_satisfied = 0
    cond2_satisfied = 0
    for snr_db in tqdm(snr_db_list):
        ser_gpm_trials = []
        ser_mmse_trials = []
        for _ in range(num_trials):
            H, x_true, y, sigma_v_sq = generate_mimo_system(m, n, constellation, snr_db)
            v = y - H @ x_true
            # Verify Theorem conditions
            alpha = 0.5  # as suggested in Theorem 2
            cond1, cond2, c, op_norm = verify_theorem_conditions(H, v, constellation, alpha)
            if cond1:
                cond1_satisfied += 1
            if cond2:
                cond2_satisfied += 1
            # GPM detection
            x_gpm, _ = gpm_detector(H, y, constellation, alpha=alpha)
            ser_gpm_trials.append(calculate_ser(x_true, x_gpm))
            # MMSE detection
            x_mmse = mmse_detector(H, y, sigma_v_sq, constellation)
            ser_mmse_trials.append(calculate_ser(x_true, x_mmse))
        ser_gpm.append(np.mean(ser_gpm_trials))
        ser_mmse.append(np.mean(ser_mmse_trials))
    # Calculate condition satisfaction rates
    cond1_rate = cond1_satisfied / (len(snr_db_list) * num_trials)
    cond2_rate = cond2_satisfied / (len(snr_db_list) * num_trials)
    return ser_gpm, ser_mmse, cond1_rate, cond2_rate
def plot_ser_results(snr_db_list, ser_gpm, ser_mmse, title):
    """Plot SER vs SNR results"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_db_list, ser_gpm, 'b-o', label='GPM')
    plt.semilogy(snr_db_list, ser_mmse, 'r--s', label='MMSE')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.title(title)
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.show()
def main():
    # Parameters
    u = 2  # For 16-QAM (4*2^2 = 16 points)
    m, n = 128, 64  # System dimensions
    snr_db_list = np.arange(0, 21, 2)  # SNR range in dB
    num_trials = 1000  # Number of trials per SNR point
    # Generate QAM constellation
    constellation = generate_qam_symbols(u)
    # Run simulation
    print("Running simulations for 16-QAM MIMO system...")
    ser_gpm, ser_mmse, cond1_rate, cond2_rate = simulate_ser_vs_snr(
        m, n, constellation, snr_db_list, num_trials)
    # Print condition satisfaction rates
    print(f"Condition 1 satisfaction rate: {cond1_rate:.2%}")
    print(f"Condition 2 satisfaction rate: {cond2_rate:.2%}")
    # Plot results
    plot_ser_results(snr_db_list, ser_gpm, ser_mmse,
                     f'SER vs SNR for 16-QAM MIMO Detection (m={m}, n={n})')
if __name__ == "__main__":
    main()