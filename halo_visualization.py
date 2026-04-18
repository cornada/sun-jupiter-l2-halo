import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# VISUAL PACKAGE FOR SUN-JUPITER L2 LYAPUNOV ORBIT
# Closed orbit + min-energy correction strategy
# Generates many PNG figures into ./figs/
# ============================================================

SAVE_DIR = '/Users/aleksandrvolkov/2/figs'
os.makedirs(SAVE_DIR, exist_ok=True)

def savefig(name, fig=None):
    path = os.path.join(SAVE_DIR, name)
    (fig or plt.gcf()).savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig or plt.gcf())
    print(f"  saved {path}")

# ------------------------------------------------------------
# DYNAMICS (same as halo_correction_strategy.py)
# ------------------------------------------------------------
mu = 9.53e-4

def _radii(U):
    x, y, z = U[0], U[1], U[2]
    R1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    R2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)
    return R1, R2

def cr3bp(t, U):
    x, y, z, vx, vy, vz = U
    R1, R2 = _radii(U)
    ax = 2*vy + x - (1-mu)*(x+mu)/R1**3 - mu*(x-(1-mu))/R2**3
    ay = -2*vx + y - (1-mu)*y/R1**3 - mu*y/R2**3
    az = -(1-mu)*z/R1**3 - mu*z/R2**3
    return [vx, vy, vz, ax, ay, az]

def jacobi(U):
    x, y, z, vx, vy, vz = U
    R1, R2 = _radii(U)
    Omega = 0.5*(x*x + y*y) + (1-mu)/R1 + mu/R2 + 0.5*mu*(1-mu)
    return 2*Omega - (vx*vx + vy*vy + vz*vz)

def A_matrix(U):
    x, y, z = U[0], U[1], U[2]
    R1sq = (x+mu)**2 + y**2 + z**2
    R2sq = (x-(1-mu))**2 + y**2 + z**2
    R1_3, R1_5 = R1sq**1.5, R1sq**2.5
    R2_3, R2_5 = R2sq**1.5, R2sq**2.5
    mu1 = 1 - mu
    Oxx = 1 - mu1*(1/R1_3 - 3*(x+mu)**2/R1_5) - mu*(1/R2_3 - 3*(x-(1-mu))**2/R2_5)
    Oyy = 1 - mu1*(1/R1_3 - 3*y**2/R1_5)       - mu*(1/R2_3 - 3*y**2/R2_5)
    Ozz =     -mu1*(1/R1_3 - 3*z**2/R1_5)       - mu*(1/R2_3 - 3*z**2/R2_5)
    Oxy = 3*y*(mu1*(x+mu)/R1_5 + mu*(x-(1-mu))/R2_5)
    Oxz = 3*z*(mu1*(x+mu)/R1_5 + mu*(x-(1-mu))/R2_5)
    Oyz = 3*y*z*(mu1/R1_5 + mu/R2_5)
    return np.array([
        [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],
        [Oxx,Oxy,Oxz,0,2,0],[Oxy,Oyy,Oyz,-2,0,0],[Oxz,Oyz,Ozz,0,0,0],
    ], dtype=float)

def var_rhs(t, Y):
    U = Y[:6]
    Phi = Y[6:].reshape(6, 6)
    return np.concatenate([cr3bp(t, U), (A_matrix(U) @ Phi).flatten()])

# ------------------------------------------------------------
# ORBIT + MONODROMY
# ------------------------------------------------------------
def find_L2():
    def f(x):
        R1 = abs(x + mu); R2 = abs(x - (1 - mu))
        return x - (1-mu)*(x+mu)/R1**3 - mu*(x-(1-mu))/R2**3
    return root_scalar(f, bracket=[1-mu+1e-4, 1.5], method='brentq').root

def find_L1():
    def f(x):
        R1 = abs(x + mu); R2 = abs(x - (1 - mu))
        return x - (1-mu)*(x+mu)/R1**3 - mu*(x-(1-mu))/R2**3
    return root_scalar(f, bracket=[1-mu-0.5, 1-mu-1e-4], method='brentq').root

def find_L3():
    def f(x):
        R1 = abs(x + mu); R2 = abs(x - (1 - mu))
        return x - (1-mu)*(x+mu)/R1**3 - mu*(x-(1-mu))/R2**3
    return root_scalar(f, bracket=[-1.5, -mu-1e-4], method='brentq').root

x_L1, x_L2, x_L3 = find_L1(), find_L2(), find_L3()

# Differential correction
def half_period(x0, vy0, t_max=6.0):
    Y0 = np.zeros(42); Y0[:6] = [x0, 0, 0, 0, vy0, 0]; Y0[6:] = np.eye(6).flatten()
    direction = int(np.sign(vy0))
    def hit_y(t, Y): return Y[1]
    hit_y.terminal = True; hit_y.direction = -direction
    sol = solve_ivp(var_rhs, (1e-6, t_max), Y0, events=hit_y,
                    rtol=1e-12, atol=1e-14)
    return sol.t_events[0][0], sol.y_events[0][0]

def diff_correct(x0, vy0, tol=1e-12, maxiter=40):
    for _ in range(maxiter):
        T_half, Yf = half_period(x0, vy0)
        U = Yf[:6]; Phi = Yf[6:].reshape(6, 6)
        vx_f = U[3]
        if abs(vx_f) < tol: return vy0, T_half
        dU = cr3bp(T_half, U)
        coef = Phi[3, 4] - dU[3] * Phi[1, 4] / dU[1]
        vy0 -= vx_f / coef
    raise RuntimeError("no convergence")

x0 = x_L2 + 2.0e-3
A_L2 = A_matrix([x_L2, 0, 0, 0, 0, 0])
ev_in = np.linalg.eigvals(A_L2[[0, 1, 3, 4]][:, [0, 1, 3, 4]])
omega_p = float(np.max(np.abs(ev_in.imag)))
vy0_guess = -0.5 * (omega_p**2 + A_L2[3, 0]) * (x0 - x_L2)
vy0, T_half = diff_correct(x0, vy0_guess)
T = 2.0 * T_half

# Full-period integration
Y0 = np.zeros(42); Y0[:6] = [x0, 0, 0, 0, vy0, 0]; Y0[6:] = np.eye(6).flatten()
t_grid = np.linspace(0, T, 3000)
sol = solve_ivp(var_rhs, (0, T), Y0, t_eval=t_grid,
                rtol=1e-13, atol=1e-15)
U_t = sol.y[:6, :]
Phi_t = sol.y[6:, :].reshape(6, 6, -1)
C_t = np.array([jacobi(U_t[:, i]) for i in range(U_t.shape[1])])
M = Phi_t[:, :, -1]

# Eigenanalysis
eigM, VR = np.linalg.eig(M)
idx_u = int(np.argmax(np.abs(eigM)))
lam_u = eigM[idx_u].real
e_u = np.real_if_close(VR[:, idx_u]).real
e_u /= np.linalg.norm(e_u)

idx_s = int(np.argmin(np.abs(eigM)))
lam_s = eigM[idx_s].real
e_s = np.real_if_close(VR[:, idx_s]).real
e_s /= np.linalg.norm(e_s)

eigMT, VL = np.linalg.eig(M.T)
idx_uL = int(np.argmin(np.abs(eigMT - lam_u)))
w_u = np.real_if_close(VL[:, idx_uL]).real
w_u /= (w_u @ e_u)

# Propagate left eigenvector
L_t = np.empty((6, t_grid.size))
for i in range(t_grid.size):
    L_t[:, i] = np.linalg.solve(Phi_t[:, :, i].T, w_u)
norm_Lv = np.linalg.norm(L_t[3:, :], axis=0)
best = int(np.argmax(norm_Lv))
worst = int(np.argmin(norm_Lv))

# Propagate unstable direction e_u(t) = Phi(t,0) e_u
eu_t = np.einsum('ijk,j->ik', Phi_t, e_u)

print(f"x_L2={x_L2:.6f}  T={T:.4f}  lam_u={lam_u:.2f}")
print(f"Best t*/T={t_grid[best]/T:.3f}  leverage ratio={norm_Lv[best]/norm_Lv[worst]:.1f}x\n")

print("Generating figures...")

# ============================================================
# FIG 1: Hill region + orbit + primaries + Lagrange points
# ============================================================
C_ref = C_t[0]
xr = np.linspace(0.6, 1.2, 500)
yr = np.linspace(-0.35, 0.35, 500)
XX, YY = np.meshgrid(xr, yr)
R1g = np.sqrt((XX+mu)**2 + YY**2)
R2g = np.sqrt((XX-(1-mu))**2 + YY**2)
OmegaG = 0.5*(XX**2+YY**2) + (1-mu)/R1g + mu/R2g + 0.5*mu*(1-mu)
CG = 2*OmegaG

fig = plt.figure(figsize=(11, 8))
plt.contourf(XX, YY, np.where(CG < C_ref, 1, 0), levels=[0.5, 1.5],
             colors=['#ffdddd'], alpha=0.6)
plt.contour(XX, YY, CG, levels=[C_ref], colors='crimson', linewidths=1.2,
            linestyles='--')
plt.plot(U_t[0], U_t[1], 'b-', lw=2.5, label=f'Lyapunov orbit  T={T:.3f}')
plt.scatter([-mu], [0], s=800, c='gold', marker='o', ec='orange',
            label='Sun', zorder=5)
plt.scatter([1-mu], [0], s=150, c='sandybrown', marker='o', ec='saddlebrown',
            label='Jupiter', zorder=5)
for lx, name in [(x_L1, '$L_1$'), (x_L2, '$L_2$'), (x_L3, '$L_3$')]:
    plt.scatter([lx], [0], s=80, c='k', marker='X', zorder=6)
    plt.annotate(name, (lx, 0), xytext=(5, 8), textcoords='offset points',
                 fontsize=12, fontweight='bold')
plt.xlabel('x (rotating frame)'); plt.ylabel('y (rotating frame)')
plt.title('Hill region (forbidden zone shaded) + Lyapunov orbit near $L_2$')
plt.axis('equal'); plt.grid(alpha=0.3); plt.legend(loc='lower left')
savefig('01_hill_region.png')

# ============================================================
# FIG 2: Zoom on L2 + orbit
# ============================================================
fig = plt.figure(figsize=(10, 8))
plt.plot(U_t[0], U_t[1], 'b-', lw=2.5)
plt.scatter([x_L2], [0], s=200, marker='X', c='k', label='$L_2$', zorder=5)
plt.scatter([U_t[0, best]], [U_t[1, best]], s=300, marker='*',
            c='green', label='cheapest correction', zorder=6)
plt.scatter([U_t[0, worst]], [U_t[1, worst]], s=200, marker='s',
            c='red', label='most expensive', zorder=6)
# velocity arrows along orbit
N_arrows = 24
idx_arrows = np.linspace(0, len(t_grid)-1, N_arrows, dtype=int)
for i in idx_arrows:
    plt.arrow(U_t[0, i], U_t[1, i],
              U_t[3, i]*0.15, U_t[4, i]*0.15,
              head_width=1e-4, head_length=1.5e-4, fc='gray', ec='gray',
              alpha=0.6, length_includes_head=True)
plt.xlabel('x'); plt.ylabel('y')
plt.title('Orbit with velocity field\n(gray arrows = instantaneous velocity)')
plt.axis('equal'); plt.grid(alpha=0.3); plt.legend()
savefig('02_orbit_zoom.png')

# ============================================================
# FIG 3: Phase portrait x - vx
# ============================================================
fig = plt.figure(figsize=(9, 7))
plt.plot(U_t[0] - x_L2, U_t[3], 'b-', lw=2)
plt.scatter([0], [0], c='k', s=80, marker='X', label='$L_2$')
plt.scatter([U_t[0, best]-x_L2], [U_t[3, best]], c='green', s=150,
            marker='*', label='$t^*$', zorder=5)
plt.xlabel('$x - x_{L_2}$'); plt.ylabel('$\\dot{x}$')
plt.title('Phase portrait  (x - vx)')
plt.grid(alpha=0.4); plt.legend()
savefig('03_phase_x_vx.png')

# ============================================================
# FIG 4: Phase portrait y - vy
# ============================================================
fig = plt.figure(figsize=(9, 7))
plt.plot(U_t[1], U_t[4], 'b-', lw=2)
plt.scatter([0], [0], c='k', s=80, marker='X')
plt.scatter([U_t[1, best]], [U_t[4, best]], c='green', s=150, marker='*',
            label='$t^*$', zorder=5)
plt.xlabel('$y$'); plt.ylabel('$\\dot{y}$')
plt.title('Phase portrait  (y - vy)')
plt.grid(alpha=0.4); plt.legend()
savefig('04_phase_y_vy.png')

# ============================================================
# FIG 5: Time series of state components
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
labels = ['$x - x_{L_2}$', '$y$', '$\\dot x$', '$\\dot y$']
series = [U_t[0]-x_L2, U_t[1], U_t[3], U_t[4]]
for ax, lab, s in zip(axes, labels, series):
    ax.plot(t_grid/T, s, lw=2)
    ax.axvline(t_grid[best]/T, color='green', ls='--', alpha=0.7)
    ax.set_ylabel(lab); ax.grid(alpha=0.3)
axes[-1].set_xlabel('t / T'); axes[0].set_title('State components along one period')
savefig('05_time_series.png')

# ============================================================
# FIG 6: Jacobi constant conservation
# ============================================================
fig = plt.figure(figsize=(10, 5))
plt.plot(t_grid/T, C_t - C_t[0], lw=2)
plt.xlabel('t / T'); plt.ylabel('$C(t) - C(0)$')
plt.title(f'Jacobi integral conservation  (peak-to-peak drift: {np.ptp(C_t):.1e})')
plt.grid(alpha=0.4)
savefig('06_jacobi_conservation.png')

# ============================================================
# FIG 7: Monodromy eigenvalues on complex plane (log scale)
# ============================================================
fig = plt.figure(figsize=(9, 8))
theta_c = np.linspace(0, 2*np.pi, 200)
plt.plot(np.cos(theta_c), np.sin(theta_c), 'k--', alpha=0.5,
         label='unit circle')
for ev in eigM:
    r = abs(ev); a = np.angle(ev)
    # Display in log-radial coordinates for clarity
    plt.scatter(np.log10(r+1e-30)*np.cos(a) if r>1.5 else ev.real,
                np.log10(r+1e-30)*np.sin(a) if r>1.5 else ev.imag,
                s=200, edgecolors='black')
plt.axhline(0, color='gray', lw=0.5); plt.axvline(0, color='gray', lw=0.5)
plt.xlabel('Re'); plt.ylabel('Im')
plt.title(f'Monodromy spectrum  (λ_u ≈ {lam_u:.1f}, λ_s ≈ {lam_s:.1e})')
plt.axis('equal'); plt.grid(alpha=0.3)
savefig('07_monodromy_spectrum_linear.png')

# Version 2 — just real line plot because eigenvalues are widely separated
fig, ax = plt.subplots(figsize=(11, 4))
pos = [abs(e) for e in eigM]
ax.scatter(range(len(eigM)), pos, s=150)
for i, e in enumerate(eigM):
    ax.annotate(f'{e.real:+.3f}{e.imag:+.3f}j', (i, abs(e)),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax.set_yscale('log')
ax.axhline(1, color='red', ls='--', label='unit circle')
ax.set_xlabel('eigenvalue index'); ax.set_ylabel('|λ|')
ax.set_title('Monodromy eigenvalues by magnitude (log)')
ax.grid(alpha=0.3); ax.legend()
savefig('07b_monodromy_magnitudes.png')

# ============================================================
# FIG 8: Correction leverage ||L_v(t)|| along orbit
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(t_grid/T, norm_Lv, lw=2.5)
ax.axvline(t_grid[best]/T, color='green', ls='--',
           label=f'best t*/T={t_grid[best]/T:.3f}')
ax.axvline(t_grid[worst]/T, color='red', ls='--',
           label=f'worst t/T={t_grid[worst]/T:.3f}')
ax.fill_between(t_grid/T, norm_Lv, alpha=0.2)
ax.set_xlabel('t / T'); ax.set_ylabel('$\\|L_v(t)\\|$')
ax.set_title('Correction leverage along orbit\n(higher = cheaper correction)')
ax.grid(alpha=0.4); ax.legend()
savefig('08_leverage.png')

# ============================================================
# FIG 9: Correction cost 1/||L_v(t)|| (log)
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5))
ax.semilogy(t_grid/T, 1.0/norm_Lv, lw=2.5, color='purple')
ax.axvline(t_grid[best]/T, color='green', ls='--')
ax.set_xlabel('t / T')
ax.set_ylabel('$|\\Delta v| / |\\alpha_u|$  (log scale)')
ax.set_title('Correction cost per unit unstable amplitude')
ax.grid(alpha=0.4, which='both')
savefig('09_cost_log.png')

# ============================================================
# FIG 10: Orbit colored by correction cost
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))
points = np.array([U_t[0], U_t[1]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
cost = 1.0 / norm_Lv
lc = LineCollection(segments, cmap='viridis_r',
                    norm=plt.Normalize(np.log10(cost).min(),
                                       np.log10(cost).max()))
lc.set_array(np.log10(cost[:-1])); lc.set_linewidth(4)
ax.add_collection(lc)
cbar = plt.colorbar(lc, ax=ax)
cbar.set_label('$\\log_{10}$ correction cost')
ax.scatter([x_L2], [0], s=200, marker='X', c='black', label='$L_2$')
ax.scatter([U_t[0, best]], [U_t[1, best]], marker='*', s=300,
           c='lime', edgecolors='black', label='cheapest', zorder=5)
ax.scatter([U_t[0, worst]], [U_t[1, worst]], marker='s', s=150,
           c='red', edgecolors='black', label='most expensive', zorder=5)
ax.set_xlim(U_t[0].min()-1e-3, U_t[0].max()+1e-3)
ax.set_ylim(U_t[1].min()-1e-3, U_t[1].max()+1e-3)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Orbit colored by correction cost\n(dark = cheap, bright = expensive)')
ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend()
savefig('10_orbit_cost_heatmap.png')

# ============================================================
# FIG 11: Unstable direction field (position part) along orbit
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(U_t[0], U_t[1], 'b-', lw=1.5, alpha=0.5, label='orbit')
idx_arr = np.linspace(0, len(t_grid)-1, 40, dtype=int)
scale = 2e-3 / np.max(np.linalg.norm(eu_t[:2, idx_arr], axis=0))
for i in idx_arr:
    dx = eu_t[0, i] * scale; dy = eu_t[1, i] * scale
    ax.arrow(U_t[0, i], U_t[1, i], dx, dy,
             head_width=1.5e-4, head_length=2e-4, fc='red', ec='red',
             alpha=0.8, length_includes_head=True)
ax.scatter([x_L2], [0], s=100, marker='X', c='black', label='$L_2$')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Unstable direction field $e_u(t)$ along orbit\n(position components — where perturbation grows)')
ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend()
savefig('11_unstable_direction.png')

# ============================================================
# FIG 12: Divergence without correction
# ============================================================
alpha0 = 1e-6
n_periods = 3
Y0p = Y0[:6] + alpha0 * e_u
t_long = np.linspace(0, n_periods*T, 4000)
sol_div = solve_ivp(cr3bp, (0, n_periods*T), Y0p, t_eval=t_long,
                    rtol=1e-13, atol=1e-15)
# Reference orbit propagated over n_periods
U_ref_long = np.tile(U_t, n_periods)[:, :t_long.size]
# Actually just re-integrate
sol_ref = solve_ivp(cr3bp, (0, n_periods*T), Y0[:6], t_eval=t_long,
                    rtol=1e-13, atol=1e-15)
dist = np.linalg.norm(sol_div.y - sol_ref.y, axis=0)

fig, axes = plt.subplots(2, 1, figsize=(11, 8))
axes[0].plot(sol_ref.y[0], sol_ref.y[1], 'b-', lw=1.5, alpha=0.7, label='reference')
axes[0].plot(sol_div.y[0], sol_div.y[1], 'r-', lw=1.2, alpha=0.9,
             label=f'perturbed (α₀={alpha0:.0e})')
axes[0].scatter([x_L2], [0], marker='X', c='k', s=80)
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
axes[0].set_title(f'Divergence of perturbed orbit over {n_periods} periods — NO correction')
axes[0].set_aspect('equal'); axes[0].grid(alpha=0.3); axes[0].legend()

axes[1].semilogy(sol_div.t/T, dist, lw=2)
for k in range(1, n_periods+1):
    axes[1].axvline(k, color='gray', ls=':', alpha=0.5)
    axes[1].axhline(alpha0 * lam_u**k, color='orange', ls='--', alpha=0.5)
axes[1].set_xlabel('t / T'); axes[1].set_ylabel('$\\| \\delta U(t) \\|$ (log)')
axes[1].set_title(f'Growth of perturbation — theory: λ_u^k = {lam_u:.0f}, {lam_u**2:.0f}, {lam_u**3:.0f}')
axes[1].grid(alpha=0.4, which='both')
savefig('12_divergence_no_correction.png')

# ============================================================
# FIG 13: With correction vs without, many periods
# ============================================================
n_per = 5
# Without correction
sol_no = solve_ivp(cr3bp, (0, n_per*T), Y0p,
                   t_eval=np.linspace(0, n_per*T, 6000),
                   rtol=1e-13, atol=1e-15)
sol_ref_long = solve_ivp(cr3bp, (0, n_per*T), Y0[:6],
                         t_eval=sol_no.t,
                         rtol=1e-13, atol=1e-15)
err_no = np.linalg.norm(sol_no.y - sol_ref_long.y, axis=0)

# With correction at each perpendicular x-crossing (t*=0 of each period)
U_current = Y0p.copy()
t_hist, err_hist = [0.0], [alpha0]
dv_hist = []
for k in range(n_per):
    t_end = (k + 1) * T
    s = solve_ivp(cr3bp, (k*T, t_end), U_current,
                  t_eval=np.linspace(k*T, t_end, 1200),
                  rtol=1e-13, atol=1e-15)
    # reference at matching times
    s_ref = solve_ivp(cr3bp, (k*T, t_end), U_t[:, 0],
                      t_eval=s.t, rtol=1e-13, atol=1e-15)
    for i in range(s.y.shape[1]):
        t_hist.append(s.t[i])
        err_hist.append(np.linalg.norm(s.y[:, i] - s_ref.y[:, i]))
    U_current = s.y[:, -1]
    # apply correction at t = (k+1)*T  (best phase, t*/T=0 of next period)
    alpha_now = w_u @ (U_current - U_t[:, 0])
    dv = -alpha_now * w_u[3:] / (w_u[3:] @ w_u[3:])
    U_current[3:] += dv
    dv_hist.append(np.linalg.norm(dv))
    t_hist.append(t_end + 1e-6)
    err_hist.append(np.linalg.norm(U_current - U_t[:, 0]))

fig, axes = plt.subplots(2, 1, figsize=(11, 8))
axes[0].semilogy(sol_no.t/T, err_no, 'r-', lw=2, label='no correction')
axes[0].semilogy(np.array(t_hist)/T, err_hist, 'g-', lw=2,
                 label='corrected each period')
for k in range(1, n_per+1):
    axes[0].axvline(k, color='gray', ls=':', alpha=0.3)
axes[0].set_xlabel('t / T'); axes[0].set_ylabel('$\\| \\delta U \\|$ (log)')
axes[0].set_title(f'Station-keeping over {n_per} periods')
axes[0].grid(alpha=0.4, which='both'); axes[0].legend()

axes[1].bar(range(1, n_per+1), dv_hist, color='steelblue')
axes[1].set_xlabel('period #'); axes[1].set_ylabel('$|\\Delta v|$')
axes[1].set_title(f'Correction budget  (total $\\Sigma|\\Delta v|$ = {sum(dv_hist):.3e})')
axes[1].grid(alpha=0.3, axis='y')
savefig('13_station_keeping.png')

# ============================================================
# FIG 14: |Δv| vs application phase — for fixed perturbation
# ============================================================
dv_at_phase = []
for i in range(0, len(t_grid), 30):
    s_pre = solve_ivp(cr3bp, (0, max(t_grid[i], 1e-9)), Y0p,
                      rtol=1e-12, atol=1e-14,
                      t_eval=[max(t_grid[i], 1e-9)])
    U_pre = s_pre.y[:, -1] if s_pre.y.size else Y0p
    alpha = L_t[:, i] @ (U_pre - U_t[:, i])
    Lv = L_t[3:, i]
    dv = -alpha * Lv / (Lv @ Lv)
    dv_at_phase.append((t_grid[i]/T, np.linalg.norm(dv)))
phases, dvs = zip(*dv_at_phase)
fig, ax = plt.subplots(figsize=(11, 5))
ax.semilogy(phases, dvs, 'o-', lw=2, markersize=4)
ax.axvline(t_grid[best]/T, color='green', ls='--',
           label=f'theoretical min')
ax.set_xlabel('correction phase  t / T')
ax.set_ylabel('$|\\Delta v|$  (log)')
ax.set_title('Size of single-shot correction vs where you apply it\n(perturbation seeded at t=0)')
ax.grid(alpha=0.4, which='both'); ax.legend()
savefig('14_dv_vs_phase.png')

# ============================================================
# FIG 15: Work-energy identity scatter  ΔC vs -2 ΔKE
# ============================================================
rng = np.random.default_rng(42)
dKEs, dCs = [], []
for _ in range(500):
    U = U_t[:, rng.integers(0, U_t.shape[1])]
    dv = rng.normal(0, 1e-4, 3)
    U_after = U.copy(); U_after[3:] += dv
    v_before = U[3:]
    dKE = v_before @ dv + 0.5*(dv @ dv)
    dC = jacobi(U_after) - jacobi(U)
    dKEs.append(dKE); dCs.append(dC)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(dKEs, dCs, s=15, alpha=0.6)
xr2 = np.array([min(dKEs), max(dKEs)])
ax.plot(xr2, -2*xr2, 'r-', lw=2, label='$\\Delta C = -2 \\Delta KE$')
ax.set_xlabel('$\\Delta KE$ (mechanical work on spacecraft)')
ax.set_ylabel('$\\Delta C$ (Jacobi integral change)')
ax.set_title('Work-energy identity in rotating frame\n500 random impulses across orbit')
ax.grid(alpha=0.4); ax.legend(); ax.set_aspect('equal')
savefig('15_work_energy_identity.png')

# ============================================================
# FIG 16: Polar view — leverage as radius vs phase
# ============================================================
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='polar')
theta = 2*np.pi * t_grid / T
ax.plot(theta, norm_Lv, 'b-', lw=2)
ax.fill(theta, norm_Lv, alpha=0.25)
ax.plot([2*np.pi*t_grid[best]/T], [norm_Lv[best]], 'g*',
        markersize=25, markeredgecolor='black')
ax.set_title('Correction leverage (polar view)\nangle = phase along orbit, radius = $\\|L_v\\|$')
savefig('16_leverage_polar.png')

# ============================================================
# FIG 17: 3D view with orbit + L2 + unstable direction
# ============================================================
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot(U_t[0], U_t[1], np.zeros_like(U_t[0]), 'b-', lw=2,
        label='Lyapunov orbit (planar)')
ax.scatter([x_L2], [0], [0], s=100, marker='X', c='k', label='$L_2$')
ax.scatter([U_t[0, best]], [U_t[1, best]], [0], s=200, marker='*',
           c='green', label='optimal correction point')
# show unstable direction sticks up from orbit
idx_a = np.linspace(0, len(t_grid)-1, 15, dtype=int)
scale_z = 1e-3
for i in idx_a:
    ax.plot([U_t[0, i], U_t[0, i] + eu_t[0, i]*scale_z],
            [U_t[1, i], U_t[1, i] + eu_t[1, i]*scale_z],
            [0, eu_t[2, i]*scale_z if abs(eu_t[2,i])>1e-10 else 0.0],
            'r-', lw=1, alpha=0.5)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('$e_u^z$')
ax.set_title('3D view: orbit + unstable-direction probes')
ax.legend()
savefig('17_3d_view.png')

# ============================================================
# FIG 18: Propagation of reference vs perturbed (phase plane)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))
alpha0_big = 5e-5
Y0p_big = Y0[:6] + alpha0_big * e_u
sol_big = solve_ivp(cr3bp, (0, 2*T), Y0p_big,
                    t_eval=np.linspace(0, 2*T, 4000),
                    rtol=1e-13, atol=1e-15)
sol_ref_big = solve_ivp(cr3bp, (0, 2*T), Y0[:6],
                        t_eval=sol_big.t, rtol=1e-13, atol=1e-15)
ax.plot(sol_ref_big.y[0], sol_ref_big.y[1], 'b-', lw=2, label='reference')
ax.plot(sol_big.y[0], sol_big.y[1], 'r-', lw=1.5, alpha=0.8,
        label=f'perturbed α₀={alpha0_big:.0e}')
ax.scatter([x_L2], [0], marker='X', c='k', s=80)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Phase portrait: reference vs perturbed over 2 periods')
ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend()
savefig('18_reference_vs_perturbed.png')

# ============================================================
# FIG 19: Instantaneous speed |v|(t) along orbit + Jacobi decomposition
# ============================================================
speed = np.sqrt(U_t[3]**2 + U_t[4]**2 + U_t[5]**2)
R1v, R2v = np.array([_radii(U_t[:, i]) for i in range(U_t.shape[1])]).T
Omega_t = 0.5*(U_t[0]**2 + U_t[1]**2) + (1-mu)/R1v + mu/R2v + 0.5*mu*(1-mu)
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
axes[0].plot(t_grid/T, speed, lw=2); axes[0].set_ylabel('$|v|$')
axes[0].grid(alpha=0.3)
axes[1].plot(t_grid/T, 2*Omega_t, lw=2, label='$2\\Omega$')
axes[1].plot(t_grid/T, speed**2, lw=2, label='$|v|^2$')
axes[1].set_ylabel('components'); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[2].plot(t_grid/T, 2*Omega_t - speed**2, lw=2)
axes[2].axhline(C_t[0], color='red', ls='--', label=f'$C_0$={C_t[0]:.4f}')
axes[2].set_ylabel('$C = 2\\Omega - |v|^2$')
axes[2].set_xlabel('t / T'); axes[2].legend(); axes[2].grid(alpha=0.3)
axes[0].set_title('Jacobi decomposition along orbit: $C = 2\\Omega - |v|^2$')
savefig('19_jacobi_decomposition.png')

# ============================================================
# FIG 20: Overview dashboard
# ============================================================
fig = plt.figure(figsize=(16, 10))

ax = fig.add_subplot(2, 3, 1)
ax.plot(U_t[0], U_t[1], 'b-', lw=2)
ax.scatter([x_L2], [0], marker='X', c='k', s=100, label='$L_2$')
ax.scatter([U_t[0, best]], [U_t[1, best]], marker='*', c='green', s=200)
ax.set_title('Orbit'); ax.set_aspect('equal'); ax.grid(alpha=0.3)

ax = fig.add_subplot(2, 3, 2)
ax.plot(t_grid/T, norm_Lv, lw=2)
ax.axvline(t_grid[best]/T, color='green', ls='--')
ax.set_title('Leverage $\\|L_v(t)\\|$'); ax.grid(alpha=0.3)

ax = fig.add_subplot(2, 3, 3)
ax.plot(t_grid/T, C_t - C_t[0], lw=2)
ax.set_title(f'Jacobi drift (ptp {np.ptp(C_t):.1e})'); ax.grid(alpha=0.3)

ax = fig.add_subplot(2, 3, 4)
ax.semilogy(sol_no.t/T, err_no, 'r-', lw=2, label='no corr.')
ax.semilogy(np.array(t_hist)/T, err_hist, 'g-', lw=2, label='corrected')
ax.set_title('Station-keeping'); ax.legend(); ax.grid(alpha=0.3, which='both')

ax = fig.add_subplot(2, 3, 5)
ax.scatter(dKEs, dCs, s=10, alpha=0.5)
ax.plot(xr2, -2*xr2, 'r-', lw=2)
ax.set_title('$\\Delta C = -2\\Delta KE$'); ax.grid(alpha=0.3)

ax = fig.add_subplot(2, 3, 6, projection='polar')
ax.plot(theta, norm_Lv, lw=2); ax.fill(theta, norm_Lv, alpha=0.25)
ax.set_title('Leverage (polar)')

fig.suptitle('Sun-Jupiter $L_2$ Lyapunov orbit — correction strategy dashboard',
             fontsize=14)
savefig('20_dashboard.png')

print(f"\nAll figures saved to {SAVE_DIR}")
