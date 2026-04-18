import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# CLOSED LYAPUNOV ORBIT AT SUN-JUPITER L2
# + MINIMUM-ENERGY CORRECTION STRATEGY
# ============================================================
# (1) Differential correction -> periodic orbit closed in one revolution.
# (2) Monodromy / Floquet -> cheapest correction point on the orbit.
# (3) Verification: work-energy theorem in the rotating frame
#     (delta C_Jacobi = -2 * delta KE for impulsive dv).

mu = 9.53e-4

# ------------------------------------------------------------
# 1. CORE CR3BP PRIMITIVES
# ------------------------------------------------------------
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
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1],
        [Oxx,Oxy,Oxz,0,2,0],
        [Oxy,Oyy,Oyz,-2,0,0],
        [Oxz,Oyz,Ozz,0,0,0],
    ], dtype=float)

def var_rhs(t, Y):
    U = Y[:6]
    Phi = Y[6:].reshape(6, 6)
    dU = cr3bp(t, U)
    dPhi = A_matrix(U) @ Phi
    return np.concatenate([dU, dPhi.flatten()])

def find_L2():
    def f(x):
        R1 = abs(x + mu); R2 = abs(x - (1 - mu))
        return x - (1-mu)*(x+mu)/R1**3 - mu*(x-(1-mu))/R2**3
    return root_scalar(f, bracket=[1-mu+1e-4, 1.5], method='brentq').root

x_L2 = find_L2()

# ------------------------------------------------------------
# 2. DIFFERENTIAL CORRECTION -> PLANAR LYAPUNOV (closes in 1 rev)
# ------------------------------------------------------------
# Fix x0; adjust vy0 so that at first y=0 return we have vx=0.
# (Planar case: z=vz=0 preserved exactly; only vx constraint is nontrivial.)

def half_period(x0, vy0, t_max=6.0):
    Y0 = np.zeros(42)
    Y0[:6] = [x0, 0.0, 0.0, 0.0, vy0, 0.0]
    Y0[6:] = np.eye(6).flatten()
    # direction matches sign of initial vy: we want the FIRST return to y=0,
    # not the degenerate event at t=0.
    direction = int(np.sign(vy0))
    def hit_y(t, Y): return Y[1]
    hit_y.terminal = True
    hit_y.direction = -direction  # returning crossing has opposite sign
    sol = solve_ivp(var_rhs, (1e-6, t_max), Y0, events=hit_y,
                    rtol=1e-12, atol=1e-14)
    if not sol.t_events[0].size:
        raise RuntimeError("No y=0 return; bad initial guess")
    return sol.t_events[0][0], sol.y_events[0][0]

def diff_correct_lyapunov(x0, vy0, tol=1e-12, maxiter=40):
    for _ in range(maxiter):
        T_half, Yf = half_period(x0, vy0)
        U = Yf[:6]
        Phi = Yf[6:].reshape(6, 6)
        vx_f = U[3]
        if abs(vx_f) < tol:
            return vy0, T_half, U, Phi
        dU = cr3bp(T_half, U)
        ydot, ax_f = dU[1], dU[3]
        # d(vx_f)/d(vy0) with fixed x0, accounting for variable T:
        # delta_vx_f = Phi[3,4] * d_vy0 + ax_f * dT,  dT = -Phi[1,4]/ydot * d_vy0
        coef = Phi[3, 4] - ax_f * Phi[1, 4] / ydot
        vy0 -= vx_f / coef
    raise RuntimeError("Differential correction did not converge")

x0 = x_L2 + 2.0e-3
# Linearization at L2 gives vy0 ~ -((omega_p^2 + Oxx)/2) * (x0 - x_L2).
A_L2 = A_matrix([x_L2, 0.0, 0.0, 0.0, 0.0, 0.0])
Oxx_L2 = A_L2[3, 0]
ev_inplane = np.linalg.eigvals(A_L2[[0, 1, 3, 4]][:, [0, 1, 3, 4]])
omega_p = float(np.max(np.abs(ev_inplane.imag)))
vy0_guess = -0.5 * (omega_p**2 + Oxx_L2) * (x0 - x_L2)
vy0, T_half, _, _ = diff_correct_lyapunov(x0, vy0_guess)
T = 2.0 * T_half
if T_half < 0.1:
    raise RuntimeError(f"Spurious convergence: T_half = {T_half:.3e}")

print("="*60)
print("PERIODIC LYAPUNOV ORBIT AT L2")
print("="*60)
print(f"x_L2     = {x_L2:.10f}")
print(f"x0       = {x0:.10f}  (fixed)")
print(f"vy0      = {vy0:.10f}  (corrected)")
print(f"Period T = {T:.6f}")

# ------------------------------------------------------------
# 3. FULL-PERIOD INTEGRATION + MONODROMY
# ------------------------------------------------------------
Y0 = np.zeros(42)
Y0[:6] = [x0, 0.0, 0.0, 0.0, vy0, 0.0]
Y0[6:] = np.eye(6).flatten()

t_grid = np.linspace(0, T, 4000)
sol = solve_ivp(var_rhs, (0, T), Y0, t_eval=t_grid,
                rtol=1e-13, atol=1e-15)

U_t   = sol.y[:6, :]
Phi_t = sol.y[6:, :].reshape(6, 6, -1)

# Jacobi conservation check
C_t = np.array([jacobi(U_t[:, i]) for i in range(U_t.shape[1])])
print(f"\nJacobi drift over one period: {np.ptp(C_t):.2e}")
print(f"Closure error |U(T)-U(0)|  : {np.linalg.norm(U_t[:,-1]-U_t[:,0]):.2e}")

M = Phi_t[:, :, -1]
eigM, VR = np.linalg.eig(M)
print("\nMonodromy eigenvalues (|lambda|):")
for e in eigM:
    print(f"  lambda = {e.real:+.6f}{e.imag:+.6f}j   |lambda| = {abs(e):.6f}")

# Pick unstable mode
idx_u = int(np.argmax(np.abs(eigM)))
lam_u = eigM[idx_u]
e_u   = np.real_if_close(VR[:, idx_u]).real
e_u  /= np.linalg.norm(e_u)
print(f"\nUnstable eigenvalue : {lam_u.real:.4f}  (growth factor per period)")

# Left eigenvector (dual basis) via eig of M^T
eigMT, VL = np.linalg.eig(M.T)
idx_uL = int(np.argmin(np.abs(eigMT - lam_u)))
w_u = np.real_if_close(VL[:, idx_uL]).real
w_u /= (w_u @ e_u)   # biorthogonal normalization: w_u^T e_u = 1

# ------------------------------------------------------------
# 4. COST FUNCTION ALONG THE ORBIT
# ------------------------------------------------------------
# L(t) = Phi(t,0)^{-T} w_u  is the left eigenvector field.
# For perturbation dU(t) the unstable-mode amplitude is  alpha = L(t)^T dU(t).
# Correction Dv in velocity subspace kills alpha with:
#   Dv_min = -alpha * L_v(t) / ||L_v(t)||^2     ->   |Dv| = |alpha| / ||L_v(t)||
# => CHEAPEST CORRECTION POINT = argmax_t ||L_v(t)||.

L_t = np.empty((6, t_grid.size))
for i in range(t_grid.size):
    L_t[:, i] = np.linalg.solve(Phi_t[:, :, i].T, w_u)

norm_Lv = np.linalg.norm(L_t[3:, :], axis=0)
best = int(np.argmax(norm_Lv))
worst = int(np.argmin(norm_Lv))

print("\n" + "="*60)
print("MIN-ENERGY CORRECTION STRATEGY")
print("="*60)
print(f"Best  t*/T = {t_grid[best]/T:.4f}   ||L_v|| = {norm_Lv[best]:.4f}")
print(f"Worst t /T = {t_grid[worst]/T:.4f}   ||L_v|| = {norm_Lv[worst]:.4f}")
print(f"Leverage ratio (worst/best): {norm_Lv[best]/norm_Lv[worst]:.2f}x")
print(f"\nBest-point state U(t*):")
print(f"  r = ({U_t[0,best]:.6f}, {U_t[1,best]:+.6e}, 0)")
print(f"  v = ({U_t[3,best]:+.6e}, {U_t[4,best]:+.6e}, 0)")

# ------------------------------------------------------------
# 5. DEMONSTRATION: inject perturbation, correct, verify
# ------------------------------------------------------------
alpha0 = 1e-5
dU0 = alpha0 * e_u                          # pure unstable seed
Y0_pert = Y0.copy()
Y0_pert[:6] += dU0

def propagate_state(U_start, t_end):
    if t_end <= 1e-12:
        return np.array(U_start, dtype=float)
    s = solve_ivp(cr3bp, (0.0, t_end), U_start,
                  rtol=1e-13, atol=1e-15, t_eval=[t_end])
    return s.y[:, -1]

def correction_cost(U_pert_0, t_target, i_target):
    U_now = propagate_state(U_pert_0, t_target)
    alpha_now = L_t[:, i_target] @ (U_now - U_t[:, i_target])
    Lv = L_t[3:, i_target]
    dv = -alpha_now * Lv / (Lv @ Lv)
    return U_now, dv, alpha_now

U_best,  dv_best,  a_best  = correction_cost(Y0_pert[:6], t_grid[best],  best)
U_worst, dv_worst, a_worst = correction_cost(Y0_pert[:6], t_grid[worst], worst)

# ------------------------------------------------------------
# 6. VERIFICATION VIA WORK-ENERGY RELATION (rotating frame)
# ------------------------------------------------------------
# Impulsive dv:  v -> v + dv
# delta KE   =  v.dv + 0.5 |dv|^2
# delta C    = -(|v+dv|^2 - |v|^2)  =  -2 v.dv - |dv|^2  =  -2 * delta KE

def verify_energy(U_before, dv):
    v_before = U_before[3:]
    U_after = U_before.copy(); U_after[3:] += dv
    dKE = v_before @ dv + 0.5*(dv @ dv)
    dC_measured  = jacobi(U_after) - jacobi(U_before)
    dC_predicted = -2.0 * dKE
    return dKE, dC_measured, dC_predicted

print("\n" + "="*60)
print("VERIFICATION  (work-energy theorem)")
print("="*60)
for label, U, dv in [("BEST  t*", U_best, dv_best),
                     ("WORST t ", U_worst, dv_worst)]:
    dKE, dCm, dCp = verify_energy(U, dv)
    print(f"\n[{label}]")
    print(f"  |dv|              = {np.linalg.norm(dv):.6e}")
    print(f"  delta KE (work)   = {dKE:+.6e}")
    print(f"  delta C measured  = {dCm:+.6e}")
    print(f"  delta C = -2*dKE  = {dCp:+.6e}")
    print(f"  residual          = {abs(dCm - dCp):.2e}")

print(f"\nCOST COMPARISON")
print(f"  |dv| at t* (best)  = {np.linalg.norm(dv_best):.4e}")
print(f"  |dv| at worst t    = {np.linalg.norm(dv_worst):.4e}")
print(f"  Savings factor     = {np.linalg.norm(dv_worst)/np.linalg.norm(dv_best):.2f}x")

# ------------------------------------------------------------
# 7. PLOTS
# ------------------------------------------------------------
fig = plt.figure(figsize=(13, 9))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(U_t[0], U_t[1], lw=2, label='closed Lyapunov orbit')
ax1.scatter([x_L2], [0], c='k', s=40, label='$L_2$')
ax1.scatter([U_t[0, best]],  [U_t[1, best]],  c='red',  s=80, label='min $|\\Delta v|$')
ax1.scatter([U_t[0, worst]], [U_t[1, worst]], c='blue', s=80, label='max $|\\Delta v|$')
ax1.set_xlabel('x'); ax1.set_ylabel('y')
ax1.set_title('Periodic orbit and correction points')
ax1.axis('equal'); ax1.grid(True); ax1.legend()

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(t_grid/T, norm_Lv, lw=2)
ax2.axvline(t_grid[best]/T,  color='red',  ls='--', label='cheapest')
ax2.axvline(t_grid[worst]/T, color='blue', ls='--', label='most expensive')
ax2.set_xlabel('t / T'); ax2.set_ylabel(r'$\|L_v(t)\|$')
ax2.set_title('Correction leverage (higher = cheaper)')
ax2.grid(True); ax2.legend()

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(t_grid/T, C_t - C_t[0], lw=2)
ax3.set_xlabel('t / T'); ax3.set_ylabel(r'$C(t)-C(0)$')
ax3.set_title('Jacobi integral conservation')
ax3.grid(True)

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(t_grid/T, 1.0/norm_Lv, lw=2, color='purple')
ax4.set_xlabel('t / T'); ax4.set_ylabel(r'$|\Delta v| / |\alpha_u|$')
ax4.set_title('Correction cost per unit unstable amplitude')
ax4.grid(True)

plt.tight_layout()
plt.show()
