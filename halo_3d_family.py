import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm

# ============================================================
# 3D HALO, FAMILY CONTINUATION, INVARIANT MANIFOLDS,
# MULTI-SHOOTING STATION-KEEPING
# Sun-Jupiter CR3BP, mu = 9.53e-4
# ============================================================

SAVE = '/Users/aleksandrvolkov/2/figs'
os.makedirs(SAVE, exist_ok=True)

def savefig(name, fig=None):
    path = os.path.join(SAVE, name)
    (fig or plt.gcf()).savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig or plt.gcf())
    print(f"  saved {name}")

# ------------------------------------------------------------
# DYNAMICS
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
    U = Y[:6]; Phi = Y[6:].reshape(6, 6)
    return np.concatenate([cr3bp(t, U), (A_matrix(U) @ Phi).flatten()])

def find_L2():
    def f(x):
        R1 = abs(x + mu); R2 = abs(x - (1 - mu))
        return x - (1-mu)*(x+mu)/R1**3 - mu*(x-(1-mu))/R2**3
    return root_scalar(f, bracket=[1-mu+1e-4, 1.5], method='brentq').root

x_L2 = find_L2()

# ------------------------------------------------------------
# DIFFERENTIAL CORRECTION: 3D HALO
# Fix Az = z0, adjust (x0, vy0) so that vx=vz=0 at first y=0 return.
# ------------------------------------------------------------
def half_period_3d(x0, z0, vy0, t_max=10.0):
    """Multi-revolution 3D periodic orbit: catch first perpendicular y=0 return."""
    Y0 = np.zeros(42); Y0[:6] = [x0, 0, z0, 0, vy0, 0]; Y0[6:] = np.eye(6).flatten()
    direction = int(np.sign(vy0)) if vy0 != 0 else 1
    def hit_y(t, Y): return Y[1]
    hit_y.terminal = True; hit_y.direction = -direction
    sol = solve_ivp(var_rhs, (1e-6, t_max), Y0, events=hit_y,
                    rtol=1e-11, atol=1e-13, max_step=0.05)
    if not sol.t_events[0].size:
        raise RuntimeError(f"no y=0 return within t_max={t_max}")
    return sol.t_events[0][0], sol.y_events[0][0]

def halo_dc(x0, z0, vy0, tol=1e-10, maxiter=60, damp=0.8, step_max=5e-3):
    err = np.inf
    for it in range(maxiter):
        Tf, Yf = half_period_3d(x0, z0, vy0)
        U = Yf[:6]; Phi = Yf[6:].reshape(6, 6)
        vx_f, vz_f = U[3], U[5]
        err = max(abs(vx_f), abs(vz_f))
        if err < tol:
            return x0, vy0, Tf, err
        dU = cr3bp(Tf, U)
        ydot = dU[1]
        if abs(ydot) < 1e-10:
            raise RuntimeError("ydot≈0 at crossing")
        ax_f, az_f = dU[3], dU[5]
        M = np.array([
            [Phi[3,0] - ax_f*Phi[1,0]/ydot, Phi[3,4] - ax_f*Phi[1,4]/ydot],
            [Phi[5,0] - az_f*Phi[1,0]/ydot, Phi[5,4] - az_f*Phi[1,4]/ydot]
        ])
        rhs = np.array([-vx_f, -vz_f])
        delta = np.linalg.solve(M, rhs)
        # trust-region step limiting
        nrm = np.linalg.norm(delta)
        if nrm > step_max:
            delta *= step_max / nrm
        x0  += damp * delta[0]
        vy0 += damp * delta[1]
    raise RuntimeError(f"DC failed: err={err:.2e}")

# ------------------------------------------------------------
# FAMILY CONTINUATION: sweep Az via natural-parameter continuation
# ------------------------------------------------------------
print("="*60)
print("FAMILY CONTINUATION")
print("="*60)

Az_list = np.array([0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005,
                    0.006, 0.008, 0.010, 0.012, 0.015, 0.020])
family = []
# Seed: planar Lyapunov (z0=0) — known to converge.
# Use smaller amplitude to stay in linear regime for small Az continuation.
x0_seed, vy0_seed = x_L2 + 0.002, -0.0124
for Az in Az_list:
    try:
        x0_new, vy0_new, Th, err = halo_dc(x0_seed, Az, vy0_seed, damp=0.8)
        T_full = 2 * Th
        family.append({
            'Az': Az, 'x0': x0_new, 'vy0': vy0_new, 'T': T_full,
            'ax': x0_new - x_L2, 'err': err,
        })
        x0_seed, vy0_seed = x0_new, vy0_new
        print(f"  Az={Az:.4e}  x0-xL2={x0_new-x_L2:+.4e}  vy0={vy0_new:+.4e}  T={T_full:.4f}  err={err:.1e}")
    except RuntimeError as e:
        print(f"  Az={Az:.4e}  FAILED: {e}")

# Compute monodromy for each family member
for fam in family:
    Y0 = np.zeros(42)
    Y0[:6] = [fam['x0'], 0, fam['Az'], 0, fam['vy0'], 0]
    Y0[6:] = np.eye(6).flatten()
    sol = solve_ivp(var_rhs, (0, fam['T']), Y0, rtol=1e-12, atol=1e-14)
    M = sol.y[6:, -1].reshape(6, 6)
    eigs = np.linalg.eigvals(M)
    lam_u = float(np.max(np.abs(eigs)).real)
    lam_s = float(np.min(np.abs(eigs)).real)
    nu = 0.5 * (lam_u + 1.0/lam_u) if lam_u > 1e-10 else np.inf
    fam['lam_u'] = lam_u
    fam['stab_idx'] = nu
    # Jacobi constant of the orbit
    fam['C'] = jacobi([fam['x0'], 0, fam['Az'], 0, fam['vy0'], 0])

# ------------------------------------------------------------
# CHOOSE REPRESENTATIVE HALO
# ------------------------------------------------------------
rep = family[7]  # Az = 0.006
print(f"\nRepresentative halo: Az={rep['Az']}, T={rep['T']:.4f}, lam_u={rep['lam_u']:.2f}")

Y0 = np.zeros(42)
Y0[:6] = [rep['x0'], 0, rep['Az'], 0, rep['vy0'], 0]
Y0[6:] = np.eye(6).flatten()
t_grid = np.linspace(0, rep['T'], 2500)
sol = solve_ivp(var_rhs, (0, rep['T']), Y0, t_eval=t_grid,
                rtol=1e-13, atol=1e-15)
U_ref = sol.y[:6, :]
Phi_t = sol.y[6:, :].reshape(6, 6, -1)
M_rep = Phi_t[:, :, -1]

eigM, VR = np.linalg.eig(M_rep)
idx_u = int(np.argmax(np.abs(eigM)))
idx_s = int(np.argmin(np.abs(eigM)))
lam_u_rep = eigM[idx_u].real
lam_s_rep = eigM[idx_s].real
e_u0 = np.real_if_close(VR[:, idx_u]).real
e_u0 /= np.linalg.norm(e_u0)
e_s0 = np.real_if_close(VR[:, idx_s]).real
e_s0 /= np.linalg.norm(e_s0)

eigMT, VL = np.linalg.eig(M_rep.T)
idx_uL = int(np.argmin(np.abs(eigMT - lam_u_rep)))
w_u = np.real_if_close(VL[:, idx_uL]).real
w_u /= (w_u @ e_u0)

# Propagated eigenvectors along orbit
eu_t = np.einsum('ijk,j->ik', Phi_t, e_u0)
es_t = np.einsum('ijk,j->ik', Phi_t, e_s0)
L_t = np.empty((6, t_grid.size))
for i in range(t_grid.size):
    L_t[:, i] = np.linalg.solve(Phi_t[:, :, i].T, w_u)
norm_Lv = np.linalg.norm(L_t[3:, :], axis=0)

# ------------------------------------------------------------
# INVARIANT MANIFOLDS (tubes): global stable / unstable
# ------------------------------------------------------------
print("\n" + "="*60)
print("INVARIANT MANIFOLDS")
print("="*60)

def propagate_manifold(U_seed, t_span, rtol=1e-11, atol=1e-13):
    sol = solve_ivp(cr3bp, t_span, U_seed, rtol=rtol, atol=atol,
                    t_eval=np.linspace(*t_span, 400))
    return sol.y

N_branch = 24
branch_idx = np.linspace(0, t_grid.size - 2, N_branch, dtype=int)
eps_man = 1e-6

tubes_unstable_plus, tubes_unstable_minus = [], []
tubes_stable_plus,   tubes_stable_minus   = [], []

for i in branch_idx:
    U_base = U_ref[:, i]
    # Unstable direction at this point
    eu_local = eu_t[:, i] / np.linalg.norm(eu_t[:, i])
    es_local = es_t[:, i] / np.linalg.norm(es_t[:, i])

    # Unstable (forward in time)
    tubes_unstable_plus.append(
        propagate_manifold(U_base + eps_man * eu_local,
                           (0, 1.2 * rep['T'])))
    tubes_unstable_minus.append(
        propagate_manifold(U_base - eps_man * eu_local,
                           (0, 1.2 * rep['T'])))
    # Stable (backward in time)
    tubes_stable_plus.append(
        propagate_manifold(U_base + eps_man * es_local,
                           (0, -1.2 * rep['T'])))
    tubes_stable_minus.append(
        propagate_manifold(U_base - eps_man * es_local,
                           (0, -1.2 * rep['T'])))
print(f"  computed {4 * N_branch} manifold trajectories")

# ------------------------------------------------------------
# MULTI-SHOOTING STATION-KEEPING
# Apply K corrections per period at equispaced nodes on reference orbit.
# ------------------------------------------------------------
print("\n" + "="*60)
print("MULTI-SHOOTING STATION-KEEPING")
print("="*60)

def multi_shoot(K, n_periods, alpha0, seed=11, dv_cap=1e-3):
    """K corrections per period at equispaced nodes, with dv capping for stability."""
    rng = np.random.default_rng(seed)
    t_nodes = np.linspace(0, rep['T'], K + 1)[:-1]
    idx_nodes = [int(np.argmin(np.abs(t_grid - tn))) for tn in t_nodes]
    L_nodes = [L_t[:, i] for i in idx_nodes]
    U_nodes = [U_ref[:, i] for i in idx_nodes]

    U_sc = U_ref[:, 0] + alpha0 * e_u0
    err_hist = [(0.0, np.linalg.norm(U_sc - U_ref[:, 0]))]
    dv_total = 0.0
    dv_list = []

    for p in range(n_periods):
        for k, tn in enumerate(t_nodes):
            t_prev = p * rep['T'] + (0 if k == 0 else t_nodes[k-1])
            t_now = p * rep['T'] + tn
            if t_now > t_prev + 1e-9:
                try:
                    s = solve_ivp(cr3bp, (t_prev, t_now), U_sc,
                                  t_eval=np.linspace(t_prev, t_now, 40),
                                  rtol=1e-11, atol=1e-13, max_step=0.05)
                    for j in range(s.y.shape[1]):
                        i_ref = int(np.argmin(
                            np.abs(t_grid - (s.t[j] % rep['T']))))
                        err_hist.append(
                            (s.t[j], np.linalg.norm(s.y[:, j] - U_ref[:, i_ref])))
                    U_sc = s.y[:, -1]
                except Exception:
                    break
            alpha = L_nodes[k] @ (U_sc - U_nodes[k])
            Lv = L_nodes[k][3:]
            lvn = Lv @ Lv
            if lvn < 1e-14:
                continue
            dv = -alpha * Lv / lvn
            # Cap: if correction is absurdly large, skip this node
            dvn = np.linalg.norm(dv)
            if dvn > dv_cap:
                continue
            U_sc[3:] += dv
            dv_total += dvn
            dv_list.append(dvn)
            err_hist.append((t_now + 1e-8, np.linalg.norm(U_sc - U_nodes[k])))
        U_sc[3:] += 1e-8 * rng.normal(size=3)

    return np.array(err_hist), dv_total, dv_list

alpha_pert = 1e-5
n_per = 4
results = {}
for K in [1, 2, 4, 8, 16]:
    err_hist, dv_total, dv_list = multi_shoot(K, n_per, alpha_pert)
    results[K] = {'err': err_hist, 'dv_total': dv_total, 'dv_list': dv_list}
    print(f"  K={K:2d}  total |Δv|={dv_total:.3e}  max err={err_hist[:,1].max():.3e}")

# ============================================================
# FIGURES
# ============================================================
print("\nGenerating figures...")

# ---- FIG 21: 3D halo (representative) ----
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot(U_ref[0], U_ref[1], U_ref[2], 'b-', lw=2.5, label='halo orbit')
ax.scatter([x_L2], [0], [0], s=150, marker='X', c='k', label='$L_2$')
ax.scatter([1-mu], [0], [0], s=100, c='sandybrown', label='Jupiter')
# Start point
ax.scatter([U_ref[0,0]], [U_ref[1,0]], [U_ref[2,0]], s=100, c='green',
           marker='*', label='y=0 crossing')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title(f"3D Halo orbit at $L_2$  (Az={rep['Az']:.3f}, T={rep['T']:.3f})")
ax.legend()
savefig('21_halo_3d.png')

# ---- FIG 22: halo projections ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (a, b, la, lb) in zip(axes, [(0,1,'x','y'), (0,2,'x','z'), (1,2,'y','z')]):
    ax.plot(U_ref[a], U_ref[b], 'b-', lw=2)
    ax.scatter([x_L2 if a==0 else 0], [0], s=80, marker='X', c='k', label='$L_2$')
    ax.set_xlabel(la); ax.set_ylabel(lb); ax.grid(alpha=0.3); ax.axis('equal')
    ax.set_title(f'{la}-{lb} projection')
plt.tight_layout()
savefig('22_halo_projections.png')

# ---- FIG 23: Family of halos in 3D ----
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection='3d')
colors = cm.viridis(np.linspace(0, 1, len(family)))
for fam, c in zip(family, colors):
    Y0f = np.zeros(6)
    Y0f[:] = [fam['x0'], 0, fam['Az'], 0, fam['vy0'], 0]
    t_eval = np.linspace(0, fam['T'], 500)
    s = solve_ivp(cr3bp, (0, fam['T']), Y0f, t_eval=t_eval,
                  rtol=1e-11, atol=1e-13)
    ax.plot(s.y[0], s.y[1], s.y[2], color=c, lw=1.3,
            label=f"Az={fam['Az']:.3f}" if fam['Az'] in [0.0005, 0.005, 0.020] else None)
ax.scatter([x_L2], [0], [0], s=120, marker='X', c='k')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title(f'Halo family at Sun-Jupiter $L_2$  ({len(family)} orbits)')
ax.legend(loc='upper left')
savefig('23_halo_family_3d.png')

# ---- FIG 24: Family continuation diagrams ----
Az_arr = np.array([f['Az'] for f in family])
ax_arr = np.array([f['ax'] for f in family])
T_arr = np.array([f['T'] for f in family])
lam_arr = np.array([f['lam_u'] for f in family])
nu_arr = np.array([f['stab_idx'] for f in family])
C_arr = np.array([f['C'] for f in family])

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes[0,0].plot(Az_arr, ax_arr, 'o-', lw=2, markersize=6)
axes[0,0].set_xlabel('$A_z$ (z amplitude)')
axes[0,0].set_ylabel('$A_x = x_0 - x_{L_2}$')
axes[0,0].set_title('Planar amplitude vs vertical amplitude')
axes[0,0].grid(alpha=0.3)

axes[0,1].plot(Az_arr, T_arr, 'o-', lw=2, color='green', markersize=6)
axes[0,1].set_xlabel('$A_z$'); axes[0,1].set_ylabel('Period T')
axes[0,1].set_title('Period along the family')
axes[0,1].grid(alpha=0.3)

axes[1,0].semilogy(Az_arr, lam_arr, 'o-', lw=2, color='red', markersize=6)
axes[1,0].set_xlabel('$A_z$'); axes[1,0].set_ylabel('$|\\lambda_u|$ (log)')
axes[1,0].set_title('Dominant unstable eigenvalue')
axes[1,0].grid(alpha=0.3, which='both')

axes[1,1].plot(Az_arr, C_arr, 'o-', lw=2, color='purple', markersize=6)
axes[1,1].set_xlabel('$A_z$'); axes[1,1].set_ylabel('Jacobi constant C')
axes[1,1].set_title('Energy along the family')
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
savefig('24_family_diagrams.png')

# ---- FIG 25: Stability index vs Az ----
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(Az_arr, nu_arr, 'o-', lw=2, markersize=7)
ax.axhline(1, color='red', ls='--', label='stability boundary ($\\nu=1$)')
ax.set_xlabel('$A_z$'); ax.set_ylabel('stability index $\\nu = (\\lambda_u + 1/\\lambda_u)/2$')
ax.set_title(f'Stability index along halo family at Sun-Jupiter $L_2$ ($\\mu$={mu:.3e})')
ax.grid(alpha=0.4, which='both'); ax.legend()
savefig('25_stability_index.png')

# ---- FIG 26: Unstable manifold tube (3D) ----
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot(U_ref[0], U_ref[1], U_ref[2], 'b-', lw=3, label='halo orbit', alpha=0.9)
for Y in tubes_unstable_plus:
    ax.plot(Y[0], Y[1], Y[2], 'r-', lw=0.7, alpha=0.5)
for Y in tubes_unstable_minus:
    ax.plot(Y[0], Y[1], Y[2], 'orange', lw=0.7, alpha=0.5)
ax.scatter([x_L2], [0], [0], s=100, marker='X', c='k', label='$L_2$')
ax.scatter([1-mu], [0], [0], s=120, c='sandybrown', label='Jupiter')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('Unstable manifold tube  $W^u$  (forward-time globalization)')
ax.legend()
savefig('26_manifold_unstable.png')

# ---- FIG 27: Stable manifold tube (3D) ----
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.plot(U_ref[0], U_ref[1], U_ref[2], 'b-', lw=3, label='halo orbit', alpha=0.9)
for Y in tubes_stable_plus:
    ax.plot(Y[0], Y[1], Y[2], 'g-', lw=0.7, alpha=0.5)
for Y in tubes_stable_minus:
    ax.plot(Y[0], Y[1], Y[2], 'lime', lw=0.7, alpha=0.5)
ax.scatter([x_L2], [0], [0], s=100, marker='X', c='k', label='$L_2$')
ax.scatter([1-mu], [0], [0], s=120, c='sandybrown', label='Jupiter')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('Stable manifold tube  $W^s$  (backward-time globalization)')
ax.legend()
savefig('27_manifold_stable.png')

# ---- FIG 28: Combined manifolds (tubes) ----
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
for Y in tubes_unstable_plus:
    ax.plot(Y[0], Y[1], Y[2], 'r-', lw=0.5, alpha=0.4)
for Y in tubes_unstable_minus:
    ax.plot(Y[0], Y[1], Y[2], 'red', lw=0.5, alpha=0.4)
for Y in tubes_stable_plus:
    ax.plot(Y[0], Y[1], Y[2], 'g-', lw=0.5, alpha=0.4)
for Y in tubes_stable_minus:
    ax.plot(Y[0], Y[1], Y[2], 'green', lw=0.5, alpha=0.4)
ax.plot(U_ref[0], U_ref[1], U_ref[2], 'b-', lw=3, alpha=0.9)
ax.scatter([x_L2], [0], [0], s=100, marker='X', c='k')
ax.scatter([1-mu], [0], [0], s=120, c='sandybrown')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('Invariant tubes:  $W^u$ (red) and $W^s$ (green) around halo (blue)')
savefig('28_manifolds_combined.png')

# ---- FIG 29: Manifold projection (x-y plane) ----
fig, ax = plt.subplots(figsize=(11, 9))
for Y in tubes_unstable_plus + tubes_unstable_minus:
    ax.plot(Y[0], Y[1], 'r-', lw=0.4, alpha=0.4)
for Y in tubes_stable_plus + tubes_stable_minus:
    ax.plot(Y[0], Y[1], 'g-', lw=0.4, alpha=0.4)
ax.plot(U_ref[0], U_ref[1], 'b-', lw=2.5)
ax.scatter([x_L2], [0], s=100, marker='X', c='k', label='$L_2$')
ax.scatter([1-mu], [0], s=120, c='sandybrown', label='Jupiter')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Manifolds projected onto x-y plane')
ax.legend(); ax.grid(alpha=0.3); ax.set_aspect('equal')
savefig('29_manifolds_xy.png')

# ---- FIG 30: Manifold Poincare section (y=0 plane) ----
fig, ax = plt.subplots(figsize=(10, 8))
def collect_crossings(tubes, color):
    pts = []
    for Y in tubes:
        y = Y[1]
        for i in range(len(y) - 1):
            if y[i] * y[i+1] < 0:
                # linear interpolation
                frac = -y[i] / (y[i+1] - y[i])
                x_c = Y[0, i] + frac * (Y[0, i+1] - Y[0, i])
                z_c = Y[2, i] + frac * (Y[2, i+1] - Y[2, i])
                pts.append((x_c, z_c))
    if pts:
        pts = np.array(pts)
        ax.scatter(pts[:,0], pts[:,1], c=color, s=15, alpha=0.7)
collect_crossings(tubes_unstable_plus + tubes_unstable_minus, 'red')
collect_crossings(tubes_stable_plus + tubes_stable_minus, 'green')
ax.scatter([x_L2], [0], s=120, marker='X', c='k', label='$L_2$')
ax.set_xlabel('x'); ax.set_ylabel('z')
ax.set_title('Poincaré section $\\{y=0\\}$ through manifolds\nred: $W^u$, green: $W^s$')
ax.legend(); ax.grid(alpha=0.3)
savefig('30_poincare_section.png')

# ---- FIG 31: Multi-shooting — error trajectories ----
fig, ax = plt.subplots(figsize=(12, 6))
for K, res in results.items():
    eh = res['err']
    ax.semilogy(eh[:,0]/rep['T'], eh[:,1], lw=1.5, label=f'K={K}')
for p in range(1, n_per+1):
    ax.axvline(p, color='gray', ls=':', alpha=0.4)
ax.set_xlabel('t / T'); ax.set_ylabel('$\\|\\delta U\\|$ (log)')
ax.set_title(f'Multi-shooting station-keeping: K corrections per period')
ax.grid(alpha=0.3, which='both'); ax.legend()
savefig('31_multishoot_error.png')

# ---- FIG 32: Multi-shooting — ΔV budget ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
Ks = list(results.keys())
dv_totals = [results[K]['dv_total'] for K in Ks]
max_errs = [results[K]['err'][:,1].max() for K in Ks]

axes[0].loglog(Ks, dv_totals, 'o-', lw=2, markersize=10)
axes[0].set_xlabel('K (corrections per period)')
axes[0].set_ylabel('total $\\Sigma|\\Delta v|$ over %d periods' % n_per)
axes[0].set_title('Fuel budget vs correction frequency')
axes[0].grid(alpha=0.4, which='both')

axes[1].loglog(Ks, max_errs, 'o-', lw=2, markersize=10, color='red')
axes[1].set_xlabel('K (corrections per period)')
axes[1].set_ylabel('peak $\\|\\delta U\\|$')
axes[1].set_title('Peak deviation vs correction frequency')
axes[1].grid(alpha=0.4, which='both')

plt.tight_layout()
savefig('32_multishoot_tradeoff.png')

# ---- FIG 33: Multi-shooting — individual ΔV per correction ----
fig, ax = plt.subplots(figsize=(12, 5))
colors_K = cm.plasma(np.linspace(0.1, 0.9, len(Ks)))
for K, c in zip(Ks, colors_K):
    dv_list = results[K]['dv_list']
    t_corrections = np.arange(len(dv_list)) * (n_per * rep['T']) / len(dv_list)
    ax.semilogy(t_corrections / rep['T'], dv_list, 'o-',
                color=c, lw=1.5, markersize=4, label=f'K={K}')
ax.set_xlabel('t / T'); ax.set_ylabel('$|\\Delta v|$ per correction (log)')
ax.set_title('Individual correction magnitudes')
ax.grid(alpha=0.3, which='both'); ax.legend()
savefig('33_multishoot_dv_series.png')

# ---- FIG 34: Family overview dashboard ----
fig = plt.figure(figsize=(15, 10))

ax1 = fig.add_subplot(2, 3, 1, projection='3d')
for fam, c in zip(family, colors):
    Y0f = [fam['x0'], 0, fam['Az'], 0, fam['vy0'], 0]
    s = solve_ivp(cr3bp, (0, fam['T']), Y0f,
                  t_eval=np.linspace(0, fam['T'], 300),
                  rtol=1e-11, atol=1e-13)
    ax1.plot(s.y[0], s.y[1], s.y[2], color=c, lw=1)
ax1.scatter([x_L2], [0], [0], marker='X', c='k', s=80)
ax1.set_title('Halo family 3D')

ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(Az_arr, ax_arr, 'o-')
ax2.set_xlabel('Az'); ax2.set_ylabel('Ax'); ax2.set_title('Amplitudes')
ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(Az_arr, T_arr, 'o-', color='green')
ax3.set_xlabel('Az'); ax3.set_ylabel('T'); ax3.set_title('Period')
ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(2, 3, 4)
ax4.semilogy(Az_arr, lam_arr, 'o-', color='red')
ax4.set_xlabel('Az'); ax4.set_ylabel(r'$|\lambda_u|$')
ax4.set_title('Instability growth factor')
ax4.grid(alpha=0.3, which='both')

ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(Az_arr, C_arr, 'o-', color='purple')
ax5.set_xlabel('Az'); ax5.set_ylabel('C'); ax5.set_title('Jacobi constant')
ax5.grid(alpha=0.3)

ax6 = fig.add_subplot(2, 3, 6)
for K, c in zip(Ks, colors_K):
    eh = results[K]['err']
    ax6.semilogy(eh[:,0]/rep['T'], eh[:,1], lw=1, color=c, label=f'K={K}')
ax6.set_xlabel('t/T'); ax6.set_ylabel('error')
ax6.set_title('Multi-shoot error')
ax6.grid(alpha=0.3, which='both'); ax6.legend(fontsize=8)

fig.suptitle('Halo family + station-keeping — overview', fontsize=14)
plt.tight_layout()
savefig('34_family_dashboard.png')

# ---- FIG 35: Manifolds with orbit family together (3D) ----
fig = plt.figure(figsize=(13, 10))
ax = fig.add_subplot(111, projection='3d')
for fam, c in zip(family[::2], colors[::2]):
    Y0f = [fam['x0'], 0, fam['Az'], 0, fam['vy0'], 0]
    s = solve_ivp(cr3bp, (0, fam['T']), Y0f,
                  t_eval=np.linspace(0, fam['T'], 300),
                  rtol=1e-11, atol=1e-13)
    ax.plot(s.y[0], s.y[1], s.y[2], color=c, lw=0.8, alpha=0.7)
for Y in tubes_unstable_plus[::3]:
    ax.plot(Y[0], Y[1], Y[2], 'r-', lw=0.4, alpha=0.3)
for Y in tubes_stable_plus[::3]:
    ax.plot(Y[0], Y[1], Y[2], 'g-', lw=0.4, alpha=0.3)
ax.scatter([x_L2], [0], [0], s=100, marker='X', c='k')
ax.scatter([1-mu], [0], [0], s=120, c='sandybrown')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('Halo family + invariant manifolds')
savefig('35_family_manifolds.png')

# Save family data to file for the report
import json
with open('/Users/aleksandrvolkov/2/family_data.json', 'w') as f:
    out = {
        'mu': mu, 'x_L2': x_L2,
        'family': [{k: float(v) for k,v in fam.items()} for fam in family],
        'representative': {k: float(v) for k,v in rep.items()},
        'lam_u_rep': float(lam_u_rep),
        'lam_s_rep': float(lam_s_rep),
        'multishoot': {str(K): {'dv_total': float(results[K]['dv_total']),
                                 'max_err': float(results[K]['err'][:,1].max())}
                       for K in Ks},
    }
    json.dump(out, f, indent=2)
print(f"\nFamily data written to family_data.json")
print(f"Total figures: 35")
