"""Export orbit + manifold + family data for the interactive HTML demo."""
import os
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

mu = 9.53e-4

def _radii(U):
    x,y,z = U[0],U[1],U[2]
    return (np.sqrt((x+mu)**2+y**2+z**2),
            np.sqrt((x-(1-mu))**2+y**2+z**2))

def cr3bp(t, U):
    x,y,z,vx,vy,vz = U
    R1,R2 = _radii(U)
    return [vx,vy,vz,
        2*vy+x-(1-mu)*(x+mu)/R1**3-mu*(x-(1-mu))/R2**3,
        -2*vx+y-(1-mu)*y/R1**3-mu*y/R2**3,
        -(1-mu)*z/R1**3-mu*z/R2**3]

def jacobi(U):
    x,y,z,vx,vy,vz = U
    R1,R2 = _radii(U)
    O = 0.5*(x*x+y*y) + (1-mu)/R1 + mu/R2 + 0.5*mu*(1-mu)
    return 2*O - (vx*vx+vy*vy+vz*vz)

def A_matrix(U):
    x,y,z=U[0],U[1],U[2]
    R1sq=(x+mu)**2+y**2+z**2; R2sq=(x-(1-mu))**2+y**2+z**2
    R1_3,R1_5=R1sq**1.5,R1sq**2.5
    R2_3,R2_5=R2sq**1.5,R2sq**2.5
    mu1=1-mu
    Oxx=1-mu1*(1/R1_3-3*(x+mu)**2/R1_5)-mu*(1/R2_3-3*(x-(1-mu))**2/R2_5)
    Oyy=1-mu1*(1/R1_3-3*y**2/R1_5)-mu*(1/R2_3-3*y**2/R2_5)
    Ozz=-mu1*(1/R1_3-3*z**2/R1_5)-mu*(1/R2_3-3*z**2/R2_5)
    Oxy=3*y*(mu1*(x+mu)/R1_5+mu*(x-(1-mu))/R2_5)
    Oxz=3*z*(mu1*(x+mu)/R1_5+mu*(x-(1-mu))/R2_5)
    Oyz=3*y*z*(mu1/R1_5+mu/R2_5)
    return np.array([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],
                     [Oxx,Oxy,Oxz,0,2,0],[Oxy,Oyy,Oyz,-2,0,0],
                     [Oxz,Oyz,Ozz,0,0,0]],float)

def var_rhs(t, Y):
    U=Y[:6]; Phi=Y[6:].reshape(6,6)
    return np.concatenate([cr3bp(t,U),(A_matrix(U)@Phi).flatten()])

def find_L2():
    def f(x):
        R1=abs(x+mu); R2=abs(x-(1-mu))
        return x-(1-mu)*(x+mu)/R1**3-mu*(x-(1-mu))/R2**3
    return root_scalar(f,bracket=[1-mu+1e-4,1.5],method='brentq').root

x_L2 = find_L2()

def find_L1():
    def f(x):
        R1=abs(x+mu); R2=abs(x-(1-mu))
        return x-(1-mu)*(x+mu)/R1**3-mu*(x-(1-mu))/R2**3
    return root_scalar(f,bracket=[1-mu-0.5,1-mu-1e-4],method='brentq').root

x_L1 = find_L1()

# ---- Differential correction ----
def half_period_3d(x0, z0, vy0, t_max=10.0):
    Y0=np.zeros(42); Y0[:6]=[x0,0,z0,0,vy0,0]; Y0[6:]=np.eye(6).flatten()
    direction = int(np.sign(vy0)) if vy0 != 0 else 1
    def hit_y(t,Y): return Y[1]
    hit_y.terminal=True; hit_y.direction=-direction
    sol=solve_ivp(var_rhs,(1e-6,t_max),Y0,events=hit_y,
                  rtol=1e-11,atol=1e-13,max_step=0.05)
    return sol.t_events[0][0], sol.y_events[0][0]

def halo_dc(x0, z0, vy0, tol=1e-10, maxiter=60, damp=0.8, step_max=5e-3):
    for _ in range(maxiter):
        Tf, Yf = half_period_3d(x0, z0, vy0)
        U=Yf[:6]; Phi=Yf[6:].reshape(6,6)
        vx_f,vz_f = U[3], U[5]
        err=max(abs(vx_f),abs(vz_f))
        if err<tol: return x0, vy0, Tf
        dU=cr3bp(Tf,U); ydot=dU[1]; ax_f,az_f=dU[3],dU[5]
        M=np.array([
            [Phi[3,0]-ax_f*Phi[1,0]/ydot, Phi[3,4]-ax_f*Phi[1,4]/ydot],
            [Phi[5,0]-az_f*Phi[1,0]/ydot, Phi[5,4]-az_f*Phi[1,4]/ydot]])
        d=np.linalg.solve(M,[-vx_f,-vz_f])
        nrm=np.linalg.norm(d)
        if nrm>step_max: d*=step_max/nrm
        x0+=damp*d[0]; vy0+=damp*d[1]
    return None

# ---- Build family ----
print("Building halo family...")
Az_list = [0.002, 0.005, 0.008, 0.012, 0.016, 0.020]
family_data = []
x0_s, vy0_s = x_L2+0.002, -0.0124
for Az in Az_list:
    r = halo_dc(x0_s, Az, vy0_s)
    if r is None:
        print(f"  Az={Az}: FAILED")
        continue
    x0_c, vy0_c, Th = r
    T = 2*Th
    # Integrate full period
    Y0=np.zeros(42); Y0[:6]=[x0_c,0,Az,0,vy0_c,0]; Y0[6:]=np.eye(6).flatten()
    t_eval = np.linspace(0,T,300)
    sol = solve_ivp(var_rhs,(0,T),Y0,t_eval=t_eval,rtol=1e-12,atol=1e-14)
    U_t = sol.y[:6,:]
    Phi_t = sol.y[6:,:].reshape(6,6,-1)
    M = Phi_t[:,:,-1]
    eigs, VR = np.linalg.eig(M)
    idx_u = int(np.argmax(np.abs(eigs)))
    lam_u = float(eigs[idx_u].real)
    e_u0 = np.real_if_close(VR[:,idx_u]).real
    e_u0 /= np.linalg.norm(e_u0)

    eigsT, VL = np.linalg.eig(M.T)
    idx_uL = int(np.argmin(np.abs(eigsT - lam_u)))
    w_u = np.real_if_close(VL[:,idx_uL]).real
    w_u /= (w_u @ e_u0)

    # Propagate left eigenvector along orbit
    L_t = np.empty((6, t_eval.size))
    for i in range(t_eval.size):
        L_t[:,i] = np.linalg.solve(Phi_t[:,:,i].T, w_u)
    norm_Lv = np.linalg.norm(L_t[3:,:], axis=0)

    # Propagated unstable direction field
    eu_t = np.einsum('ijk,j->ik', Phi_t, e_u0)

    family_data.append({
        'Az': float(Az),
        'x0': float(x0_c), 'vy0': float(vy0_c), 'T': float(T),
        'lam_u': float(lam_u),
        'C': float(jacobi([x0_c,0,Az,0,vy0_c,0])),
        'orbit': [[float(U_t[k,i]) for k in range(6)] for i in range(t_eval.size)],
        't_grid': [float(t) for t in t_eval],
        'norm_Lv': [float(v) for v in norm_Lv],
        'e_u_field': [[float(eu_t[k,i]) for k in range(6)]
                      for i in range(0,t_eval.size,15)],
        'e_u0': [float(v) for v in e_u0],
        'w_u': [float(v) for v in w_u],
    })
    x0_s, vy0_s = x0_c, vy0_c
    print(f"  Az={Az}: T={T:.3f} lam_u={lam_u:.1f}")

# ---- Pick representative, compute manifolds ----
rep_idx = 3  # Az = 0.012
rep = family_data[rep_idx]
Y0 = np.zeros(42); Y0[:6]=[rep['x0'],0,rep['Az'],0,rep['vy0'],0]; Y0[6:]=np.eye(6).flatten()
sol = solve_ivp(var_rhs,(0,rep['T']),Y0,t_eval=np.array(rep['t_grid']),
                rtol=1e-12,atol=1e-14)
Phi_t = sol.y[6:,:].reshape(6,6,-1)
eigs, VR = np.linalg.eig(Phi_t[:,:,-1])
idx_s = int(np.argmin(np.abs(eigs)))
e_s0 = np.real_if_close(VR[:,idx_s]).real
e_s0 /= np.linalg.norm(e_s0)

# Globalize manifolds (fewer branches for web)
print("Globalizing manifolds...")
N_branch = 12
branch_idx = np.linspace(0, len(rep['t_grid'])-2, N_branch, dtype=int)
eps_m = 1e-6
U_ref = np.array(rep['orbit']).T
eu_t = np.einsum('ijk,j->ik', Phi_t, np.array(rep['e_u0']))
es_t = np.einsum('ijk,j->ik', Phi_t, e_s0)
manifolds_u_p, manifolds_u_m = [], []
manifolds_s_p, manifolds_s_m = [], []
for i in branch_idx:
    U_base = U_ref[:,i]
    eu_loc = eu_t[:,i]/np.linalg.norm(eu_t[:,i])
    es_loc = es_t[:,i]/np.linalg.norm(es_t[:,i])
    for tub_list, seed, tspan in [
        (manifolds_u_p, U_base + eps_m*eu_loc, (0, 1.3*rep['T'])),
        (manifolds_u_m, U_base - eps_m*eu_loc, (0, 1.3*rep['T'])),
        (manifolds_s_p, U_base + eps_m*es_loc, (0, -1.3*rep['T'])),
        (manifolds_s_m, U_base - eps_m*es_loc, (0, -1.3*rep['T'])),
    ]:
        s = solve_ivp(cr3bp, tspan, seed,
                      t_eval=np.linspace(*tspan, 150),
                      rtol=1e-10, atol=1e-12)
        tub_list.append([[float(s.y[k,j]) for k in range(3)]
                          for j in range(s.y.shape[1])])

# ---- Write JSON ----
data = {
    'mu': float(mu),
    'x_L1': float(x_L1),
    'x_L2': float(x_L2),
    'sun': [-float(mu), 0, 0],
    'jupiter': [1-float(mu), 0, 0],
    'family': family_data,
    'representative_idx': rep_idx,
    'manifolds': {
        'unstable_plus':  manifolds_u_p,
        'unstable_minus': manifolds_u_m,
        'stable_plus':    manifolds_s_p,
        'stable_minus':   manifolds_s_m,
    },
}
with open('/Users/aleksandrvolkov/2/web_data.json', 'w') as f:
    json.dump(data, f)
print(f"Wrote web_data.json  ({os.path.getsize('/Users/aleksandrvolkov/2/web_data.json')/1024:.1f} KB)")
