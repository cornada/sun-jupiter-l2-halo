"""Scan halo DC seeds to find which converge to true 1-period halos."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

mu = 9.53e-4

def _radii(U):
    x,y,z = U[0],U[1],U[2]
    return np.sqrt((x+mu)**2+y**2+z**2), np.sqrt((x-(1-mu))**2+y**2+z**2)
def cr3bp(t, U):
    x,y,z,vx,vy,vz = U
    R1,R2 = _radii(U)
    ax = 2*vy + x - (1-mu)*(x+mu)/R1**3 - mu*(x-(1-mu))/R2**3
    ay = -2*vx + y - (1-mu)*y/R1**3 - mu*y/R2**3
    az = -(1-mu)*z/R1**3 - mu*z/R2**3
    return [vx,vy,vz,ax,ay,az]
def A_matrix(U):
    x,y,z = U[0],U[1],U[2]
    R1sq=(x+mu)**2+y**2+z**2; R2sq=(x-(1-mu))**2+y**2+z**2
    R1_3,R1_5 = R1sq**1.5, R1sq**2.5
    R2_3,R2_5 = R2sq**1.5, R2sq**2.5
    mu1 = 1-mu
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
    U = Y[:6]; Phi = Y[6:].reshape(6,6)
    return np.concatenate([cr3bp(t,U),(A_matrix(U)@Phi).flatten()])

def find_L2():
    def f(x):
        R1=abs(x+mu); R2=abs(x-(1-mu))
        return x-(1-mu)*(x+mu)/R1**3-mu*(x-(1-mu))/R2**3
    return root_scalar(f, bracket=[1-mu+1e-4,1.5], method='brentq').root
x_L2 = find_L2()

def half_period_3d(x0, z0, vy0, t_max=3.5):
    Y0 = np.zeros(42); Y0[:6] = [x0,0,z0,0,vy0,0]; Y0[6:] = np.eye(6).flatten()
    direction = int(np.sign(vy0)) if vy0 != 0 else 1
    def hit_y(t,Y): return Y[1]
    hit_y.terminal = True; hit_y.direction = -direction
    sol = solve_ivp(var_rhs,(1e-6,t_max),Y0,events=hit_y,
                    rtol=1e-11,atol=1e-13,max_step=0.03)
    if not sol.t_events[0].size: return None, None
    return sol.t_events[0][0], sol.y_events[0][0]

def halo_dc(x0, z0, vy0, tol=1e-9, maxiter=80, damp=0.3, step_max=1e-3):
    err = np.inf
    for it in range(maxiter):
        Tf, Yf = half_period_3d(x0, z0, vy0)
        if Tf is None:
            return None
        U = Yf[:6]; Phi = Yf[6:].reshape(6,6)
        vx_f, vz_f = U[3], U[5]
        err = max(abs(vx_f), abs(vz_f))
        if err < tol: return (x0, vy0, Tf, err)
        dU = cr3bp(Tf, U)
        ydot = dU[1]
        if abs(ydot) < 1e-10: return None
        ax_f, az_f = dU[3], dU[5]
        M = np.array([
            [Phi[3,0]-ax_f*Phi[1,0]/ydot, Phi[3,4]-ax_f*Phi[1,4]/ydot],
            [Phi[5,0]-az_f*Phi[1,0]/ydot, Phi[5,4]-az_f*Phi[1,4]/ydot]])
        rhs = np.array([-vx_f, -vz_f])
        try:
            delta = np.linalg.solve(M, rhs)
        except:
            return None
        nrm = np.linalg.norm(delta)
        if nrm > step_max: delta *= step_max/nrm
        x0 += damp*delta[0]; vy0 += damp*delta[1]
    return None

print(f"x_L2 = {x_L2:.6f}")
print("Scanning seeds for halo orbits (T<4)...")
print(f"{'x0-xL2':>10s} {'Az':>8s} {'vy0_g':>8s} {'x0_conv':>11s} {'vy0_conv':>11s} {'T':>6s}")
found = []
for x0_off in [0.02, 0.03, 0.04, 0.05, 0.07]:
    for Az in [0.01, 0.02, 0.03, 0.05, 0.08]:
        for vy0_g in [-0.05, -0.08, -0.12, -0.15, -0.20, -0.25]:
            r = halo_dc(x_L2+x0_off, Az, vy0_g)
            if r is not None:
                x0n, vy0n, Th, err = r
                T = 2*Th
                if 2.0 < T < 4.0:
                    print(f"{x0_off:+.4f}   {Az:.4f}   {vy0_g:+.3f}      "
                          f"{x0n-x_L2:+.4e} {vy0n:+.4e}  {T:.3f}")
                    found.append((x0_off, Az, vy0_g, x0n, vy0n, T))
print(f"\nFound {len(found)} candidate halo orbits.")
