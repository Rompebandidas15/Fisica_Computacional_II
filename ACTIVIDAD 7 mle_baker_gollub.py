import numpy as np
from numpy import sin, cos, pi, log
import matplotlib.pyplot as plt

# Parámetros del sistema (Baker & Gollub)
q = 2.0
om = 2.0 / 3.0
T = 2.0 * pi / om

# Condiciones iniciales
x0 = 1.25
v0 = 0.0
t = 0.0

# Rango del parámetro de forzamiento gamma
gamma_min = 0.9
gamma_max = 1.8
dgamma = 0.001  # Cambiar a 0.0001 para mayor resolución
gamma_values = np.arange(gamma_min, gamma_max + dgamma, dgamma)
n_orbits = len(gamma_values)

# Parámetros del método numérico
nTrans = 300
nLyap = 600
steps_per_T = 300
dt = T / steps_per_T

# Estado inicial y vector tangente unitario
x = np.full(n_orbits, x0)
v = np.full(n_orbits, v0)
xi = np.full(n_orbits, 1.0 / np.sqrt(2.0))
eta = np.full(n_orbits, 1.0 / np.sqrt(2.0))

y = np.concatenate([x, v, xi, eta])

# Dinámica + Ecuaciones variacionales
def dyn(t, y):
    x_vec = y[:n_orbits]
    v_vec = y[n_orbits:2*n_orbits]
    xi_vec = y[2*n_orbits:3*n_orbits]
    eta_vec = y[3*n_orbits:]
    
    dx = v_vec
    dv = -(1.0/q)*v_vec - sin(x_vec) + gamma_values * cos(om * t)
    dxi = eta_vec
    deta = -cos(x_vec)*xi_vec - (1.0/q)*eta_vec
    
    return np.concatenate([dx, dv, dxi, deta])

# Método de Runge-Kutta de 4to orden
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2.0, y + k1/2.0)
    k3 = h * f(t + h/2.0, y + k2/2.0)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

# Integración: Régimen transitorio
for period in range(nTrans):
    for step in range(steps_per_T):
        y = rk4(dyn, t, y, dt)
        t += dt
    xi_vec = y[2*n_orbits:3*n_orbits]
    eta_vec = y[3*n_orbits:]
    tangent_norm = np.sqrt(xi_vec**2 + eta_vec**2)
    y[2*n_orbits:3*n_orbits] = xi_vec / tangent_norm
    y[3*n_orbits:] = eta_vec / tangent_norm

# Integración y acumulación del MLE
sum_log = np.zeros(n_orbits)
for period in range(nLyap):
    for step in range(steps_per_T):
        y = rk4(dyn, t, y, dt)
        t += dt
    xi_vec = y[2*n_orbits:3*n_orbits]
    eta_vec = y[3*n_orbits:]
    tangent_norm = np.sqrt(xi_vec**2 + eta_vec**2)
    sum_log += log(tangent_norm)
    y[2*n_orbits:3*n_orbits] = xi_vec / tangent_norm
    y[3*n_orbits:] = eta_vec / tangent_norm

# Exponente Máximo de Lyapunov por unidad de tiempo
lambda_max = sum_log / (nLyap * T)
imax = np.argmax(lambda_max)
gamma_at_max = gamma_values[imax]
lambda_at_max = lambda_max[imax]

print(f"Punto máximo = ({gamma_at_max:.6f}, {lambda_at_max:.6f})")

# Generación del gráfico
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ax.plot(gamma_values, lambda_max, linewidth=0.5, color='blue')
ax.axhline(y=0, linewidth=0.5, color='black')
ax.set_xlabel(r"$\gamma$", fontsize=16)
ax.set_ylabel(r"$\lambda_{\max}$", fontsize=16)
ax.tick_params(axis='both', labelsize=12)
ax.text(0.03, 0.93, rf"$\gamma = {gamma_at_max:.2f}$", transform=ax.transAxes, fontsize=14)
ax.set_xlim(gamma_values[0], gamma_values[-1])
ax.set_box_aspect(0.65)

plt.tight_layout()
plt.savefig('Lya.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()