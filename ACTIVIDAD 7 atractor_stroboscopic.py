import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

# Parámetros del sistema
q = 2.0
om = 2.0 / 3.0
T = 2.0 * pi / om
gamma = 1.1799

# Condiciones iniciales
x0 = 1.25
v0 = 0.0
t = 0.0

# Parámetros numéricos
nTrans = 300         # Períodos transitorios a descartar
nKeep = 4000          # Períodos a graficar en el atractor
steps_per_T = 300
dt = T / steps_per_T

def dyn(t, y):
    x, v = y[0], y[1]
    dx = v
    dv = -(1.0/q)*v - sin(x) + gamma * cos(om * t)
    return np.array([dx, dv])

def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2.0, y + k1/2.0)
    k3 = h * f(t + h/2.0, y + k2/2.0)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

y = np.array([x0, v0])

# Régimen transitorio
for period in range(nTrans):
    for step in range(steps_per_T):
        y = rk4(dyn, t, y, dt)
        t += dt

# Construcción del mapa estroboscópico
x_strobe = []
v_strobe = []

for period in range(nKeep):
    for step in range(steps_per_T):
        y = rk4(dyn, t, y, dt)
        t += dt
    # Re-mapeo del ángulo theta a [-pi, pi]
    theta_wrapped = (y[0] + pi) % (2.0 * pi) - pi
    x_strobe.append(theta_wrapped)
    v_strobe.append(y[1])

# Generación de la figura del Atractor
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ax.scatter(x_strobe, v_strobe, s=0.15, color='#3B0B58', alpha=0.8, rasterized=True)
ax.set_xlabel(r"$\theta$", fontsize=16)
ax.set_ylabel(r"$\dot{\theta}$", fontsize=16)
ax.text(0.03, 0.92, r"$q = 2$", transform=ax.transAxes, fontsize=14)
ax.text(0.03, 0.05, r"$\omega_D = 2/3$", transform=ax.transAxes, fontsize=14)
ax.text(0.80, 0.92, r"$\gamma = 1.1799$", transform=ax.transAxes, fontsize=14)
ax.set_xlim(-pi, pi)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig('Atractor.pdf', format='pdf', bbox_inches='tight', dpi=800)
plt.show()


#Andres Felipe Rojas Tafur
#070400372024
#Para hacer el codigo como no pude asistir a clases:
# Estudié el PDF de la Clase 8, 
# tomé la estructura de integración RK4 que dejó en las diapositivas y 
# la adapté para registrar el estado del sistema cada período T
# tambien me apoyé en IA como tutor para ajustar la condición del mapa estroboscópico