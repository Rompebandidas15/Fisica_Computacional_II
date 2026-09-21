import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Parámetros Del Sistema
# ---------------------------------------------------
m = 1
k = 1
l = 1
F = 1
om = 2/3 * pi
T = 2 * pi / om

# ---------------------------------------------------
# Condiciones Iniciales
# ---------------------------------------------------
x0 = 1.8
v0 = 0

# ---------------------------------------------------
# Parámetros del Método
# ---------------------------------------------------
Trans = 100
Nperiods = 15000
tmax = Nperiods * T
dt = 0.001

# ----------------------------------------------------
# Dinámica: Oscilador de Duffing Forzado
# ----------------------------------------------------
def dyn(t, y):
    x, v = y
    dx = v
    dv = (F * cos(om * t) / m) + ((k / m) * x) - ((1 / m) * x**3)
    return np.array((dx, dv))

# ----------------------------------------------------
# Método RK4
# ----------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# ----------------------------------------------------
# Integración mediante RK4
# ----------------------------------------------------
n = int(tmax / dt)
t = np.linspace(0, n*dt, n+1)
y = np.empty((n+1, 2))
y[0] = [x0, v0]

print("Iniciando integración numérica...")
for i in range(n):
    y[i+1] = rk4(dyn, t[i], y[i], dt)
    
    # Imprime avance cada 300,000 pasos para monitorear la ejecución
    if (i + 1) % 300000 == 0:
        print(f"Progreso integración: {((i+1)/n)*100:.0f}%")

# ----------------------------------------------------
# Separación de Variables
# ----------------------------------------------------
x, v = y[:, 0], y[:, 1]

# ----------------------------------------------------
# Puntos Estroboscópicos (Optimizado sin np.argmin)
# ----------------------------------------------------
print("Calculando Mapa Estroboscópico...")
PE = []
for j in range(Trans, Nperiods + 1):
    tj = j * T
    index = int(round(tj / dt)) # Cálculo directo de índice
    if index < len(x):
        PE.append([x[index], v[index]])

PE = np.array(PE)

# ----------------------------------------------------
# Gráfica del Mapa Estroboscópico
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(PE[:, 0], PE[:, 1], s=5, color='red')
ax.set_xlabel(r'$x$', fontsize=22)
ax.set_ylabel(r'$\dot{x}$', fontsize=22)
ax.tick_params(axis='both', labelsize=18)
ax.set_title('Stroboscopic map', fontsize=20)
ax.set_box_aspect(0.65)

plt.tight_layout()
plt.savefig("Stroboscopic.pdf", format="pdf", bbox_inches="tight")
print("Listo! Calgando...")
plt.show()