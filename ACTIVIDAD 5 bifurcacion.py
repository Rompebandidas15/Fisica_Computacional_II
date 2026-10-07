
#=============================================================================
#UNIVERSIDAD DEL TOLIMA - FÍSICA COMPUTACIONAL II
#Actividad 5: Diagramas de Bifurcación (Oscilador de Duffing)
#Estudiante: Andrés Felipe Rojas Tafur | Código: 070400372024
#Profesor: Dr. César Andrés Morales Rodríguez
#=============================================================================


import numpy as np
import matplotlib.pyplot as plt

# Parámetros del sistema de la Actividad 5
delta = 0.1
alpha = 2.0
beta = 2.0
omega = 1.2
T = 2.0 * np.pi / omega  # Período de forzamiento T = 2*pi / 1.2 s

# Condición Inicial fijada en la guía
x0, v0 = 1.0, 1.0

# Barrido del parámetro de forzamiento gamma en [0.1, 7.0]
gamma_vals = np.linspace(0.1, 7.0, 1000)
n_gamma = len(gamma_vals)

# Parámetros del método numérico
Trans = 300       # Períodos transitorios a descartar
Nkeep = 150       # Períodos conservados en el atractor
steps_per_T = 200 # Pasos de integración por período T
dt = T / steps_per_T
total_steps = (Trans + Nkeep) * steps_per_T

print(f"Calculando diagramas de bifurcación para {n_gamma} valores de gamma en [0.1, 7.0]...")

# Vector de estados iniciales para las n_gamma órbitas en paralelo
y = np.zeros(2 * n_gamma)
y[:n_gamma] = x0
y[n_gamma:] = v0

def dyn(t, y_vec):
    x = y_vec[:n_gamma]
    v = y_vec[n_gamma:]
    dx = v
    dv = - delta * v + alpha * x - beta * (x**3) + gamma_vals * np.cos(omega * t)
    return np.concatenate([dx, dv])

def rk4_step_vec(f, t, y_vec, h):
    k1 = h * f(t, y_vec)
    k2 = h * f(t + 0.5 * h, y_vec + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y_vec + 0.5 * k2)
    k4 = h * f(t + h, y_vec + k3)
    return y_vec + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

x_strobe = np.empty((Nkeep, n_gamma))
v_strobe = np.empty((Nkeep, n_gamma))

save_idx = 0
for step in range(total_steps):
    t_curr = step * dt
    y = rk4_step_vec(dyn, t_curr, y, dt)
    comp_period = (step + 1) // steps_per_T
    if (step + 1) % steps_per_T == 0 and comp_period > Trans:
        x_strobe[save_idx] = y[:n_gamma]
        v_strobe[save_idx] = y[n_gamma:]
        save_idx += 1

print("Simulación completada. Renderizando figuras...")

# Figura 1: Diagrama de Bifurcación x vs gamma (Dutta & Prajapati 2016, Fig. 1a)
fig1, ax1 = plt.subplots(figsize=(10, 5.5), dpi=150)
for i in range(n_gamma):
    ax1.scatter(np.full(Nkeep, gamma_vals[i]), x_strobe[:, i], s=0.1, color='blue', alpha=0.6, rasterized=True)
ax1.set_xlim(0, 7)
ax1.set_ylim(-1.8, 3.2)
ax1.set_xlabel(r"$\gamma$", fontsize=14)
ax1.set_ylabel(r"$x$", fontsize=14)
ax1.set_title(r"Actividad 5 - Ejercicio 2(a): Diagrama de Bifurcación $x$ vs $\gamma$ ($\delta=0.1, \alpha=2, \beta=2, \omega=1.2$)", fontsize=11, fontweight='bold')
ax1.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("actividad5_bifurcacion_x.png", dpi=300, bbox_inches="tight")

# Figura 2: Diagrama de Bifurcación v vs gamma (Dutta & Prajapati 2016, Fig. 1b)
fig2, ax2 = plt.subplots(figsize=(10, 5.5), dpi=150)
for i in range(n_gamma):
    ax2.scatter(np.full(Nkeep, gamma_vals[i]), v_strobe[:, i], s=0.1, color='blue', alpha=0.6, rasterized=True)
ax2.set_xlim(0, 7)
ax2.set_ylim(-6.5, 6.5)
ax2.set_xlabel(r"$\gamma$", fontsize=14)
ax2.set_ylabel(r"$\dot{x}$", fontsize=14)
ax2.set_title(r"Actividad 5 - Ejercicio 2(b): Diagrama de Bifurcación $\dot{x}$ vs $\gamma$ ($\delta=0.1, \alpha=2, \beta=2, \omega=1.2$)" + "\n" +
              r"Comparación con Dutta & Prajapati (2016, Fig. 1b)", fontsize=11, fontweight='bold', pad=12)
ax2.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("actividad5_bifurcacion_v.png", dpi=300, bbox_inches="tight")

print("¡Gráficas guardadas exitosamente como 'actividad5_bifurcacion_x.png' y 'actividad5_bifurcacion_v.png'!")

try:
    plt.show()
except Exception:
    pass
