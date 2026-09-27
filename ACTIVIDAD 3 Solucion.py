import numpy as np
import matplotlib
import os

# Configuración de backend adaptativo (TkAgg en Windows/VS Code local, Agg en entornos headless)
# Este es un codigo que use especificamente para mi pc, es un bloque hace que el código sea inteligente y universal
# detecta si estoy en Windows para abrir la ventana nativa, o si estoy en un servidor,
# para guardar la imagen silenciosamente sin que el programa se caiga
try:
    if os.name == 'nt' or 'DISPLAY' in os.environ:
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# =====================================================================
# Parámetros Físicos del sistema
# =====================================================================
# Ecuación de movimiento: d2theta/dt2 + (g/l)*sin(theta) = (F0 / (m*l)) * cos(omega * t)
g_l = 1.0           # g / l = 1.0
m_l = 1.0           # m * l = 1.0
F0 = 0.01           # Amplitud de la fuerza externa F0 = 0.01
omega = 2.0 / np.pi  # Frecuencia angular de la fuerza omega = 2 / pi

# Período de la fuerza impulsora: T = 2*pi / omega = pi^2 ≈ 9.8696 s
T = 2.0 * np.pi / omega

# =====================================================================
# Sistema de Ecuaciones Diferenciales (Vectorizado)
# =====================================================================
def derivatives_vec(t, theta, w):
    dtheta = w
    dw = - g_l * np.sin(theta) + (F0 / m_l) * np.cos(omega * t)
    return dtheta, dw

# =====================================================================
# Integrador Manual Runge-Kutta de 4º Orden (RK4 Vectorizado)
# =====================================================================
def rk4_step_vec(t, theta, w, dt):
    k1_th, k1_w = derivatives_vec(t, theta, w)
    k2_th, k2_w = derivatives_vec(t + 0.5*dt, theta + 0.5*dt*k1_th, w + 0.5*dt*k1_w)
    k3_th, k3_w = derivatives_vec(t + 0.5*dt, theta + 0.5*dt*k2_th, w + 0.5*dt*k2_w)
    k4_th, k4_w = derivatives_vec(t + dt, theta + dt*k3_th, w + dt*k3_w)
    
    theta_new = theta + (dt / 6.0) * (k1_th + 2.0*k2_th + 2.0*k3_th + k4_th)
    w_new = w + (dt / 6.0) * (k1_w + 2.0*k2_w + 2.0*k3_w + k4_w)
    return theta_new, w_new

# =====================================================================
# Generación de la Malla 2D de Condiciones Iniciales
# =====================================================================
N_th, N_w = 30, 30  # Malla uniforme de 30x30 = 900 trayectorias
theta_grid = np.linspace(-np.pi, np.pi, N_th)
w_grid = np.linspace(-2.5, 2.5, N_w)
TH0, W0 = np.meshgrid(theta_grid, w_grid)

th = TH0.flatten()
w = W0.flatten()
N_trajectories = len(th)

# Asignar un color característico a cada trayectoria basado en la velocidad inicial
color_id = W0.flatten()

# Parámetros del mapa estroboscópico
n_periods = 250        # Número de flashes/puntos por trayectoria
steps_per_period = 100 # Pasos de integración RK4 por período T
dt = T / steps_per_period

strobe_th = np.zeros((n_periods, N_trajectories))
strobe_w = np.zeros((n_periods, N_trajectories))

print(f"Calculando mapa estroboscópico para {N_trajectories} condiciones iniciales...")

# =====================================================================
# Simulación Estroboscópica
# =====================================================================
t = 0.0
for n in range(n_periods):
    # Avanzar exactamente 1 período T
    for _ in range(steps_per_period):
        th, w = rk4_step_vec(t, th, w, dt)
        t += dt
    
    # Mapear theta al intervalo [-pi, pi]
    th_wrapped = (th + np.pi) % (2.0 * np.pi) - np.pi
    strobe_th[n, :] = th_wrapped
    strobe_w[n, :] = w

print("Simulación completada. Renderizando figura...")

# =====================================================================
# Preparación de Datos y Graficación
# =====================================================================
th_flat = strobe_th.flatten()
w_flat = strobe_w.flatten()
color_flat = np.tile(color_id, n_periods)

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)

# Graficar todos los puntos con paleta multicolor 'rainbow'
sc = ax.scatter(th_flat, w_flat, c=color_flat, cmap='rainbow', s=0.2, alpha=0.85)

ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-2.6, 2.6)
ax.set_xlabel(r"Posición Angular $\theta$ (rad)", fontsize=12)
ax.set_ylabel(r"Velocidad Angular $\dot{\theta}$ (rad/s)", fontsize=12)
ax.set_title(r"Mapa Estroboscópico del Péndulo Simple Forzado ($F_0=0.01, \omega=2/\pi$)" + "\n" +
             r"Espacio de Fase Coloreado por Condición Inicial (RK4 Vectorizado)", fontsize=11, fontweight='bold')
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()

# Guardar figura en el directorio local
plt.savefig("actividad3_mapa_estroboscopico.png", bbox_inches="tight", dpi=300)
print("¡Imagen guardada exitosamente como 'actividad3_mapa_estroboscopico.png'!")

# Este es otro codigo porque mi pc esta lenta y no me cargan rapido el mapa estreboscopico, entonces
# se abre la ventana gráfica en la pantalla de Windows para que pueda ver la figura e interactuar con ella
try:
    plt.show()
except Exception:
    pass
