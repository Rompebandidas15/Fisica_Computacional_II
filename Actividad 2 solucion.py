import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------------------------------
# Parametros del sistema
# -----------------------------------------------------
g = 9.81      
M = 2.0       
m = 1.0       
k = 10.0      
l = 1.0       

# -----------------------------------------------------
# Condiciones Iniciales
# -----------------------------------------------------
x0 = 0.3                  # Posicion inicial del bloque M (m)
v0 = 0.0                  # Velocidad inicial del bloque (m/s)
th0 = np.radians(30.0)    # angulo inicial del pendulo (30 grados)
w0 = 0.0                  # Velocidad angular inicial (rad/s)

# -----------------------------------------------------
# Parametros del metodo
# -----------------------------------------------------
tmax = 12.0   # Tiempo maximo de simulacion (s)
dt = 0.01     # Paso temporal de integracion 
STRIDE = 2   # Los fotogramas que muestra en la animacion

# -----------------------------------------------------
# La dinamica del Bloque mas el Pendulo con resorte
# -----------------------------------------------------
def dyn(t, y):
    x, v, th, w = y
    sd = sin(th)
    cd = cos(th)
    
    # Denominador comun segun Ec. (1) 
    den = (M / m) + sd**2
    
    # Aceleracion del bloque M (vDot = x'')
    dvdt = ((g * cd + l * w**2) * sd - (k / m) * x) / den
    
    # Aceleracion angular del pendulo m (wDot = theta'')
    dwdt = -(1.0 / l) * (g * (1.0 + M / m) * sd + cd * (l * w**2 * sd - (k / m) * x)) / den
    
    return np.array([v, dvdt, w, dwdt])

# -----------------------------------------------------
# El metodo de RK4
# -----------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# -----------------------------------------------------
# Integrar usando RK4
# -----------------------------------------------------
n = int(tmax / dt)
t = np.linspace(0, n * dt, n + 1)
y = np.empty((n + 1, 4))
y[0] = np.array([x0, v0, th0, w0])

for i in range(n):
    y[i + 1] = rk4(dyn, t[i], y[i], dt)

# -----------------------------------------------------
# Separacion de variables despues de la integracion
# -----------------------------------------------------
x_block = y[:, 0]
v_block = y[:, 1]
th_pend = y[:, 2]
w_pend = y[:, 3]

# -----------------------------------------------------
# Cinematica
# -----------------------------------------------------
# la posicion cartesiana del bloque M
x_M = x_block
y_M = np.zeros_like(x_M)

# la posicion cartesiana de la masa m
x_m = x_M + l * sin(th_pend)
y_m = y_M - l * cos(th_pend)

# -----------------------------------------------------
# Codigo para la figura
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.set(xlim=(-2.0, 2.0), ylim=(-1.8, 0.8), aspect="equal", title="Actividad 2: Bloque-Resorte + Péndulo (RK4)")
ax.title.set_fontsize(14)
ax.tick_params(axis="both", labelsize=10)
ax.grid(alpha=0.3)

# los elementos graficos de la animacion
block_plot, = ax.plot([], [], "s", markersize=12, color="#e74c3c", label="Bloque M")
line_plot, = ax.plot([], [], "o-", lw=2, color="#2c3e50", markersize=8, label="Péndulo m")
trace_plot, = ax.plot([], [], "-", lw=1.2, alpha=0.6, color="#3498db", label="Estela masa m")
spring_line, = ax.plot([], [], "--", lw=1.5, color="#7f8c8d", label="Resorte k")
clock = ax.text(0.05, 0.92, "", transform=ax.transAxes, fontsize=11, fontweight='bold')

ax.legend(loc="upper right", fontsize=9)

# -----------------------------------------------------
# se crea la animacion, el trance_plot es para que dibuje la trayectoria del pendulo y cree un dibujo asi
# -----------------------------------------------------
def animate(i):
    spring_line.set_data([-1.5, x_M[i]], [0, 0])
    block_plot.set_data([x_M[i]], [y_M[i]])
    line_plot.set_data([x_M[i], x_m[i]], [y_M[i], y_m[i]])
    start_idx = max(0, i - 100)
    trace_plot.set_data(x_m[start_idx:i+1], y_m[start_idx:i+1])
    clock.set_text(f"t = {t[i]:.2f} s")
    return block_plot, line_plot, trace_plot, spring_line, clock

# -----------------------------------------------------
# Animacion
# -----------------------------------------------------
ani = FuncAnimation(fig, animate, frames=range(0, n + 1, STRIDE),
                    interval=STRIDE * dt * 1000, blit=True)

plt.tight_layout()
plt.show()