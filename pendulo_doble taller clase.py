import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------------------------------
# Parámetros físicos del sistema (de la Clase 2)
# -----------------------------------------------------
g = 9.81     # Aceleración de la gravedad (m/s^2)
l1 = 2.0     # Longitud de la primera barra (m)
l2 = 1.0     # Longitud de la segunda barra (m)
m1 = 4.0     # Masa del primer péndulo (kg)
m2 = 1.0     # Masa del segundo péndulo (kg)

# -----------------------------------------------------
# Condiciones iniciales
# -----------------------------------------------------
# Ángulos iniciales convertidos a radianes (10 y -10 grados)
th10 = np.radians(10.0)
th20 = np.radians(-10.0)
# Velocidades angulares iniciales (en reposo)
ome10 = 0.0
ome20 = 0.0

# -----------------------------------------------------
# Parámetros del método de integración
# -----------------------------------------------------
tmax = 20.0  # Tiempo máximo de simulación (s)
dt = 0.01    # Tamaño del paso temporal (s)
STRIDE = 2   # Mostrar 1 de cada 2 fotogramas en la animación

# -----------------------------------------------------
# Dinámica: Sistema de ecuaciones del péndulo doble
# -----------------------------------------------------
def dyn(t, y_state):
    th1, w1, th2, w2 = y_state
    d = th1 - th2
    sd = sin(d)
    cd = cos(d)
    
    # Denominador común de las ecuaciones de movimiento
    den = 2*m1 + m2 - m2*cos(2*th1 - 2*th2)
    
    # Aceleración angular del péndulo 1 (theta1'')
    a1 = (-g*(2*m1 + m2)*sin(th1) - g*m2*sin(th1 - 2*th2) - 2*m2*sd*(l1*cd*w1**2 + l2*w2**2))/(l1*den)
    
    # Aceleración angular del péndulo 2 (theta2'')
    a2 = (2*sd*(g*(m1 + m2)*cos(th1) + l1*(m1 + m2)*w1**2 + l2*m2*cd*w2**2))/(l2*den)
    
    # Retornamos las derivadas: [dth1/dt, dw1/dt, dth2/dt, dw2/dt]
    return np.array([w1, a1, w2, a2])

# -----------------------------------------------------
# Método de integración numérica: Runge-Kutta de 4to Orden (RK4)
# -----------------------------------------------------
def rk4(f, t, y_state, h):
    k1 = h * f(t, y_state)
    k2 = h * f(t + h/2, y_state + k1/2)
    k3 = h * f(t + h/2, y_state + k2/2)
    k4 = h * f(t + h, y_state + k3)
    return y_state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# -----------------------------------------------------
# Proceso de integración en el tiempo
# -----------------------------------------------------
n = int(tmax / dt)
t = np.linspace(0, n * dt, n + 1)
y_data = np.empty((n + 1, 4))
y_data[0] = np.array([th10, ome10, th20, ome20])

# Llenamos la matriz de estados aplicando RK4 paso a paso
for i in range(n):
    y_data[i + 1] = rk4(dyn, t[i], y_data[i], dt)

# Extraemos los ángulos resultantes
th1 = y_data[:, 0]
th2 = y_data[:, 2]

# -----------------------------------------------------
# Cinemática: Cambio de coordenadas angulares a cartesianas (X, Y)
# -----------------------------------------------------
# Posición de la masa 1 (colgada del origen)
x1, y1 = l1 * sin(th1), -l1 * cos(th1)
# Posición de la masa 2 (colgada de la masa 1)
x2, y2 = x1 + l2 * sin(th2), y1 - l2 * cos(th2)

# -----------------------------------------------------
# Configuración visual de la animación
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
R = l1 + l2 + 0.2
ax.set(xlim=(-R, R), ylim=(-R, R), aspect="equal", title="Simulación: Péndulo Doble (RK4 - Clase 2)")
ax.tick_params(axis="both", labelsize=10)
ax.grid(alpha=0.3)

# Elementos gráficos que se actualizarán
line, = ax.plot([], [], "o-", lw=2, color="#2c3e50", markersize=8, label="Péndulos")
trace, = ax.plot([], [], "-", lw=1.5, alpha=0.6, color="#e74c3c", label="Estela (Masa 2)")
clock = ax.text(0.05, 0.93, "", transform=ax.transAxes, fontsize=11, fontweight='bold')

ax.legend(loc="upper right")

# Función que dibuja cada fotograma
def animate(i):
    # Dibuja la línea que une el Origen (0,0) -> Masa 1 (x1, y1) -> Masa 2 (x2, y2)
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    # Dibuja el rastro histórico recorrido por la masa 2
    trace.set_data(x2[:i+1], y2[:i+1])
    # Actualiza el reloj de tiempo
    clock.set_text(f"t = {t[i]:.2f} s")
    return line, trace, clock

# Creamos la animación interactiva
ani = FuncAnimation(fig, animate, frames=range(0, n + 1, STRIDE), 
                    interval=STRIDE * dt * 1000, blit=True)

plt.tight_layout()
plt.show()
