import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# 1. PARÁMETROS FÍSICOS DEL SISTEMA 

g = 9.81      # Gravedad (m/s^2)
l = 1.0       # Longitud de la cuerda (m)
alpha = 0.5   # Parámetro de la parábola z = alpha * x^2
m = 1.0       # Masa del péndulo (kg)
M = 0.5       # Masa del soporte (kg) -> Evita la división por cero matemática


# 2. ECUACIONES DE MOVIMIENTO (Derivadas del Lagrangiano)

def ecuaciones_pendulo(t, variables):
    x, v, theta, omega = variables
    
    # Denominador (siempre positivo por la masa del soporte M)
    den = (4 * M * alpha**2 * x**2 + M + 2 * alpha**2 * m * x**2 * np.cos(2*theta) 
           + 2 * alpha**2 * m * x**2 - 2 * alpha * m * x * np.sin(2*theta) 
           - 0.5 * m * np.cos(2*theta) + 0.5 * m)
    
    # x'' (Aceleración lineal del soporte en la parábola)
    num_x = (- 4 * M * alpha**2 * x * v**2 
             - 2 * M * alpha * g * x 
             - 2 * alpha**2 * m * x * np.cos(2*theta) * v**2 
             - 2 * alpha**2 * m * x * v**2 
             - alpha * g * m * x * np.cos(2*theta) 
             - alpha * g * m * x 
             - 2 * alpha * l * m * x * np.cos(theta) * omega**2 
             + alpha * m * np.sin(2*theta) * v**2 
             + 0.5 * g * m * np.sin(2*theta) 
             + l * m * np.sin(theta) * omega**2)
    
    # theta'' (Aceleración angular del péndulo)
    num_theta = (4 * M * alpha**2 * x * np.cos(theta) * v**2 
                 + 2 * M * alpha * g * x * np.cos(theta) 
                 - 2 * M * alpha * np.sin(theta) * v**2 
                 - M * g * np.sin(theta) 
                 + 2 * alpha**2 * l * m * x**2 * np.sin(2*theta) * omega**2 
                 + 4 * alpha**2 * m * x * np.cos(theta) * v**2 
                 + 2 * alpha * g * m * x * np.cos(theta) 
                 + 2 * alpha * l * m * x * np.cos(2*theta) * omega**2 
                 - 2 * alpha * m * np.sin(theta) * v**2 
                 - g * m * np.sin(theta) 
                 - 0.5 * l * m * np.sin(2*theta) * omega**2)
    
    #  [dx/dt, dv/dt, dtheta/dt, domega/dt]
    return [v, num_x / den, omega, num_theta / (l * den)]


# 3. CONDICIONES INICIALES 

# Posición inicial: Soporte en x = 0.8 metros, ángulo del péndulo a 45 grados (pi/4 radianes)
condiciones_iniciales = [0.8, 0.0, np.pi/4, 0.0]
tiempo_simulacion = (0, 10)  # De 0 a 10 segundos
puntos_tiempo = np.linspace(0, 10, 300)  # 300 fotogramas para la animación

# Se resuelve usando el método RK45 (Runge-Kutta de 4to y 5to orden)
solucion = solve_ivp(ecuaciones_pendulo, tiempo_simulacion, condiciones_iniciales, t_eval=puntos_tiempo)

# extraer los datos del resultado seleccionando la fila correcta:
x_soporte = solucion.y[0]      # [0] es la posición x del soporte
theta_pendulo = solucion.y[2]  # [2] es el ángulo theta del péndulo


# 4. CONFIGURACIÓN DE LA ANIMACIÓN VISUAL

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Animación: Péndulo en Guía Parabólica", fontsize=12, fontweight='bold')
ax.set_xlabel("Eje X (metros)")
ax.set_ylabel("Eje Z (metros)")

# Dibujo de la guía parabólica estática de fondo (z = alpha * x^2)
x_parabola = np.linspace(-2.0, 2.0, 300)
z_parabola = alpha * x_parabola**2
ax.plot(x_parabola, z_parabola, '--', color='#bdc3c7', label='Riel Parabólico')

# Elementos móviles
estela, = ax.plot([], [], '-', color='#95a5a6', alpha=0.4, linewidth=1.5, label='Trayectoria anterior')
cuerda, = ax.plot([], [], 'o-', color='#2c3e50', linewidth=2, zorder=2)
soporte, = ax.plot([], [], 's', color='#e74c3c', markersize=10, zorder=3, label='Soporte móvil')
masa, = ax.plot([], [], 'o', color='#3498db', markersize=12, zorder=3, label='Masa péndulo ($m$)')

ax.legend(loc='upper right')

# Historial para dibujar la estela
historial_x = []
historial_z = []

# Función para actualizar cada fotograma 
def actualizar(frame):
    # Coordenadas del soporte
    xs = x_soporte[frame]
    zs = alpha * xs**2
    
    # Coordenadas de la masa del péndulo
    xm = xs + l * np.sin(theta_pendulo[frame])
    zm = zs - l * np.cos(theta_pendulo[frame])
    
    # historial de la estela
    historial_x.append(xm)
    historial_z.append(zm)
    
    # Actualizar gráficos
    estela.set_data(historial_x[-50:], historial_z[-50:]) # Muestra los últimos 50 puntos
    cuerda.set_data([xs, xm], [zs, zm])
    soporte.set_data([xs], [zs])
    masa.set_data([xm], [zm])
    
    return cuerda, soporte, masa, estela

#animación a 30 fotogramas por segundo (interval = 33 milisegundos)
ani = animation.FuncAnimation(fig, actualizar, frames=len(puntos_tiempo), blit=True, interval=33)

plt.tight_layout()
plt.show()