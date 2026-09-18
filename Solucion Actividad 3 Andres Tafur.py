import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# Datos de la actividad
# =====================================================================

g_l = 1.0       # g / l = 1
m_l = 1.0       # m * l = 1
F0 = 0.01       # F0 = 0.01
omega = 2.0 / np.pi  # omega = 2 / pi
T = 2.0 * np.pi / omega  # Periodo de parpadeo

# =====================================================================
# Ecuaciones del sistema
# =====================================================================
# y = [theta, omega_vel]
def derivatives(t, y):
    theta, w = y
    dtheta = w
    dw = - g_l * np.sin(theta) + (F0 / m_l) * np.cos(omega * t)
    return np.array([dtheta, dw])

# =====================================================================
# Integrador Manual Runge-Kutta de 4º Orden (RK4)
# =====================================================================
def rk4_step(t, y, dt):
    k1 = derivatives(t, y)
    k2 = derivatives(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = derivatives(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = derivatives(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def simulate_stroboscopic_orbit(y0, n_periods=300, steps_per_period=100):
    dt = T / steps_per_period
    y = np.array(y0, dtype=float)
    t = 0.0
    
    strobe_theta = np.zeros(n_periods)
    strobe_w = np.zeros(n_periods)
    
    for n in range(n_periods):
        # Integrar exactamente un periodo T
        for _ in range(steps_per_period):
            y = rk4_step(t, y, dt)
            t += dt
        
        # Mapear theta al intervalo [-pi, pi]
        theta_wrapped = (y[0] + np.pi) % (2.0 * np.pi) - np.pi
        strobe_theta[n] = theta_wrapped
        strobe_w[n] = y[1]
        
    return strobe_theta, strobe_w

# =====================================================================
# Generación de Condiciones Iniciales para Escanear el Espacio de Fase
# =====================================================================
np.random.seed(42)
initial_conditions = []

# 1. Órbitas en el centro (libración regular)
for w0 in np.linspace(0.05, 1.8, 20):
    initial_conditions.append((0.0, w0))
    initial_conditions.append((0.0, -w0))

# 2. Órbitas cerca de la separatriz (región caótica)
for theta0 in np.linspace(-3.0, 3.0, 25):
    initial_conditions.append((theta0, 1.95))
    initial_conditions.append((theta0, 2.05))
    initial_conditions.append((theta0, -1.95))

# 3. Órbitas en la zona de rotación (foliada exterior)
for w0 in np.linspace(2.1, 2.8, 12):
    initial_conditions.append((0.0, w0))
    initial_conditions.append((0.0, -w0))

# 4. Barrido fino a lo largo del eje theta=0 y theta=pi para capturar islas
for w0 in np.linspace(0.1, 2.5, 30):
    initial_conditions.append((0.0, w0))
    initial_conditions.append((np.pi * 0.9, w0))

# =====================================================================
# Ejecución del Mapa Estroboscópico
# =====================================================================
print(f"Simulando mapa estroboscópico para {len(initial_conditions)} condiciones iniciales...")

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# Paleta de colores variada
colors = plt.cm.jet(np.linspace(0, 1, len(initial_conditions)))

for idx, y0 in enumerate(initial_conditions):
    th, w = simulate_stroboscopic_orbit(y0, n_periods=250, steps_per_period=80)
    ax.scatter(th, w, s=0.4, color=colors[idx], alpha=0.85)

ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-2.6, 2.6)
ax.set_xlabel(r"$\theta$", fontsize=14)
ax.set_ylabel(r"$\dot{\theta}$", fontsize=14)
ax.set_title(r"Mapa Estroboscópico del Péndulo Simple Forzado ($F_0=0.01, \omega=2/\pi$)" + "\n" +
             r"Región Caótica, Resonante e Islas Foliadas (RK4 Manual)", fontsize=12, fontweight='bold')
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('actividad3_mapa_estroboscopico.png', dpi=150, bbox_inches='tight')
plt.show()

print("¡Mapa estroboscópico generado e imagen guardada exitosamente!")

## Andres Felipe Rojas Tafur
## 070400372024