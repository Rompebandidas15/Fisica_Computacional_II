# Andres Felipe Rojas Tafur
# Codigo:070400372024


import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# Condiciones Iniciales y Arrays 
# -----------------------------------------------------
P = np.array([0.5, 0.0])
N = 2000000
x = np.zeros(N)
y = np.zeros(N)

# -----------------------------------------------------
# Juego del Caos 
# -----------------------------------------------------
np.random.seed(42)

for n in range(N):
    r = np.random.random()
    xn = P[0]
    yn = P[1]

    # T1: P(T1) = 0.10 -> (r < 0.10)
    if r < 0.10:
        P = np.array([0.05 * xn, 0.6 * yn])

    # T2: P(T2) = 0.10 -> (0.10 <= r < 0.20)
    elif r < 0.20:
        P = np.array([0.05 * xn, -0.5 * yn + 1.0])

    # T3: P(T3) = 0.20 -> (0.20 <= r < 0.40)
    elif r < 0.40:
        P = np.array([0.46 * xn - 0.15 * yn, 0.39 * xn + 0.38 * yn + 0.6])

    # T4: P(T4) = 0.20 -> (0.40 <= r < 0.60)
    elif r < 0.60:
        P = np.array([0.47 * xn - 0.15 * yn, 0.17 * xn + 0.42 * yn + 1.1])

    # T5: P(T5) = 0.20 -> (0.60 <= r < 0.80)
    elif r < 0.80:
        P = np.array([0.43 * xn + 0.28 * yn, -0.25 * xn + 0.45 * yn + 1.0])

    # T6: P(T6) = 0.20 -> (0.80 <= r <= 1.00)
    else:
        P = np.array([0.42 * xn + 0.26 * yn, -0.35 * xn + 0.31 * yn + 0.7])

    x[n] = P[0]
    y[n] = P[1]

# -----------------------------------------------------
# Algoritmo de Conteo de Cajas (Box-Counting)
# -----------------------------------------------------
num_sizes = 35
epsilons = np.logspace(np.log10(0.0002), np.log10(0.5), num_sizes)
box_counts = []

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

for eps in epsilons:
    ix = np.floor((x - xmin) / eps).astype(int)
    iy = np.floor((y - ymin) / eps).astype(int)
    unique_boxes = set(zip(ix, iy))
    box_counts.append(len(unique_boxes))

box_counts = np.array(box_counts)
inv_eps = 1.0 / epsilons
log_inv_eps = np.log(inv_eps)
log_N = np.log(box_counts)

# Intervalo de escalamiento autosimilar (0.002 <= eps <= 0.08)
mask_scaling = (epsilons >= 0.002) & (epsilons <= 0.08)
slope, intercept = np.polyfit(log_inv_eps[mask_scaling], log_N[mask_scaling], 1)

print(f"Dimensión Fractal estimada (D) = {slope:.4f}")

# -----------------------------------------------------
# Figura 1 - Fractal Ocobo
# -----------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(7, 9), dpi=100)
ax1.scatter(x, y, s=0.01, color='#e6007e', alpha=0.6, rasterized=True)
ax1.set_xlim(-0.55, 0.85)
ax1.set_ylim(-0.05, 2.05)
ax1.set_aspect('equal')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.tick_params(axis='both', labelsize=10)
ax1.grid(True, linestyle=':', alpha=0.4)
ax1.set_title(rf"Actividad 10 - Fractal Ocobo ($N = {N:,}$, $D_{{box}} = {slope:.4f}$)" + "\n" +
              r"Semillero MECA - Universidad del Tolima", fontsize=11, fontweight='bold', pad=12)

# (Se omitió ax1.text para que la copa del árbol quede 100% limpia sin superposiciones)

plt.tight_layout()
plt.savefig("actividad10_ocobo_fractal.pdf", bbox_inches='tight', dpi=800)
plt.savefig("actividad10_ocobo_fractal.png", bbox_inches='tight', dpi=300)

# -----------------------------------------------------
# Figura 2 - Conteo de Cajas (Box-Counting Limpio)
# -----------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=100)

ax2.plot(log_inv_eps[~mask_scaling], log_N[~mask_scaling], 'ro', markersize=6, label='Zonas de desviación (Saturación/Discreción)')
ax2.plot(log_inv_eps[mask_scaling], log_N[mask_scaling], 'bo', markersize=7, label='Región de escalamiento (Ajuste lineal)')

fit_x = np.linspace(log_inv_eps.min(), log_inv_eps.max(), 100)
fit_y = slope * fit_x + intercept
ax2.plot(fit_x, fit_y, 'm--', linewidth=2, label=f'Ajuste lineal: $D = {slope:.4f}$')

ax2.set_xlabel(r'$\ln(1/\epsilon)$', fontsize=14)
ax2.set_ylabel(r'$\ln(N(\epsilon))$', fontsize=14)
ax2.set_title(r'Estimación de la Dimensión Fractal mediante Conteo de Cajas (Box-Counting)' + '\n' +
              r'Actividad 10 - Fractal Ocobo', fontsize=11, fontweight='bold', pad=12)
ax2.tick_params(axis='both', labelsize=11)
ax2.set_box_aspect(0.65)
ax2.grid(True, linestyle=':', alpha=0.6)


ax2.legend(loc='lower right', fontsize=10)
plt.tight_layout()

plt.savefig("actividad10_boxcounting.pdf", bbox_inches='tight', dpi=800)
plt.savefig("actividad10_boxcounting.png", bbox_inches='tight', dpi=300)


plt.show()
