import numpy as np
import matplotlib.pyplot as plt

# Semilla aleatoria para reproducibilidad
np.random.seed(42)

# Definir los 8 vértices del cuadrado unidad
V = np.array([
    [0.0, 0.0],
    [0.5, 0.0],
    [1.0, 0.0],
    [0.0, 0.5],
    [1.0, 0.5],
    [0.0, 1.0],
    [0.5, 1.0],
    [1.0, 1.0]
])

N = 1000000
P = np.zeros((N, 2))
P[0] = [0.25, 0.75]

# Juego del caos
indices = np.random.randint(0, 8, size=N - 1)
for n in range(N - 1):
    P[n + 1] = (1.0 / 3.0) * P[n] + (2.0 / 3.0) * V[indices[n]]

# Crear figura con DPI por defecto para la pantalla
fig, ax = plt.subplots(figsize=(6, 6))

# Dibujar puntos descartando el transitorio
ax.scatter(P[100:, 0], P[100:, 1], s=0.08, color='red', alpha=1.0, linewidths=0, rasterized=True)

# Encuadre perfecto
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')

# Texto posicionado en la esquina superior izquierda sin chocar con el borde
ax.set_title(r'$N = 1000000$', loc='left', fontsize=11, pad=6)

# Configurar números de los ejes
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis='both', labelsize=10)

plt.tight_layout()

# Guardar la imagen en alta definición (aquí sí va el dpi=300)
plt.savefig('actividad9_sierpinski.png', format='png', dpi=300, bbox_inches='tight')
plt.savefig('actividad9_sierpinski.pdf', format='pdf', bbox_inches='tight')

# Mostrar en pantalla limpia
plt.show()