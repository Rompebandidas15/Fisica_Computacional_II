## actividad en clase 19/10
## Andres Felipe Rojas Tafur

import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

#parametros sistema
mu = 2.8
# Valores extremos teóricos
rdot_max = np.sqrt(2 / (mu + 1))  # = 0.725
r_max = 1 / (mu - 1)             # = 5/9 ≈ 0.555
# Tramo de 500 fronteras
rdot = np.linspace(-rdot_max, rdot_max, 500)
# Puntos para la velocidad radial dot(r). ecuacion de la frontera para no moelstar con una raiz cuadrada de doble signo, 
# se usa la ecuacion de la frontera
r_boundary = (2 - (1 + mu) * rdot**2) / (2 * (mu - 1))


# Creación de la figura
fig, ax = plt.subplots(figsize=(7, 5))

# Curva frontera y sombreado de la región admisible
ax.plot(r_boundary, rdot, color='#1f4e79', linewidth=1.5)
ax.fill_betweenx(rdot, 0, r_boundary, color='#7faecb', alpha=0.7)

# Etiquetas y límites de los ejes
ax.set_xlabel(r'$r$', fontsize=14)
ax.set_ylabel(r'$\dot{r}$', fontsize=14)
ax.set_xlim(0, 0.6)
ax.set_ylim(-0.8, 0.8)

# Etiqueta del parámetro mu
ax.text(0.92, 0.93, r'$\mu = 2.8$', transform=ax.transAxes, 
        ha='right', va='top', fontsize=12)

# Estilo y cuadrícula
ax.tick_params(axis='both', labelsize=11)
ax.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()

# Mostrar y guardar figura
plt.savefig('seccion_poincare.pdf', format='pdf', bbox_inches='tight')
plt.show()