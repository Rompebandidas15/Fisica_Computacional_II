#------------------------------------------------------
# Andres Felipe Rojas Tafur
# 070400372024
#------------------------------------------------------



import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------
# Parameters
#---------------------------------------------------
m = 1
k = 0.5
l = 1

#---------------------------------------------------
# Low Energies
#---------------------------------------------------
# mínimo físico real para el límite inferior:
E_min = -(k**2) / (4 * l)      

a1 = E_min                     # Se empieza en el fondo del pozo que se calcula con la ecuacion (en vez de -3) Corregí eso
b1 = 0.0                       # Se llega hasta la colina central (E = 0)
h1 = 0.005                     # Paso fino para dibujar contornos perfectos
v1 = np.arange(a1, b1 + h1, h1)

#----------------------------------------------------
# High energies
#----------------------------------------------------
a2 = 0.1
b2 = 10.0
h2 = 0.5
v2 = np.arange(a2, b2 + h2, h2)

#----------------------------------------------------
# Combine the energy vectors
#----------------------------------------------------
v = np.concatenate((v1, v2))

#----------------------------------------------------
# Duffing potential
#----------------------------------------------------
def U(x):
  return -(k/2)*x**2 + (l/4)*x**4

#----------------------------------------------------
# Espacio de fase
#----------------------------------------------------
x = np.linspace(-3, 3, 500)
y = np.linspace(-3, 3, 500)
X, Y = np.meshgrid(x, y)

#----------------------------------------------------
# Total energy (Hamiltoniano del sistema)
#----------------------------------------------------
E = (m/2)*Y**2 + U(X)

#----------------------------------------------------
# Constant-energy curves
#----------------------------------------------------
plt.figure(figsize=(10, 6))
plt.contour(X, Y, E, levels=v)
plt.xlabel(r"$x$", fontsize=16)
plt.ylabel(r"$\dot{x}$", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


