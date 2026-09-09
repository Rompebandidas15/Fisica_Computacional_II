import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt

#------------------------------------------------
# System Parameters
#------------------------------------------------
g = 9.81
m = 1
k = 1
l = 1
F = 1
om = 2/3*pi

#------------------------------------------------
# Initial Conditions
#------------------------------------------------
x0 = 1.0
v0 = 0.0

#------------------------------------------------
# Method parameters
#------------------------------------------------
tmax = 300
dt = 0.001

#------------------------------------------------
# Dynamics: Forced Duffing Oscillator
#------------------------------------------------
def dyn(t,y):
    x, v = y
    dx = v
    dv = (k/m)*x - (l/m)*x**3 + (F/m)*cos(om*t)
    return np.array([dx,dv])

#------------------------------------------------
# Fourth-order Runge Kutta method
#------------------------------------------------
def rk4(f,t,y,h):
    k1 = h * f(t,y)
    k2 = h * f(t + h/2, y + k1/2)
    k3=h * f(t+h/2,y+k2/2)
    k4=h * f(t+h,y+k3)
    return y+(k1+2*k2+2*k3+k4)/6
 #----------------------------------------------------

#IntegrationusingRK4
#----------------------------------------------------
n = int(tmax/dt)
t = np.linspace(0,n * dt, n+1) 
y = np.empty((n+1,2))
y[0]=[x0,v0]
for i in range(n):
  y[i+1]=rk4(dyn,t[i],y[i],dt)

#----------------------------------------------------
 #Separatevariablesafter integration
#----------------------------------------------------
x=y[:,0]
v=y[:,1]

#----------------------------------------------------
#Phasespace
#----------------------------------------------------
## Le cambie el diseño
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#121212') # Fondo figura
ax.set_facecolor('#121212')        # Fondo gráfica

# Línea brillante con transparencia
ax.plot(x, v, color='#00ffcc', linewidth=0.2, alpha=0.5)

# Configuración de ejes para que sean blancos
ax.set_xlabel(r'$x$', fontsize=22, color='white')
ax.set_ylabel(r'$\dot{x}$', fontsize=22, color='white')
ax.tick_params(axis='both', colors='white', labelsize=20)
for spine in ax.spines.values():
    spine.set_edgecolor('white')

plt.tight_layout()
ax.set_box_aspect(0.65)
plt.savefig("Dufpy_Dark.pdf", format="pdf", bbox_inches="tight")
plt.show()

##Andres Felipe Rojas Tafur
##070400372024