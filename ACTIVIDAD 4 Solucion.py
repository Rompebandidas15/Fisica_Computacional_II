"""
=============================================================================
UNIVERSIDAD DEL TOLIMA - FÍSICA COMPUTACIONAL II
Actividad 4: Mapas Estroboscópicos II 
Estudiante: Andrés Felipe Rojas Tafur
Profesor: Dr. César Andrés Morales Rodríguez
=============================================================================
"""

import os
import numpy as np
import matplotlib

# Configuración de backend adaptativo para ejecución local (VS Code / Windows) o servidor
try:
    if os.name == 'nt' or 'DISPLAY' in os.environ:
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# =============================================================================
# Método de Integración Numérica: Runge-Kutta de 4º Orden (RK4) Vectorizado
# =============================================================================
def rk4_step_vec(f, t, y_vec, h, *args):
    k1 = h * f(t, y_vec, *args)
    k2 = h * f(t + 0.5 * h, y_vec + 0.5 * k1, *args)
    k3 = h * f(t + 0.5 * h, y_vec + 0.5 * k2, *args)
    k4 = h * f(t + h, y_vec + k3, *args)
    return y_vec + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

# =============================================================================
# EJERCICIO 2: Oscilador de Duffing con delta=0.02, alpha=-1, beta=5, gamma=8, omega=0.5
# =============================================================================
def resolver_ejercicio2():
    print("--- Calculando Ejercicio 2 (Comparación con Harrison, 2011) ---")
    delta, alpha, beta, gamma, omega = 0.02, -1.0, 5.0, 8.0, 0.5
    T = 2.0 * np.pi / omega  # T = 4*pi s
    
    # Grilla de condiciones iniciales (30x30 = 900 C.I. en el espacio de fase)
    x0_vals = np.linspace(1.1, 1.7, 30)
    v0_vals = np.linspace(-2.2, 2.2, 30)
    X0, V0 = np.meshgrid(x0_vals, v0_vals)
    x_flat, v_flat = X0.ravel(), V0.ravel()
    n_orbits = len(x_flat)
    
    y = np.concatenate([x_flat, v_flat])
    
    def dyn2(t, y_vec):
        x = y_vec[:n_orbits]
        v = y_vec[n_orbits:]
        dx = v
        dv = - delta * v + alpha * x - beta * (x**3) + gamma * np.cos(omega * t)
        return np.concatenate([dx, dv])
    
    Trans = 300        # Períodos transitorios a descartar
    Nkeep = 600        # Períodos conservados en el atractor
    steps_per_T = 200  # Pasos de integración por período T
    dt = T / steps_per_T
    
    total_steps = (Trans + Nkeep) * steps_per_T
    x_strobe = np.empty((Nkeep, n_orbits))
    v_strobe = np.empty((Nkeep, n_orbits))
    
    save_idx = 0
    for step in range(total_steps):
        t_curr = step * dt
        y = rk4_step_vec(dyn2, t_curr, y, dt)
        comp_period = (step + 1) // steps_per_T
        if (step + 1) % steps_per_T == 0 and comp_period > Trans:
            x_strobe[save_idx] = y[:n_orbits]
            v_strobe[save_idx] = y[n_orbits:]
            save_idx += 1
            
    # Graficación con ventana adaptada a pantalla
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    colors = plt.cm.jet(np.linspace(0, 1, n_orbits))
    
    for i in range(n_orbits):
        ax.scatter(x_strobe[:, i], v_strobe[:, i], s=0.25, color=colors[i], alpha=0.75, rasterized=True)
        
    ax.set_xlim(1.05, 1.75)
    ax.set_ylim(-2.6, 2.6)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$\dot{x}$", fontsize=12)
    
    # Ajuste del tamaño de los números en los ejes x e y
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Título ajustado con margen de separación (pad)
    ax.set_title(r"Actividad 4 - Ejercicio 2: Mapa Estroboscópico de Duffing ($\delta=0.02, \gamma=8, \omega=0.5$)" + "\n" +
                 r"Atractor Extraño Caótico (Comparación con Harrison 2011, Fig. 1a)", fontsize=11, fontweight='bold', pad=12)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    outfile = "actividad4_mapa_ejercicio2.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"¡Gráfica del Ejercicio 2 guardada exitosamente como '{outfile}'!")
    return fig

# =============================================================================
# EJERCICIO 3: Oscilador de Duffing con delta=0.15, alpha=1, beta=1, gamma=0.3, omega=1.0
# =============================================================================
def resolver_ejercicio3():
    print("--- Calculando Ejercicio 3 (Comparación con Guckenheimer & Holmes 1983) ---")
    delta, alpha, beta, gamma, omega = 0.15, 1.0, 1.0, 0.3, 1.0
    T = 2.0 * np.pi / omega  # T = 2*pi s
    
    # Grilla de condiciones iniciales en el pozo doble (30x30 = 900 C.I.)
    x0_vals = np.linspace(-1.5, 1.5, 30)
    v0_vals = np.linspace(-1.0, 1.2, 30)
    X0, V0 = np.meshgrid(x0_vals, v0_vals)
    x_flat, v_flat = X0.ravel(), V0.ravel()
    n_orbits = len(x_flat)
    
    y = np.concatenate([x_flat, v_flat])
    
    def dyn3(t, y_vec):
        x = y_vec[:n_orbits]
        v = y_vec[n_orbits:]
        dx = v
        dv = - delta * v + alpha * x - beta * (x**3) + gamma * np.cos(omega * t)
        return np.concatenate([dx, dv])
    
    Trans = 400        # Períodos transitorios a descartar
    Nkeep = 800        # Períodos conservados en el atractor
    steps_per_T = 200
    dt = T / steps_per_T
    
    total_steps = (Trans + Nkeep) * steps_per_T
    x_strobe = np.empty((Nkeep, n_orbits))
    v_strobe = np.empty((Nkeep, n_orbits))
    
    save_idx = 0
    for step in range(total_steps):
        t_curr = step * dt
        y = rk4_step_vec(dyn3, t_curr, y, dt)
        comp_period = (step + 1) // steps_per_T
        if (step + 1) % steps_per_T == 0 and comp_period > Trans:
            x_strobe[save_idx] = y[:n_orbits]
            v_strobe[save_idx] = y[n_orbits:]
            save_idx += 1
            
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    colors = plt.cm.jet(np.linspace(0, 1, n_orbits))
    
    for i in range(n_orbits):
        ax.scatter(x_strobe[:, i], v_strobe[:, i], s=0.25, color=colors[i], alpha=0.75, rasterized=True)
        
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.7, 1.3)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$\dot{x}$", fontsize=12)
    
    # Ajuste del tamaño de los números en los ejes x e y
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Título ajustado con margen de separación (pad)
    ax.set_title(r"Actividad 4 - Ejercicio 3: Mapa Estroboscópico de Duffing ($\delta=0.15, \gamma=0.3, \omega=1.0$)" + "\n" +
                 r"Sección de Poincaré (Comparación con Guckenheimer & Holmes 1983, p. 90, Fig. 1b)", fontsize=11, fontweight='bold', pad=12)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    outfile = "actividad4_mapa_ejercicio3.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"¡Gráfica del Ejercicio 3 guardada exitosamente como '{outfile}'!")
    return fig

if __name__ == "__main__":
    resolver_ejercicio2()
    resolver_ejercicio3()
    print("\n¡Simulaciones de la Actividad 4 completadas con éxito!")
    
    # Desplegar ventana interactiva si está disponible en entorno local
    try:
        plt.show()
    except Exception:
        pass
