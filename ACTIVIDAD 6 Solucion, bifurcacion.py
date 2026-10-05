## Andres Felipe Rojas Tafur##
## 070400372024 ##
import os
import time
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

def rk4_step_vec(f, t, y_vec, h):
    k1 = h * f(t, y_vec)
    k2 = h * f(t + 0.5 * h, y_vec + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y_vec + 0.5 * k2)
    k4 = h * f(t + h, y_vec + k3)
    return y_vec + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

def generar_bifurcacion_actividad6():
    print("=====================================================================")
    print("UNIVERSIDAD DEL TOLIMA - FÍSICA COMPUTACIONAL II")
    print("Actividad 6: Diagrama de Bifurcación II (Péndulo Forzado Paramétricamente)")
    print("Estudiante: Andrés Felipe Rojas Tafur | Código: 070400372024")
    print("Profesor: Dr. César Andrés Morales Rodríguez")
    print("=====================================================================\n")

    # Parámetros físicos según la guía
    alpha = 0.1       # Coeficiente de amortiguamiento
    omega0 = 1.0      # Frecuencia natural propia del oscilador
    omega = 2.0       # Frecuencia del forzamiento externo
    T = 2.0 * np.pi / omega  # Período de forzamiento T = pi s

    # Rango del parámetro de bifurcación gamma: 0 <= gamma <= 2.25
    gamma_min = 0.0
    gamma_max = 2.25
    n_gammas = 800
    gamma_vals = np.linspace(gamma_min, gamma_max, n_gammas)

    # Condición inicial dada en el taller: (theta, theta_dot) = (1, 1)
    theta0 = np.full(n_gammas, 1.0)
    w0 = np.full(n_gammas, 1.0)
    y = np.concatenate([theta0, w0])

    # Ecuación diferencial: d2theta/dt2 + alpha*dtheta/dt + omega0^2*sin(theta) = gamma*cos(omega*t)*sin(theta)
    def dyn(t, y_vec):
        th = y_vec[:n_gammas]
        w = y_vec[n_gammas:]
        dth = w
        dw = - alpha * w - (omega0**2) * np.sin(th) + gamma_vals * np.cos(omega * t) * np.sin(th)
        return np.concatenate([dth, dw])

    # Parámetros de integración numéricos
    Trans = 300         # Períodos transitorios a descartar para limpiar el estado inicial
    Nkeep = 150         # Períodos estroboscópicos a registrar en el diagrama
    steps_per_T = 160   # Pasos por período T
    dt = T / steps_per_T
    total_steps = (Trans + Nkeep) * steps_per_T

    print(f"Calculando diagrama de bifurcación para {n_gammas} valores de gamma en [0, 2.25]...")
    t0 = time.time()

    abs_w_strobe = np.empty((Nkeep, n_gammas))

    save_idx = 0
    for step in range(total_steps):
        t_curr = step * dt
        y = rk4_step_vec(dyn, t_curr, y, dt)
        comp_period = (step + 1) // steps_per_T
        if (step + 1) % steps_per_T == 0 and comp_period > Trans:
            w_curr = y[n_gammas:]
            abs_w_strobe[save_idx] = np.abs(w_curr)
            save_idx += 1

    t1 = time.time()
    print(f"¡Simulación completada con éxito en {t1 - t0:.2f} segundos!\n")

    # Graficación en ventana adaptada a pantalla
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Trazar puntos del diagrama de bifurcación
    for i in range(n_gammas):
        ax.scatter(np.full(Nkeep, gamma_vals[i]), abs_w_strobe[:, i], s=0.15, color='darkblue', alpha=0.6, rasterized=True)

    ax.set_xlim(0.0, 2.25)
    ax.set_ylim(0.0, 3.5)
    ax.set_xlabel(r"$\gamma$ (Amplitud de forzamiento)", fontsize=13)
    ax.set_ylabel(r"$|\dot{\theta}|$ (Velocidad angular estroboscópica)", fontsize=13)
    
    ax.tick_params(axis='both', which='major', labelsize=11)

    ax.set_title(r"Actividad 6: Diagrama de Bifurcación para $|\dot{\theta}|$ vs $\gamma$" + "\n" +
                 r"Transición al Caos en el Oscilador Paramétrico (Landau et al. 2007, Fig. 1)", 
                 fontsize=11, fontweight='bold', pad=12)

    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    outfile = "actividad6_bifurcacion.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"¡Gráfica guardada exitosamente en alta resolución como '{outfile}'!")

    return fig

if __name__ == "__main__":
    generar_bifurcacion_actividad6()
    try:
        plt.show()
    except Exception:
        pass
