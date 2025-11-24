#!/usr/bin/env python
"""
Script 5: Phase Tracks + Son & Lee CPL Contours

Lee los contornos reconstruidos de Son & Lee (2025) en el plano (w0, wa)
y superpone las trayectorias "Siamese phase tracks" para distintos valores
de la velocidad de fase gamma.

Entrada:
    ../data/SonLee_contours.csv

Salidas:
    ../figures/phase_tracks_SonLee_overlay.png
    ../results/phase_tracks_SonLee_overlay_summary.json
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Paths
# ----------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "data", "SonLee_contours.csv")
fig_path = os.path.join(base_dir, "..", "figures", "phase_tracks_SonLee_overlay.png")
summary_path = os.path.join(base_dir, "..", "results", "phase_tracks_SonLee_overlay_summary.json")

os.makedirs(os.path.dirname(fig_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_path), exist_ok=True)

# ----------------------------------------------------------------------
# 2. Leer contornos de Son & Lee
# ----------------------------------------------------------------------
df = pd.read_csv(data_path)

cont_68 = df[df["level"] == "68%"]
cont_95 = df[df["level"] == "95%"]

# Medias aproximadas (para información en el JSON)
w0_mean = df["w0"].mean()
wa_mean = df["wa"].mean()

# ----------------------------------------------------------------------
# 3. Definir el mapeo Siamés (toy, pero coherente con el paper)
# ----------------------------------------------------------------------
def siamese_cpl_from_phase(delta_phi, gamma, delta_phi_max=np.pi * 0.42):
    """
    Mapea (delta_phi, gamma) -> (w0, wa) efectivos tipo CPL.

    delta_phi: fase actual (rad)
    gamma: velocidad de fase (adimensional)
    delta_phi_max: fase máxima considerada (~0.42*pi)

    Retorna:
        w0, wa
    """
    # Fase normalizada [0, 1]
    x = np.clip(delta_phi / delta_phi_max, 0.0, 1.0)

    # w0 pasa de -1 (Lambda) a ~ -0.2 cuando x -> 1
    w0 = -1.0 + 0.8 * x**2

    # wa se hace muy negativo a medida que crece x y con gamma
    # Ajustado para que exista un punto cercano a (w0, wa) = (-0.7, -1.8)
    # para gamma ~ 1.5
    A = 2.23  # factor de escala empírico
    wa = -0.2 - A * gamma * x**1.5

    return w0, wa

# ----------------------------------------------------------------------
# 4. Generar trayectorias de fase
# ----------------------------------------------------------------------
gamma_values = [1.0, 1.5, 2.0]
delta_phi_max = np.pi * 0.42
delta_phi_grid = np.linspace(0.0, delta_phi_max, 300)

tracks = {}

for gamma in gamma_values:
    w0_list = []
    wa_list = []
    for dphi in delta_phi_grid:
        w0_val, wa_val = siamese_cpl_from_phase(dphi, gamma, delta_phi_max)
        w0_list.append(w0_val)
        wa_list.append(wa_val)
    tracks[gamma] = {
        "delta_phi": delta_phi_grid.tolist(),
        "w0": w0_list,
        "wa": wa_list,
    }

# Punto de referencia usado en el paper (aprox)
w0_ref = -0.7
wa_ref = -1.8

# ----------------------------------------------------------------------
# 5. Figura
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 7))

# Contornos Son & Lee
plt.fill(cont_95["w0"], cont_95["wa"],
         alpha=0.15, color="red", label="Son & Lee 95% (DES5Y corrected)")
plt.fill(cont_68["w0"], cont_68["wa"],
         alpha=0.30, color="red", label="Son & Lee 68%")

# Trayectorias Siamés
colors = {1.0: "tab:blue", 1.5: "tab:orange", 2.0: "tab:purple"}

for gamma in gamma_values:
    w0_arr = np.array(tracks[gamma]["w0"])
    wa_arr = np.array(tracks[gamma]["wa"])
    plt.plot(w0_arr, wa_arr, color=colors[gamma],
             label=f"Siamese phase track (gamma = {gamma:.1f})")

# Punto de referencia
plt.scatter(w0_ref, wa_ref, marker="*", s=120, color="black",
            label="Reference Siamese point")

# Formato del gráfico
plt.xlabel(r"$w_0$")
plt.ylabel(r"$w_a$")
plt.title("Siamese Phase Tracks vs. Son & Lee (2025) CPL Constraints")

plt.xlim(-1.1, -0.2)
plt.ylim(-2.6, -1.2)

plt.grid(alpha=0.3)
plt.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig(fig_path, dpi=300)
plt.close()

# ----------------------------------------------------------------------
# 6. Guardar resumen
# ----------------------------------------------------------------------
summary = {
    "input_contours_csv": os.path.relpath(data_path, start=os.path.dirname(summary_path)),
    "gamma_values": gamma_values,
    "delta_phi_max_rad": delta_phi_max,
    "w0_axis_limits": [-1.1, -0.2],
    "wa_axis_limits": [-2.6, -1.2],
    "SonLee_mean_w0": float(w0_mean),
    "SonLee_mean_wa": float(wa_mean),
    "reference_siamese_point": {"w0": w0_ref, "wa": wa_ref},
    "figure_path": os.path.relpath(fig_path, start=os.path.dirname(summary_path)),
}

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("[OK] Figura generada ->", fig_path)
print("[OK] Resumen guardado ->", summary_path)
