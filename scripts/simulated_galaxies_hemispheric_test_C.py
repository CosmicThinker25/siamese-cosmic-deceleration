#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulated Galaxies Hemispheric Test (Option C)
---------------------------------------------
Genera un catálogo sintético de galaxias/SNe y calcula Δq0
entre hemisferios respecto al eje siamés y 2000 ejes aleatorios.

Salidas:
- ../figures/hemispheric_histogram_C.png
- ../results/hemispheric_test_C_summary.json
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# -----------------------------
# Parámetros de la simulación
# -----------------------------
N_GAL = 100_000          # número de galaxias simuladas
Z_MAX = 1.5              # z máximo
Z0 = 0.9                 # escala de la distribución p(z) ~ z^2 exp(-z/Z0)
Q0_FID = -0.23           # valor isotrópico de fondo
DELTA_Q0_ANISO = 0.05    # amplitud de anisotropía hemisférica total
SIGMA_Q = 0.03           # ruido gaussiano por galaxia
Z_ANISO_MAX = 0.7        # sólo hay anisotropía para z <= Z_ANISO_MAX

N_ROT = 2000             # número de ejes aleatorios para Monte Carlo

# Eje siamés en grados
RA_AXIS_DEG = 170.0
DEC_AXIS_DEG = 40.0

# Carpetas de salida
BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "figures"
RES_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


# -----------------------------
# Utilidades geométricas
# -----------------------------
def radec_to_unitvec(ra_deg, dec_deg):
    """Convierte RA,Dec (grados) en vector unitario cartesiano."""
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.vstack((x, y, z)).T  # shape (N,3)


def random_radec(n):
    """RA,Dec isotrópicos en la esfera."""
    ra = np.random.uniform(0.0, 360.0, size=n)
    u = np.random.uniform(-1.0, 1.0, size=n)
    dec = np.rad2deg(np.arcsin(u))
    return ra, dec


def random_unit_vectors(n):
    """Genera n vectores unitarios isotrópicos."""
    phi = np.random.uniform(0.0, 2.0 * np.pi, size=n)
    u = np.random.uniform(-1.0, 1.0, size=n)
    theta = np.arccos(u)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.vstack((x, y, z)).T


# -----------------------------
# Distribución de redshift
# -----------------------------
def sample_redshifts(n, z_max=Z_MAX, z0=Z0):
    """
    Muestra redshifts según p(z) ~ z^2 exp(-z/z0) truncado en [0, z_max].
    Usamos muestreo por rechazo simple.
    """
    z = []
    n_target = n
    # cota superior aproximada de la pdf (en torno a z ~ 2 z0, pero truncamos)
    f_max = (z_max**2) * np.exp(-z_max / z0)
    while len(z) < n_target:
        z_trial = np.random.uniform(0.0, z_max, size=n_target)
        u = np.random.uniform(0.0, f_max, size=n_target)
        f = (z_trial**2) * np.exp(-z_trial / z0)
        accept = u < f
        z.extend(z_trial[accept])
    z = np.array(z[:n_target])
    return z


# -----------------------------
# Construcción del catálogo
# -----------------------------
def build_mock_catalog():
    """
    Genera:
    - RA, Dec isotrópicos
    - z con distribución cosmológica simple
    - q_obs por galaxia, con anisotropía hemisférica a bajo z
    """
    print("[INFO] Generando catálogo sintético de galaxias...")
    ra, dec = random_radec(N_GAL)
    z = sample_redshifts(N_GAL)

    # Eje siamés unitario
    axis_vec = radec_to_unitvec(RA_AXIS_DEG, DEC_AXIS_DEG)[0]
    gal_vecs = radec_to_unitvec(ra, dec)

    # Hemisferio según eje siamés: dot > 0 => Norte, dot < 0 => Sur
    dots = np.dot(gal_vecs, axis_vec)
    hemi_sign = np.sign(dots)  # +1 Norte, -1 Sur, 0 casi nunca

    # Perfil de anisotropía: sólo para z <= Z_ANISO_MAX
    aniso_mask = z <= Z_ANISO_MAX
    aniso_factor = np.zeros_like(z)
    aniso_factor[aniso_mask] = 1.0

    # Asignamos q_obs por galaxia
    # q_obs = q0_fid + (DELTA_Q0_ANISO/2)*hemi_sign*aniso_factor + ruido
    q_true = Q0_FID + 0.5 * DELTA_Q0_ANISO * hemi_sign * aniso_factor
    q_obs = q_true + np.random.normal(loc=0.0, scale=SIGMA_Q, size=N_GAL)

    return ra, dec, z, q_obs, axis_vec


# -----------------------------
# Cálculo de Δq0 para un eje dado
# -----------------------------
def delta_q0_for_axis(gal_vecs, q_obs, axis_vec):
    dots = np.dot(gal_vecs, axis_vec)
    north = dots >= 0.0
    south = dots < 0.0

    q_north = np.mean(q_obs[north])
    q_south = np.mean(q_obs[south])
    return q_north - q_south


# -----------------------------
# Experimento hemisférico C
# -----------------------------
def run_hemispheric_test_C():
    # Construir catálogo
    ra, dec, z, q_obs, siamese_axis_vec = build_mock_catalog()
    gal_vecs = radec_to_unitvec(ra, dec)

    print("[INFO] Calculando Δq0 para el eje siamés...")
    delta_q_siamese = delta_q0_for_axis(gal_vecs, q_obs, siamese_axis_vec)

    print("[INFO] Ejecutando Monte Carlo con ejes aleatorios...")
    deltas_random = []
    n_print = max(1, N_ROT // 10)

    for i in range(N_ROT):
        axis_rand = random_unit_vectors(1)[0]
        dq = delta_q0_for_axis(gal_vecs, q_obs, axis_rand)
        deltas_random.append(dq)

        if (i + 1) % n_print == 0:
            print(f"  - Rotación {i+1}/{N_ROT}")

    deltas_random = np.array(deltas_random)

    mu = np.mean(deltas_random)
    sigma = np.std(deltas_random, ddof=1)

    z_score = (delta_q_siamese - mu) / sigma
    # p-value unilaterial (cola derecha)
    p_value = np.mean(deltas_random >= delta_q_siamese)

    print("\n[RESULTADOS TEST C]")
    print(f"  media ruido        = {mu:.5f}")
    print(f"  sigma ruido        = {sigma:.5f}")
    print(f"  Δq0 eje siamés     = {delta_q_siamese:.5f}")
    print(f"  z-score            = {z_score:.2f}")
    print(f"  p-value (cola der) = {p_value:.5f}")

    # Guardar resumen
    summary = {
        "N_GAL": int(N_GAL),
        "Z_MAX": float(Z_MAX),
        "Z0": float(Z0),
        "Q0_FID": float(Q0_FID),
        "DELTA_Q0_ANISO": float(DELTA_Q0_ANISO),
        "SIGMA_Q": float(SIGMA_Q),
        "Z_ANISO_MAX": float(Z_ANISO_MAX),
        "N_ROT": int(N_ROT),
        "mean_noise": float(mu),
        "sigma_noise": float(sigma),
        "delta_q0_siamese": float(delta_q_siamese),
        "z_score": float(z_score),
        "p_value_right_tail": float(p_value),
    }

    out_json = RES_DIR / "hemispheric_test_C_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"[OK] Resumen guardado en: {out_json}")

    # Histograma
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(deltas_random, bins=50, alpha=0.7, edgecolor="k")
    ax.axvline(delta_q_siamese, color="r", linewidth=2)
    ax.set_xlabel(r"$\Delta q_0$ (North - South)")
    ax.set_ylabel("Counts")
    ax.set_title("Monte Carlo Hemispheric Rotation Test (Simulated Galaxies, Test C)")
    txt = rf"Siamese axis $\rightarrow \Delta q_0 = {delta_q_siamese:.3f}$"
    ax.text(0.98, 0.95, txt,
            transform=ax.transAxes,
            ha="right", va="top", color="r", fontsize=11)

    fig.tight_layout()
    out_fig = FIG_DIR / "hemispheric_histogram_C.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"[OK] Histograma generado → {out_fig}")


if __name__ == "__main__":
    run_hemispheric_test_C()
