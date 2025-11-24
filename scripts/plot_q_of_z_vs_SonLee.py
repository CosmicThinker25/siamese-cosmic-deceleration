import numpy as np
import matplotlib.pyplot as plt
import json

def w_CPL(a, w0, wa):
    return w0 + wa * (1 - a)

def q_from_w(a, w):
    # q(a) = 1/2 * (1 + 3w * Ω_DE(a))
    # Para visualización rápida usamos Ω_DE(a) ≈ 0.7 a^{-3(1+w_eff)}
    w_eff = w
    Omega_DE = 0.7 * a**(-3 * (1 + w_eff))
    Omega_DE /= (0.3 * a**(-3) + Omega_DE)
    return 0.5 * (1 + 3 * w_eff * Omega_DE)


# Redshift grid
z = np.linspace(0, 2, 400)
a = 1 / (1 + z)

# LCDM (w = -1)
w_lcdm = -1.0
q_LCDM = q_from_w(a, w_lcdm)

# Siamese parameters (best match from Phase Tracks)
w0_s = -0.7
wa_s = -1.8
w_siamese = w_CPL(a, w0_s, wa_s)
q_siamese = q_from_w(a, w_siamese)

plt.figure(figsize=(9,6))
plt.plot(z, q_LCDM, color="black", linewidth=2, label=r"ΛCDM ($w=-1$)")
plt.plot(z, q_siamese, color="red", linewidth=2, label=fr"Siamese ($w_0={w0_s}, w_a={wa_s}$)")

# Observational region — Son & Lee (deceleration at z = 0)
q0_min = 0.02   # intervalo aproximado recuperado de Son & Lee
q0_max = 0.10
plt.axhspan(q0_min, q0_max, color="lightgray", alpha=0.5,
            label="Son & Lee 2025 — deceleration interval")

plt.axvline(0, linestyle="--", color="gray", linewidth=1)
plt.xlabel("Redshift z")
plt.ylabel("Deceleration parameter  q(z)")
plt.title("Present-Day Deceleration: q(z) Comparison")
plt.legend()
plt.grid(alpha=0.3)

output_figure = "../figures/q_of_z_vs_SonLee.png"
plt.savefig(output_figure, dpi=300)
print(f"[OK] Figura guardada en {output_figure}")

# Save summary
summary = {
    "LCDM_q0": float(q_LCDM[z==z[0]] if False else q_LCDM[0]),
    "Siamese_q0": float(q_siamese[0]),
    "SonLee_interval_q0": [q0_min, q0_max]
}
output_summary = "../results/q_of_z_vs_SonLee_summary.json"
json.dump(summary, open(output_summary, "w"), indent=4)
print(f"[OK] Resumen guardado en {output_summary}")

