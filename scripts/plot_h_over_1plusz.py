import numpy as np
import matplotlib.pyplot as plt
import json

# -------------------------------------------------------
# CONFIGURACIÓN DEL MODELO
# -------------------------------------------------------
Om = 0.3
Ol = 0.7
H0 = 70.0

# Modelos CPL
models = {
    "LCDM": {"w0": -1.0, "wa": 0.0, "color": "black"},
    "Siamese": {"w0": -0.7, "wa": -1.8, "color": "crimson"}
}

z = np.linspace(0, 2.0, 400)

def E_of_z(z, w0, wa):
    # CPL evolution for dark energy
    w = w0 + wa * (z / (1 + z))
    de_factor = (1 + z) ** (3 * (1 + w))
    matter_factor = Om * (1 + z) ** 3
    return np.sqrt(matter_factor + Ol * de_factor)

# -------------------------------------------------------
# GENERAR CURVAS
# -------------------------------------------------------
plt.figure(figsize=(8, 6))

summary = {}

for label, params in models.items():
    w0, wa = params["w0"], params["wa"]
    E = E_of_z(z, w0, wa)
    H_over = E / (1 + z)

    plt.plot(z, H_over, label=f"{label} (w0={w0}, wa={wa})",
             lw=2.5, color=params["color"])
    
    summary[label] = {
        "w0": w0,
        "wa": wa,
        "H_over_min_z": float(z[np.argmin(H_over)]),
        "H_over_min_value": float(np.min(H_over))
    }

# -------------------------------------------------------
# FORMATO DE LA FIGURA
# -------------------------------------------------------
plt.title(r"Expansion Turnover: $H(z)/(1+z)$ vs. $z$")
plt.xlabel("Redshift z")
plt.ylabel(r"$H(z)/(1+z)$  (normalized)")
plt.grid(alpha=0.3)
plt.legend()
plt.ylim(0.4, 1.2)

output_fig = "../figures/h_of_z_turnover.png"
output_json = "../results/h_of_z_turnover_summary.json"

plt.savefig(output_fig, dpi=300, bbox_inches="tight")
plt.close()

with open(output_json, "w") as f:
    json.dump(summary, f, indent=4)

print("[OK] Figura guardada en", output_fig)
print("[OK] Resumen guardado en", output_json)
for k,v in summary.items():
    print(f"{k}: min at z={v['H_over_min_z']:.2f}")
