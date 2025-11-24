import numpy as np
import json

# --------------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------------
N_SNe = 5000
axis_RA = np.deg2rad(170.0)   # eje siamés
axis_DEC = np.deg2rad(40.0)
anisotropy_strength = 0.05    # Δq0 = 5%

# Salida
output_path = "../results/hemispheric_test_output.json"

# --------------------------------------------------------
# Función util para ángulo entre direcciones en esfera
# --------------------------------------------------------
def angular_separation(ra1, dec1, ra2, dec2):
    return np.arccos(
        np.sin(dec1) * np.sin(dec2) +
        np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )

# --------------------------------------------------------
# 1) Generar catálogo sintético de supernovas
# --------------------------------------------------------
# Distribución uniforme en el cielo
RA = np.random.uniform(0, 2*np.pi, N_SNe)
DEC = np.arcsin(np.random.uniform(-1, 1, N_SNe))  # distribución uniforme en esfera
Z = np.random.uniform(0.01, 1.2, N_SNe)           # redshift cosmológico típico

# --------------------------------------------------------
# 2) Modelo base para q0 sintético (arbitrario pero realista)
# --------------------------------------------------------
q0_fid = -0.55   # desaceleración ligera como en modelos dinámicos

q0_values = np.full(N_SNe, q0_fid)

# --------------------------------------------------------
# 3) Determinar hemisferios con respecto al eje Siamés
# --------------------------------------------------------
ANGLE = angular_separation(RA, DEC, axis_RA, axis_DEC)

mask_siamese = ANGLE < np.pi/2
mask_antipodal = ~mask_siamese

# --------------------------------------------------------
# 4) Aplicar anisotropía sintética de 5%
# --------------------------------------------------------
q0_values[mask_siamese] += anisotropy_strength / 2.0
q0_values[mask_antipodal] -= anisotropy_strength / 2.0

# --------------------------------------------------------
# 5) Calcular q0 efectivo por hemisferio
# --------------------------------------------------------
q0_siamese = float(np.mean(q0_values[mask_siamese]))
q0_antipodal = float(np.mean(q0_values[mask_antipodal]))
delta_q0 = float(q0_siamese - q0_antipodal)

# --------------------------------------------------------
# 6) Guardar resultados para el histograma Monte Carlo
# --------------------------------------------------------
results = {
    "N_SNe": N_SNe,
    "axis_RA_deg": 170.0,
    "axis_DEC_deg": 40.0,
    "anisotropy": anisotropy_strength,
    "q0_siamese": q0_siamese,
    "q0_antipodal": q0_antipodal,
    "delta_q0": delta_q0
}

with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print("[OK] Test sintético completado")
print(f"[OK] Resultados guardados en: {output_path}")
