import numpy as np
import pandas as pd
import os

# --- PARÁMETROS EXACTOS DE SON & LEE (2025) ---
# Tabla 2, fila: BAO+CMB+DES5Y (corrected)
MU_W0 = -0.337
MU_WA = -1.902
SIGMA_W0 = 0.062
SIGMA_WA = 0.246
RHO = -0.90  # Correlación estimada para CPL estándar

# Definir la matriz de covarianza
cov_00 = SIGMA_W0**2
cov_aa = SIGMA_WA**2
cov_0a = RHO * SIGMA_W0 * SIGMA_WA
cov = np.array([[cov_00, cov_0a], [cov_0a, cov_aa]])

def get_ellipse_points(mu, cov, delta_chi2, num_points=100):
    """Genera puntos (w0, wa) para una elipse de confianza dada."""
    # Radio en espacio de Mahalanobis
    # Para 2 grados de libertad (w0, wa):
    # 68.3% (1 sigma) -> Delta Chi2 = 2.30
    # 95.4% (2 sigma) -> Delta Chi2 = 6.17
    radius = np.sqrt(delta_chi2)
    
    angles = np.linspace(0, 2*np.pi, num_points)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
    
    # Transformación usando Cholesky
    L = np.linalg.cholesky(cov)
    ellipse = circle @ L.T + mu
    return ellipse

# --- GENERACIÓN DE DATOS ---
print("Generando contornos...")

# 1 Sigma (68%) y 2 Sigma (95%)
pts_68 = get_ellipse_points(np.array([MU_W0, MU_WA]), cov, delta_chi2=2.30, num_points=100)
pts_95 = get_ellipse_points(np.array([MU_W0, MU_WA]), cov, delta_chi2=6.17, num_points=100)

# Crear DataFrame
df_68 = pd.DataFrame(pts_68, columns=['w0', 'wa'])
df_68['level'] = '68%'

df_95 = pd.DataFrame(pts_95, columns=['w0', 'wa'])
df_95['level'] = '95%'

df_final = pd.concat([df_68, df_95])

# --- GUARDADO ---
# Intentar guardar en subcarpeta 'data' si existe, si no, en local
filename = "SonLee_contours.csv"
if os.path.exists("data"):
    filepath = os.path.join("data", filename)
else:
    filepath = filename

try:
    df_final.to_csv(filepath, index=False)
    print(f"¡ÉXITO! Archivo guardado correctamente en: {filepath}")
    print("Primeras filas:")
    print(df_final.head())
except Exception as e:
    print(f"Error al guardar: {e}")
    # Fallback por si acaso
    df_final.to_csv("contornos_simple.csv", index=False)
    print("Se guardó como 'contornos_simple.csv' por seguridad.")