# Siamese Cosmic Deceleration: Analysis Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17694936.svg)](https://zenodo.org/records/17694936)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Replication Data & Code for: "Transient Cosmic Acceleration from Phase Interference: A Siamese Cosmology Confronted with Son & Lee (2025)"**

---

## 🔭 Project Overview

Recent observational analyses combining BAO, CMB, and Type Ia supernovae with progenitor age-bias corrections (**Son et al., 2025**) suggest that the Universe may be exiting its phase of cosmic acceleration, favoring a present-day deceleration parameter ($q_0 > 0$).

This repository contains the complete Python analysis pipeline to test the **"Siamese Interference" framework**—a CPT-symmetric cosmological model—against these new observational constraints. We demonstrate that cosmic deceleration emerges naturally from phase interference, predicting a falsifiable hemispheric anisotropy for upcoming surveys like Euclid and LSST.

---

## 📊 Key Results

### 1. Theoretical Fit against 2025 Constraints
![Siamese Overlay Plot](figures/phase_tracks_SonLee_overlay.jpg)
> **Solving the Tension:** Overlay of Siamese Interference trajectories on the confidence contours from Son et al. (2025). The red regions indicate the parameter space favored by combined BAO + CMB + SN data (age-corrected). The blue tracks show that the Siamese model naturally predicts these values (around $w_0 \approx -0.34$, $w_a \approx -1.9$), passing through the 68% confidence region where the standard $\Lambda$CDM model fails.

### 2. Statistical Significance of the Hemispheric Signal
![Monte Carlo Histogram](figures/hemispheric_histogram_C.png)
> **Falsifiability:** Results from a Monte Carlo simulation with 100,000 synthetic galaxies. The blue histogram represents the background noise derived from 2,000 random sky orientations. The vertical red line marks the specific signal predicted by the Siamese Interference model ($Z \approx 2.36$), confirming that the predicted anisotropy is distinct from random noise.

### 3. Transient Acceleration Signature
![Expansion History H(z)](figures/h_of_z_turnover.png)
> **Dynamics:** Comparison of the normalized expansion rate $H(z)/(1+z)$. The deeper minimum in the Siamese curve at $z \approx 0.76$ signals a stronger but transient acceleration phase, distinguishing it from the eternal acceleration predicted by a cosmological constant.

---

## 🛠️ Repository Structure

| Directory/File | Description |
| :--- | :--- |
| **`scripts/`** | Python source code for analysis and plotting. |
| `reconstruct_sonlee_contours.py` | **Constraint Generator.** Reconstructs the 68% and 95% confidence ellipses using the exact parameters from Son & Lee (2025, Table 2). |
| `run_hemispheric_test_C.py` | **Anisotropy Test.** Monte Carlo simulation testing the detectability of the Siamese hemispheric signal. |
| **`data/`** | Input datasets. |
| `SonLee_contours.csv` | Reconstructed observational constraint data used for plotting. |
| **`results/`** | Numerical outputs. |
| `hemispheric_test_C_summary.json` | Statistical metrics (Z-score, p-value) from the hemispheric simulation. |

---

## 🚀 How to Run

To reproduce the results and figures, follow this execution order:

### 1. Setup Environment
```bash
pip install -r requirements.txt

# Step 1: Generate Observational Constraints (Creates data/SonLee_contours.csv)
python scripts/reconstruct_sonlee_contours.py

# Step 2: Run the Hemispheric Simulation (Generates Figure 1)
# Note: Monte Carlo simulation for N=100k may take a few minutes.
python scripts/run_hemispheric_test_C.py

# Step 3: Generate Theoretical Plots (Generates Figures 2, 3, 4)
python scripts/calculate_phase_tracks_and_overlay.py
python scripts/calculate_expansion_turnover.py
python scripts/calculate_present_day_deceleration.py

@dataset{siamese_deceleration_2025,
  author       = {CosmicThinker},
  title        = {Replication Data & Code for: "Transient Cosmic Acceleration from Phase Interference: A Siamese Cosmology Confronted with Son & Lee (2025)"},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17694936},
  url          = {[https://zenodo.org/records/17694936](https://zenodo.org/records/17694936)}
}
