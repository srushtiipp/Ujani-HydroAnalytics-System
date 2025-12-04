# anomaly_detection.py
# Requirements: pandas, numpy, scikit-learn, matplotlib
# Install if needed:
# pip install pandas numpy scikit-learn matplotlib

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os

# ----------------- SETTINGS -----------------
INPUT = "Data csv.csv"   # your cleaned rainfall file
DATE_COL = "Date"
STATION_COLS = ["Alandi", "Budhwad (Velholi)", "Koliye", "Paud"]  # adjust if needed
OUT_DIR = "anomaly_output"
os.makedirs(OUT_DIR, exist_ok=True)
# ---------------------------------------------

# Load data
df = pd.read_csv(INPUT)
df.columns = [c.strip() for c in df.columns]

# Parse date
df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
df = df.sort_values(DATE_COL).reset_index(drop=True)

# Convert station rainfall to numeric
for c in STATION_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Basin total
df["basin_total"] = df[STATION_COLS].sum(axis=1)

# -------------------------------
# 1. Z-SCORE ANOMALY DETECTION
# -------------------------------
mean_val = df["basin_total"].mean()
std_val  = df["basin_total"].std()

df["z_score"] = (df["basin_total"] - mean_val) / std_val
df["z_flag"] = (df["z_score"].abs() > 2.5).astype(int)  # threshold = 2.5 or 3

# -------------------------------
# 2. ISOLATION FOREST ANOMALY
# -------------------------------
model = IsolationForest(
    contamination=0.03,   # ~3% anomalies
    random_state=42
)

iso_values = df[["basin_total"]].values
model.fit(iso_values)

df["iso_score"] = model.decision_function(iso_values)
df["iso_flag"] = model.predict(iso_values)

# IsolationForest outputs:
# -1 = anomaly, 1 = normal
df["iso_flag"] = df["iso_flag"].replace({1: 0, -1: 1})

# -------------------------------
# Save output CSV
# -------------------------------
df_out = df[[DATE_COL] + STATION_COLS + ["basin_total", "z_score", "z_flag", "iso_score", "iso_flag"]]
df_out.to_csv(os.path.join(OUT_DIR, "anomalies.csv"), index=False)

print("Saved anomalies.csv in:", OUT_DIR)

# -------------------------------
# Plot anomalies (Z-score)
# -------------------------------
plt.figure(figsize=(12,5))
plt.plot(df[DATE_COL], df["basin_total"], label="Rainfall")
plt.scatter(
    df[df["z_flag"] == 1][DATE_COL],
    df[df["z_flag"] == 1]["basin_total"],
    color="red",
    label="Z-Score Anomaly",
    s=25
)
plt.title("Rainfall Anomalies (Z-Score)")
plt.ylabel("Basin Total Rainfall (mm)")
plt.xlabel("Date")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "anomaly_zscore.png"), dpi=150)
plt.close()

# -------------------------------
# Plot anomalies (Isolation Forest)
# -------------------------------
plt.figure(figsize=(12,5))
plt.plot(df[DATE_COL], df["basin_total"], label="Rainfall")
plt.scatter(
    df[df["iso_flag"] == 1][DATE_COL],
    df[df["iso_flag"] == 1]["basin_total"],
    color="orange",
    label="Isolation Forest Anomaly",
    s=25
)
plt.title("Rainfall Anomalies (Isolation Forest)")
plt.ylabel("Basin Total Rainfall (mm)")
plt.xlabel("Date")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "anomaly_isoforest.png"), dpi=150)
plt.close()

print("All plots saved in:", OUT_DIR)
