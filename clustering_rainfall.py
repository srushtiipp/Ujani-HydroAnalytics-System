# clustering_rainfall.py
# Requirements: pandas, numpy, scikit-learn, matplotlib, seaborn
# Install if needed: pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------- USER SETTINGS ----------
INPUT_CSV = "Data csv.csv"   # path to your data
DATE_COL = "Date "
STATION_COLS = ["Alandi", "Budhwad (Velholi)", "Koliye", "Paud"]  # change if your columns are named differently
OUT_DIR = "clustering_output"
K = 3                        # default clusters
MAX_LAG = 7                  # how many lag days to compute
ROLL_WINDOWS = [3, 7, 14]    # rolling sums to compute
# ------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# 1) Load
df = pd.read_csv(INPUT_CSV)
# parse date
df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors='coerce')
df = df.sort_values(DATE_COL).reset_index(drop=True)

# 2) Basic cleaning: ensure station cols numeric
for c in STATION_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

# 3) Feature engineering
# 3a) daily basin total
df["basin_total"] = df[STATION_COLS].sum(axis=1)

# 3b) lags for basin_total (to capture storm movement/delay)
for lag in range(1, MAX_LAG+1):
    df[f"basin_total_lag_{lag}"] = df["basin_total"].shift(lag).fillna(0.0)

# 3c) rolling sums per station and basin-wide
for w in ROLL_WINDOWS:
    df[f"basin_roll_{w}"] = df["basin_total"].rolling(window=w, min_periods=1).sum().fillna(0.0)
    for s in STATION_COLS:
        df[f"{s}_roll_{w}"] = df[s].rolling(window=w, min_periods=1).sum().fillna(0.0)

# 3d) temporal features
df["dayofyear"] = df[DATE_COL].dt.dayofyear
df["month"] = df[DATE_COL].dt.month

# Choose features for clustering (you can customize)
feature_cols = [
    "basin_total",
    # lags
] + [f"basin_total_lag_{lag}" for lag in range(1, MAX_LAG+1)] \
  + [f"basin_roll_{w}" for w in ROLL_WINDOWS] \
  + [f"{s}_roll_{w}" for s in STATION_COLS for w in ROLL_WINDOWS] \
  + ["dayofyear", "month"]

# Make sure all chosen features exist and fill na
feature_cols = [c for c in feature_cols if c in df.columns]
X = df[feature_cols].fillna(0.0).values

# 4) scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5) Find good K (elbow + silhouette) — quick scan
inertia = []
sil_scores = []
k_range = range(2, 7)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# Plot elbow + silhouette
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(list(k_range), inertia, marker='o')
plt.title("Elbow (Inertia)")
plt.xlabel("k")
plt.subplot(1,2,2)
plt.plot(list(k_range), sil_scores, marker='o')
plt.title("Silhouette Score")
plt.xlabel("k")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "elbow_silhouette.png"), dpi=150)
plt.close()

# 6) Fit final KMeans (default K = 3)
k = K
km = KMeans(n_clusters=k, random_state=42, n_init=20)
labels = km.fit_predict(X_scaled)
df["cluster"] = labels

# 7) Map clusters to human-readable regimes by sorting centroids by basin_total contribution
# compute centroid in original feature space by inverse-scaling centroid of KMeans
centroids_scaled = km.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)
# find index of basin_total in feature_cols
if "basin_total" in feature_cols:
    bt_idx = feature_cols.index("basin_total")
    centroid_basin_totals = centroids[:, bt_idx]
    # sort clusters by centroid basin_total value
    order = np.argsort(centroid_basin_totals)  # low -> high
    regime_names = ["Low", "Normal", "High"]
    # if K not equal to 3, create generic names
    if k == 3:
        mapping = { int(order[i]) : regime_names[i] for i in range(3) }
    else:
        mapping = { int(order[i]) : f"Cluster_{i+1}" for i in range(k) }
else:
    # fallback mapping by sum of centroid features
    centroid_sums = centroids.sum(axis=1)
    order = np.argsort(centroid_sums)
    mapping = { int(order[i]) : f"Cluster_{i+1}" for i in range(k) }

df["regime"] = df["cluster"].map(mapping)

# 8) PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df["pc1"] = X_pca[:,0]
df["pc2"] = X_pca[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(x="pc1", y="pc2", hue="regime", data=df, palette="deep", s=30)
plt.title("Clusters (PCA 2D)")
plt.legend(title="Regime")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cluster_pca.png"), dpi=150)
plt.close()

# 9) Time series plot coloured by cluster (last 365 days if long)
plot_df = df.copy()
if len(plot_df) > 800:
    plot_df = plot_df.tail(800)  # limit plotting to recent portion for clarity

plt.figure(figsize=(14,5))
sns.lineplot(x=DATE_COL, y="basin_total", data=plot_df, label="basin_total")
# color background by regime blocks (optional)
for regime_val, grp in plot_df.groupby("regime"):
    plt.scatter(grp[DATE_COL], grp["basin_total"], s=10, label=regime_val)
plt.title("Time series of basin_total with cluster labels")
plt.xlabel("Date")
plt.ylabel("Daily rainfall total (mm)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "time_series_clusters.png"), dpi=150)
plt.close()

# 10) Save outputs
df_out = df[[DATE_COL] + STATION_COLS + ["basin_total", "cluster", "regime", "pc1", "pc2"]]
df_out.to_csv(os.path.join(OUT_DIR, "clusters.csv"), index=False)

# 11) Save cluster centroids summary
centroid_df = pd.DataFrame(centroids, columns=feature_cols)
centroid_df["cluster"] = range(k)
centroid_df["mapped_regime"] = centroid_df["cluster"].map(mapping)
centroid_df.to_csv(os.path.join(OUT_DIR, "centroids_summary.csv"), index=False)

print("Done. Outputs in:", OUT_DIR)
print("Cluster mapping:", mapping)
