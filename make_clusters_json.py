# make_clusters_json_fixed.py
import pandas as pd, os, json, numpy as np

CSV = "clusters.csv"
OUT_DIR = "frontend_data"
OUT = os.path.join(OUT_DIR, "clusters.json")
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(CSV):
    print("ERROR: clusters.csv not found.")
    raise SystemExit(1)

df = pd.read_csv(CSV)
if df.empty:
    print("ERROR: clusters.csv is empty.")
    raise SystemExit(1)

# detect columns
candidates = lambda names: next((c for c in names if c in df.columns), None)
date_col = candidates(["date","Date","date_of_input","Date_of_input"])
rain_col = candidates(["basin_total","rainfall","rain","total"])
cluster_col = candidates(["cluster","regime","regime_label"])
pc1_col = candidates(["pc1","PC1","pc_1"])
pc2_col = candidates(["pc2","PC2","pc_2"])

print("Detected:", date_col, rain_col, cluster_col, pc1_col, pc2_col)

records = []
for _, r in df.iterrows():
    d = r.get(date_col) if date_col else None
    # coerce to date-string
    try:
        if pd.notna(d):
            d = pd.to_datetime(d, dayfirst=True, errors='coerce')
            if pd.isna(d):
                continue
            d = d.strftime("%Y-%m-%d")
        else:
            continue
    except Exception:
        continue

    rec = {
        "date": d,
        "rainfall": None if pd.isna(r.get(rain_col)) else float(r.get(rain_col)),
        "cluster": "" if pd.isna(r.get(cluster_col)) else str(r.get(cluster_col)),
        "pc1": None if pd.isna(r.get(pc1_col)) else float(r.get(pc1_col)),
        "pc2": None if pd.isna(r.get(pc2_col)) else float(r.get(pc2_col))
    }
    records.append(rec)

if not records:
    print("No valid rows with dates found in clusters.csv.")
    raise SystemExit(1)

with open(OUT, "w", encoding="utf8") as f:
    json.dump({"data": records}, f, indent=2, allow_nan=False)

print(f"Wrote {OUT} ({len(records)} rows).")
