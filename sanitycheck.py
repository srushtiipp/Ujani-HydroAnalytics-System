# repair_clusters_and_write_json.py
import pandas as pd, os, json

# Filenames (adjust if yours differ)
CLUST_CSV = "clusters.csv"
DATA_CSV = "Data csv.csv"   # original rainfall file you used for modelling
OUT_DIR = "frontend_data"
OUT_JSON = os.path.join(OUT_DIR, "clusters.json")
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(CLUST_CSV):
    print("ERROR: clusters.csv not found."); raise SystemExit(1)
if not os.path.exists(DATA_CSV):
    print("WARNING: Data csv.csv not found. We will try to fill dates from clusters.csv only.")

# load clusters CSV (preserve columns)
dfc = pd.read_csv(CLUST_CSV)
print("clusters.csv columns:", list(dfc.columns)[:20])

# find the date column (trim whitespace)
date_col = None
for c in dfc.columns:
    if c.strip().lower() == "date":
        date_col = c
        break

if date_col is None:
    print("No date-like column named 'Date' found in clusters.csv. Existing columns:", list(dfc.columns))
else:
    print("Detected date column in clusters.csv:", repr(date_col))

# parse the date column if found
if date_col:
    dfc[date_col] = pd.to_datetime(dfc[date_col], dayfirst=True, errors="coerce")

# if Data csv exists, use it as canonical full date list
if os.path.exists(DATA_CSV):
    dfd = pd.read_csv(DATA_CSV)
    dfd.columns = [c.strip() for c in dfd.columns]
    if 'Date' in dfd.columns:
        dfd['Date'] = pd.to_datetime(dfd['Date'], dayfirst=True, errors='coerce')
        print("Loaded Data csv.csv dates:", dfd['Date'].notna().sum(), "valid dates.")
        # if lengths match, use dfd['Date'] to fill clusters
        if len(dfd) == len(dfc):
            print("Data csv and clusters.csv have same number of rows; using Data csv Date to fill missing cluster dates.")
            dfc['Date_filled'] = dfd['Date']
        else:
            # If different lengths, take non-null dates from dfd and align by index where possible
            valid_dates = dfd['Date'].dropna().tolist()
            if len(valid_dates) >= len(dfc):
                print("Using last", len(dfc), "dates from Data csv to fill cluster dates.")
                dfc['Date_filled'] = valid_dates[-len(dfc):]
            else:
                print("Data csv has fewer valid dates than clusters rows. Will attempt to combine existing cluster dates and forward-fill.")
                dfc['Date_filled'] = dfc[date_col] if date_col else pd.NaT
    else:
        print("Data csv.csv has no 'Date' column. We will attempt to forward/back-fill cluster dates only.")
        dfc['Date_filled'] = dfc[date_col] if date_col else pd.NaT
else:
    # no Data csv; use cluster date column if present and attempt fill
    dfc['Date_filled'] = dfc[date_col] if date_col else pd.NaT

# If Date_filled has NaT, try forward-fill then back-fill
if dfc['Date_filled'].isna().any():
    n_before = dfc['Date_filled'].isna().sum()
    dfc['Date_filled'] = dfc['Date_filled'].ffill().bfill()
    n_after = dfc['Date_filled'].isna().sum()
    print(f"Attempted ffill/bfill: NaNs before={n_before}, after={n_after}")

# Ensure all dates are valid - drop rows without date (if any remain)
valid_mask = dfc['Date_filled'].notna()
if not valid_mask.all():
    print("Dropping rows without valid dates:", int((~valid_mask).sum()))
dfc_clean = dfc[valid_mask].copy()
dfc_clean['date'] = dfc_clean['Date_filled'].dt.strftime("%Y-%m-%d")

# Map rainfall and other columns to frontend fields (robust detection)
def find_col(df, candidates):
    for x in candidates:
        if x in df.columns:
            return x
    return None

rain_col = find_col(dfc_clean, ["basin_total", "rainfall", "total", "Basin_total"])
cluster_col = find_col(dfc_clean, ["cluster", "regime", "regime_label"])
pc1_col = find_col(dfc_clean, ["pc1", "PC1", "pc_1"])
pc2_col = find_col(dfc_clean, ["pc2", "PC2", "pc_2"])

print("Mapping -> rain:", rain_col, "cluster:", cluster_col, "pc1:", pc1_col, "pc2:", pc2_col)

records = []
for _, r in dfc_clean.iterrows():
    records.append({
        "date": r['date'],
        "rainfall": None if pd.isna(r.get(rain_col)) else float(r.get(rain_col)),
        "cluster": "" if pd.isna(r.get(cluster_col)) else str(r.get(cluster_col)),
        "pc1": None if pd.isna(r.get(pc1_col)) else float(r.get(pc1_col)),
        "pc2": None if pd.isna(r.get(pc2_col)) else float(r.get(pc2_col))
    })

if not records:
    print("No valid cluster records produced. Exiting.")
    raise SystemExit(1)

with open(OUT_JSON, "w", encoding="utf8") as f:
    json.dump({"data": records}, f, indent=2, ensure_ascii=False)

print("Wrote", OUT_JSON, "rows:", len(records))
