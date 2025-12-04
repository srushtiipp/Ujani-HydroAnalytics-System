# repair_forecasts_and_write_json.py
import pandas as pd, os, json

F_CSV = "forecasts.csv"
CLUST_JSON = os.path.join("frontend_data", "clusters.json")
DATA_CSV = "Data csv.csv"
OUT_DIR = "frontend_data"
OUT_JSON = os.path.join(OUT_DIR, "forecasts.json")
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(F_CSV):
    print("ERROR: forecasts.csv missing."); raise SystemExit(1)

df = pd.read_csv(F_CSV)
print("forecasts.csv columns:", list(df.columns))
if df.empty:
    print("ERROR: forecasts.csv is empty - you must regenerate forecasts from models.")
    raise SystemExit(1)

# If date_of_input exists but is NaN for all rows, try to fill using Data csv or clusters.json
date_col = next((c for c in df.columns if "date" in c.lower()), None)
if date_col is None:
    print("No date-like column in forecasts.csv; creating 'date_of_input'")
    df['date_of_input'] = pd.NaT
    date_col = 'date_of_input'

# check if all NaN
if df[date_col].isna().all():
    print("All forecast dates are NaN — attempting to recover dates...")

    # 1) try clusters.json dates (if exists)
    if os.path.exists(CLUST_JSON):
        with open(CLUST_JSON, "r", encoding="utf8") as f:
            cl = json.load(f).get("data", [])
        cl_dates = [r.get("date") for r in cl if r.get("date")]
        if len(cl_dates) >= len(df):
            use_dates = cl_dates[-len(df):]
            df[date_col] = pd.to_datetime(use_dates, errors="coerce")
            print("Filled forecast dates from clusters.json (last entries).")
        else:
            print("clusters.json doesn't have enough dates (has", len(cl_dates), ").")
    # 2) else try Data csv
    if df[date_col].isna().all() and os.path.exists(DATA_CSV):
        dfd = pd.read_csv(DATA_CSV)
        dfd.columns = [c.strip() for c in dfd.columns]
        if 'Date' in dfd.columns:
            dfd['Date'] = pd.to_datetime(dfd['Date'], dayfirst=True, errors='coerce')
            avail = dfd['Date'].dropna().tolist()
            if len(avail) >= len(df):
                df[date_col] = pd.to_datetime(avail[-len(df):])
                print("Filled forecast dates from Data csv (last entries).")
            else:
                print("Data csv has insufficient dates (", len(avail), ") to fill forecasts (need", len(df), ").")
    # 3) if still NaN, attempt to fill by generating a contiguous date range if first valid cluster date exists
    if df[date_col].isna().all():
        print("Could not auto-fill forecast dates. Please regenerate forecasts (preferred) or provide a Date source.")
        # save a copy and exit
        df.to_csv("forecasts_dates_unfilled_debug.csv", index=False)
        raise SystemExit(1)

# At this point df[date_col] should have valid dates
df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
df = df.dropna(subset=[date_col]).copy()
df['date'] = df[date_col].dt.strftime("%Y-%m-%d")

# map columns
def first_match(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

pred_col = first_match(df.columns, ["pred_next_day","predicted","pred"])
q10_col = first_match(df.columns, ["q10","q_10","q_low"])
q50_col = first_match(df.columns, ["q50","q_50","q_median"])
q90_col = first_match(df.columns, ["q90","q_90","q_high"])
obs_col = first_match(df.columns, ["obs_next_day","obs","observed","obs_next"])

print("Detected mapping -> pred:", pred_col, "q10:", q10_col, "q90:", q90_col, "obs:", obs_col)

records = []
for _, r in df.iterrows():
    records.append({
        "date": r['date'],
        "observed": None if pd.isna(r.get(obs_col)) else float(r.get(obs_col)),
        "predicted": None if pd.isna(r.get(pred_col)) else float(r.get(pred_col)),
        "q10": None if pd.isna(r.get(q10_col)) else float(r.get(q10_col)),
        "q50": None if pd.isna(r.get(q50_col)) else (None if pd.isna(r.get(q50_col)) else float(r.get(q50_col))),
        "q90": None if pd.isna(r.get(q90_col)) else float(r.get(q90_col))
    })

if not records:
    print("No valid forecast records after filling dates.")
    raise SystemExit(1)

latest = records[-1]
out = {"data": records, "latest_forecast": {"date": latest["date"], "predicted": latest["predicted"], "q10": latest["q10"], "q90": latest["q90"]}}

with open(OUT_JSON, "w", encoding="utf8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print("Wrote", OUT_JSON, "rows:", len(records), "Latest:", latest["date"])
