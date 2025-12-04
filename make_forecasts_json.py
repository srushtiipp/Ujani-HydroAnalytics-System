# make_forecasts_json_fixed.py
import pandas as pd, os, json, numpy as np

CSV = "forecasts.csv"
OUT_DIR = "frontend_data"
OUT = os.path.join(OUT_DIR, "forecasts.json")
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(CSV):
    print("ERROR: forecasts.csv not found.")
    raise SystemExit(1)

df = pd.read_csv(CSV)
if df.empty:
    print("ERROR: forecasts.csv is empty. Regenerate forecasts first (regenerate_forecasts.py).")
    raise SystemExit(1)

# helper
candidates = lambda names: next((c for c in names if c in df.columns), None)
date_col = candidates(["date_of_input","date","Date ","date_input"])
pred_col = candidates(["pred_next_day","predicted","pred"])
q10_col = candidates(["q10","q_10","q_low"])
q50_col = candidates(["q50","q_50","q_median"])
q90_col = candidates(["q90","q_90","q_high"])
obs_col = candidates(["obs_next_day","obs","observed","obs_next"])

print("Detected:", date_col, pred_col, q10_col, q50_col, q90_col, obs_col)

records = []
for _, r in df.iterrows():
    d = r.get(date_col)
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
        "observed": None if pd.isna(r.get(obs_col)) else float(r.get(obs_col)),
        "predicted": None if pd.isna(r.get(pred_col)) else float(r.get(pred_col)),
        "q10": None if pd.isna(r.get(q10_col)) else float(r.get(q10_col)),
        "q50": None if pd.isna(r.get(q50_col)) else (None if pd.isna(r.get(q50_col)) else float(r.get(q50_col))),
        "q90": None if pd.isna(r.get(q90_col)) else float(r.get(q90_col))
    }
    records.append(rec)

if not records:
    print("No valid forecast rows with dates found.")
    raise SystemExit(1)

latest = records[-1]
out = {"data": records, "latest_forecast": {"date": latest["date"], "predicted": latest["predicted"], "q10": latest["q10"], "q90": latest["q90"]}}

with open(OUT, "w", encoding="utf8") as f:
    json.dump(out, f, indent=2, allow_nan=False)

print(f"Wrote {OUT} ({len(records)} rows). Latest: {latest['date']}")
