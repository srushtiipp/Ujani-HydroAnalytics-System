"""
forecast_nextday_fixed.py
Fixed version: trains next-day basin_total predictor + quantile models and
safely produces forecasts CSV and plots without crashing on NaT dates.

Requirements:
pip install pandas numpy scikit-learn lightgbm joblib matplotlib
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import math
import matplotlib.pyplot as plt
from lightgbm import early_stopping, log_evaluation

# ---------------- USER SETTINGS ----------------
INPUT = "Data csv.csv"   # update if your CSV name differs
DATE_COL = "Date"
STATION_COLS = ["Alandi", "Budhwad (Velholi)", "Koliye", "Paud"]
OUT_DIR = "forecasts_output"
N_LAGS = 7
ROLL_WINDOWS = [3, 7, 14]
TEST_FRACTION = 0.15
RANDOM_STATE = 42
Q_LO = 0.10
Q_HI = 0.90

LGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "objective": "regression",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1
}
LGB_Q_PARAMS = LGB_PARAMS.copy()
LGB_Q_PARAMS["objective"] = "quantile"
# ------------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)

# ----------------- Load & prep -----------------
df = pd.read_csv(INPUT)
df.columns = [c.strip() for c in df.columns]

# parse date
df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
df = df.sort_values(DATE_COL).reset_index(drop=True)

# ensure station numeric
for c in STATION_COLS:
    if c not in df.columns:
        raise KeyError(f"Station column missing: {c}. Available columns: {list(df.columns)}")
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

# basin_total
df["basin_total"] = df[STATION_COLS].sum(axis=1)

# feature engineering
for lag in range(1, N_LAGS + 1):
    df[f"basin_lag_{lag}"] = df["basin_total"].shift(lag)

for s in STATION_COLS:
    for lag in range(1, 3):
        df[f"{s}_lag_{lag}"] = df[s].shift(lag)

for w in ROLL_WINDOWS:
    df[f"basin_roll_{w}"] = df["basin_total"].rolling(window=w, min_periods=1).sum()

df["dayofyear"] = df[DATE_COL].dt.dayofyear
df["month"] = df[DATE_COL].dt.month
df["weekday"] = df[DATE_COL].dt.weekday

# target next day
df["target_next_day"] = df["basin_total"].shift(-1)

# drop rows missing required lag/target
df_model = df.dropna(subset=[f"basin_lag_{N_LAGS}", "target_next_day"]).copy()

FEATURE_COLS = [col for col in df_model.columns if
                (col.startswith("basin_lag_") or col.startswith("basin_roll_") or
                 any(col.startswith(s) and "_lag" in col for s in STATION_COLS) or
                 col in ["dayofyear", "month", "weekday"])]

print("Features used:", FEATURE_COLS)

X = df_model[FEATURE_COLS].values
y = df_model["target_next_day"].values
dates = df_model[DATE_COL].values  # date corresponding to the row (prediction for next day)

# time-based split
n = len(df_model)
test_n = max(1, int(math.ceil(TEST_FRACTION * n)))
train_idx_end = n - test_n
X_train, X_test = X[:train_idx_end], X[train_idx_end:]
y_train, y_test = y[:train_idx_end], y[train_idx_end:]
dates_test = dates[train_idx_end:]

print(f"Total rows for modelling: {n}; Train: {len(X_train)}; Test: {len(X_test)}")

# ----------------- train main regressor -----------------
print("Training main LightGBM regressor (L2)...")
model_main = lgb.LGBMRegressor(**LGB_PARAMS)
model_main.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[early_stopping(50), log_evaluation(0)]
)

# Convert test arrays to DataFrame with correct column names (avoids feature-name warnings)
X_test_df = pd.DataFrame(X_test, columns=FEATURE_COLS)
X_train_df = pd.DataFrame(X_train, columns=FEATURE_COLS)

# predictions (point)
pred_main_test = model_main.predict(X_test_df)

# ----------------- train quantile models -----------------
def train_quantile(alpha, X_tr, y_tr, X_val=None, y_val=None):
    params = LGB_Q_PARAMS.copy()
    params["alpha"] = alpha
    model_q = lgb.LGBMRegressor(**params)
    if X_val is not None:
        model_q.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[early_stopping(50), log_evaluation(0)])
    else:
        model_q.fit(X_tr, y_tr)
    return model_q

print("Training quantile models...")
model_q10 = train_quantile(Q_LO, X_train_df, y_train, X_test_df, y_test)
model_q50 = train_quantile(0.5, X_train_df, y_train, X_test_df, y_test)
model_q90 = train_quantile(Q_HI, X_train_df, y_train, X_test_df, y_test)

pred_q10_test = model_q10.predict(X_test_df)
pred_q50_test = model_q50.predict(X_test_df)
pred_q90_test = model_q90.predict(X_test_df)

# ----------------- evaluation -----------------
def rmse(a, b):
    return np.sqrt(np.mean((np.array(a) - np.array(b))**2))

def nse(obs, sim):
    obs = np.array(obs); sim = np.array(sim)
    denom = np.sum((obs - obs.mean())**2)
    num = np.sum((obs - sim)**2)
    return 1 - (num / denom) if denom != 0 else float("nan")

rmse_main = rmse(y_test, pred_main_test)
mae_main = np.mean(np.abs(np.array(y_test) - np.array(pred_main_test)))
nse_main = nse(y_test, pred_main_test)

rmse_q50 = rmse(y_test, pred_q50_test)
mae_q50 = np.mean(np.abs(np.array(y_test) - np.array(pred_q50_test)))
nse_q50 = nse(y_test, pred_q50_test)

print("\n--- Evaluation on test set ---")
print(f"Main (L2) RMSE: {rmse_main:.3f}, MAE: {mae_main:.3f}, NSE: {nse_main:.3f}")
print(f"Quantile(0.5) RMSE: {rmse_q50:.3f}, MAE: {mae_q50:.3f}, NSE: {nse_q50:.3f}")

# ----------------- produce forecasts CSV for test period (robust) -----------------
out_rows = []
for i in range(len(X_test_df)):
    # dates_test corresponds to the row used to predict next-day
    row_date = pd.to_datetime(dates_test[i], errors="coerce")  # safe coercion
    if pd.isna(row_date):
        # skip rows with invalid date
        continue

    out_rows.append({
        "date_of_input": row_date.strftime("%Y-%m-%d"),
        "pred_next_day": float(pred_main_test[i]),
        "q10": float(pred_q10_test[i]),
        "q50": float(pred_q50_test[i]),
        "q90": float(pred_q90_test[i]),
        "obs_next_day": float(y_test[i])
    })

df_forecasts = pd.DataFrame(out_rows)

if df_forecasts.empty:
    print("Warning: no valid forecast rows produced (df_forecasts is empty). Check your dates/lags.")
else:
    df_forecasts.to_csv(os.path.join(OUT_DIR, "forecasts.csv"), index=False)
    print("Saved forecasts.csv in", OUT_DIR)

# ----------------- save models -----------------
joblib.dump(model_main, os.path.join(OUT_DIR, "models", "lgb_main.pkl"))
joblib.dump(model_q10, os.path.join(OUT_DIR, "models", "lgb_q10.pkl"))
joblib.dump(model_q50, os.path.join(OUT_DIR, "models", "lgb_q50.pkl"))
joblib.dump(model_q90, os.path.join(OUT_DIR, "models", "lgb_q90.pkl"))
print("Saved models to", os.path.join(OUT_DIR, "models"))

# ----------------- plot obs vs pred for test (only if df_forecasts exists) -----------------
if not df_forecasts.empty:
    plt.figure(figsize=(12,4))
    plt.plot(pd.to_datetime(df_forecasts["date_of_input"]), df_forecasts["obs_next_day"], label="Observed next-day")
    plt.plot(pd.to_datetime(df_forecasts["date_of_input"]), df_forecasts["pred_next_day"], label="Predicted (main)")
    plt.fill_between(pd.to_datetime(df_forecasts["date_of_input"]),
                     df_forecasts["q10"], df_forecasts["q90"], alpha=0.2, label="q10-q90")
    plt.legend()
    plt.title("Observed vs Predicted (test period)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "obs_vs_pred.png"), dpi=150)
    plt.close()
    print("Saved obs_vs_pred.png")
else:
    print("Skipped plotting because df_forecasts is empty.")

# ----------------- helper: predict for latest available date -----------------
def predict_next_day_for_latest(df_full, model_point, model_q10, model_q50, model_q90, feature_cols):
    tmp = df_full.copy()
    tmp = tmp.sort_values(DATE_COL).reset_index(drop=True)

    for lag in range(1, N_LAGS+1):
        tmp[f"basin_lag_{lag}"] = tmp["basin_total"].shift(lag)
    for s in STATION_COLS:
        for lag in range(1,3):
            tmp[f"{s}_lag_{lag}"] = tmp[s].shift(lag)
    for w in ROLL_WINDOWS:
        tmp[f"basin_roll_{w}"] = tmp["basin_total"].rolling(window=w, min_periods=1).sum()
    tmp["dayofyear"] = tmp[DATE_COL].dt.dayofyear
    tmp["month"] = tmp[DATE_COL].dt.month
    tmp["weekday"] = tmp[DATE_COL].dt.weekday

    last_row = tmp.iloc[-1]
    # if any required features are missing, return None
    for fc in feature_cols:
        if fc not in last_row or pd.isna(last_row[fc]):
            return None

    feat = last_row[feature_cols].values.reshape(1,-1)
    p = float(model_point.predict(feat)[0])
    ql = float(model_q10.predict(feat)[0])
    qm = float(model_q50.predict(feat)[0])
    qh = float(model_q90.predict(feat)[0])
    next_date = (last_row[DATE_COL] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return {"date_for": next_date, "pred_next_day": p, "q10": ql, "q50": qm, "q90": qh}

latest_pred = predict_next_day_for_latest(df, model_main, model_q10, model_q50, model_q90, FEATURE_COLS)
print("Example latest prediction (for frontend):", latest_pred)

print("Done.")
