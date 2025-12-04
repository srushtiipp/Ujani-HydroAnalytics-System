# debug_and_predict.py
# Diagnoses feature / LightGBM prediction issues and attempts safe auto-alignment
# Usage: python debug_and_predict.py

import os
import pandas as pd
import numpy as np
import joblib
import traceback

INPUT = "Data csv.csv"   # your rainfall CSV
DATE_COL = "Date"
STATION_COLS = ["Alandi", "Budhwad (Velholi)", "Koliye", "Paud"]
MODELS_DIR = os.path.join("forecasts_output", "models")
MODEL_FILE = os.path.join(MODELS_DIR, "lgb_main.pkl")  # primary model to inspect

N_LAGS = 7
ROLL_WINDOWS = [3,7,14]

def safe_load_model(path):
    print("Loading model:", path)
    m = joblib.load(path)
    # get LightGBM internal feature names if available
    feature_names = None
    try:
        # if LGBMRegressor wrapper
        booster = m.booster_ if hasattr(m, "booster_") else None
        if booster is not None:
            feature_names = booster.feature_name()
        else:
            # maybe it's a direct booster
            if hasattr(m, "feature_name"):
                feature_names = m.feature_name()
    except Exception as e:
        print("Could not introspect model feature names:", e)
    print("Model loaded.")
    return m, feature_names

def build_features(df):
    df = df.copy()
    # parse date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # ensure station columns numeric
    for c in STATION_COLS:
        if c not in df.columns:
            raise KeyError(f"Station column missing from input CSV: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["basin_total"] = df[STATION_COLS].sum(axis=1)

    # lags
    for lag in range(1, N_LAGS+1):
        df[f"basin_lag_{lag}"] = df["basin_total"].shift(lag)

    for s in STATION_COLS:
        for lag in range(1,3):
            df[f"{s}_lag_{lag}"] = df[s].shift(lag)

    for w in ROLL_WINDOWS:
        df[f"basin_roll_{w}"] = df["basin_total"].rolling(window=w, min_periods=1).sum()

    df["dayofyear"] = df[DATE_COL].dt.dayofyear
    df["month"] = df[DATE_COL].dt.month
    df["weekday"] = df[DATE_COL].dt.weekday

    # target (optional)
    df["obs_next_day"] = df["basin_total"].shift(-1)
    return df

def main():
    try:
        if not os.path.exists(INPUT):
            print("ERROR: input CSV not found at", INPUT)
            return

        df_raw = pd.read_csv(INPUT)
        df_raw.columns = [c.strip() for c in df_raw.columns]
        print("Loaded input CSV. Columns:", list(df_raw.columns))

        # build features
        df = build_features(df_raw)
        print("Built features. Rows:", len(df))

        # show sample rows and NaN diagnostic
        sample = df.head(12)[["Date"] + STATION_COLS + [f"basin_lag_{i}" for i in range(1,4)] + ["basin_roll_3", "dayofyear"]]
        print("Sample (first 12 rows):\n", sample.to_string(index=False))
        print("Any NaT in Date column?", df["Date"].isna().any())
        # count NaNs per feature column
        feat_cols = [c for c in df.columns if (c.startswith("basin_lag_") or c.startswith("basin_roll_") or any(c.startswith(s) and "_lag" in c for s in STATION_COLS) or c in ["dayofyear","month","weekday"])]
        print("Number of feature columns found:", len(feat_cols))
        nan_counts = {c: int(df[c].isna().sum()) for c in feat_cols}
        print("NaN counts (feature columns):")
        for k,v in nan_counts.items():
            print(f"  {k}: {v}")

        # load model, inspect feature names
        model, model_feature_names = safe_load_model(MODEL_FILE)
        print("Model reported feature names (len):", None if model_feature_names is None else len(model_feature_names))
        if model_feature_names:
            print(model_feature_names[:50])

        # Now create a DataFrame X_ready that matches model_feature_names if available
        X_df = df[feat_cols].copy()
        print("X_df shape (before alignment):", X_df.shape)
        # if model_feature_names available, align columns
        if model_feature_names:
            model_f = list(model_feature_names)
            # find missing and extra
            missing = [c for c in model_f if c not in X_df.columns]
            extra = [c for c in X_df.columns if c not in model_f]
            print("Features missing (model expects but not present):", missing)
            print("Extra features (present but model doesn't expect):", extra)
            # if missing, create them filled with zeros (safe fallback), but warn
            for c in missing:
                print(f"Filling missing feature {c} with zeros.")
                X_df[c] = 0.0
            # drop extra
            if extra:
                print("Dropping extra features:", extra)
                X_df = X_df.drop(columns=extra)
            # reorder to model order
            X_df = X_df[model_f]
            print("Aligned X_df to model feature order. New shape:", X_df.shape)
        else:
            print("Model feature names not available. Using auto-detected feature columns order (may differ from training).")

        # check for NaNs in final X_df
        nan_after = X_df.isna().sum().sum()
        print("Total NaN cells in final feature DataFrame:", int(nan_after))
        if nan_after > 0:
            print("Filling NaNs with 0.0 for prediction (but this may be suboptimal).")
            X_df = X_df.fillna(0.0)

        # test prediction on the last row that has full features
        valid_idx = (~X_df.isna().any(axis=1)).to_numpy().nonzero()[0]
        if len(valid_idx) == 0:
            print("ERROR: no valid rows available for prediction after alignment. Exiting.")
            return
        last_i = valid_idx[-1]
        row_feat = X_df.iloc[last_i:last_i+1]
        print("Predicting for index (row) in training-aligned DF:", last_i, "Date:", df.loc[row_feat.index[0], "Date"])
        try:
            pred = model.predict(row_feat)
            print("Prediction success. Predicted value:", float(pred[0]))
        except Exception as e:
            print("Prediction failed with exception:")
            traceback.print_exc()
            # attempt to print shapes & dtypes
            print("Row features shape:", row_feat.shape)
            print("Row dtypes:\n", row_feat.dtypes)
            print("Model type:", type(model))
            # if LightGBM, try booster interface
            try:
                booster = model.booster_ if hasattr(model, "booster_") else model
                print("Booster feature names:", booster.feature_name()[:40])
            except Exception:
                pass
            return

        # optionally, produce a small CSV for the last N predictions
        N=200
        valid_rows = valid_idx[-N:] if len(valid_idx) >= 1 else valid_idx
        preds = model.predict(X_df.iloc[valid_rows])
        out = []
        for i,p in zip(valid_rows, preds):
            d = df.iloc[i]["Date"]
            out.append({"date_of_input": d.strftime("%Y-%m-%d") if pd.notna(d) else None, "pred_next_day": float(p), "obs_next_day": (None if pd.isna(df.iloc[i]['obs_next_day']) else float(df.iloc[i]['obs_next_day']))})
        out_df = pd.DataFrame(out)
        print("Sample forecasts produced (head):")
        print(out_df.head().to_string(index=False))
        # save small preview
        out_df.to_csv("forecasts_preview.csv", index=False)
        print("Wrote forecasts_preview.csv with", len(out_df), "rows.")
    except Exception as e:
        print("Fatal error during debug run:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
