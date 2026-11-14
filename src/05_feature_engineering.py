#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path("../data/processed")
OUT_DIR = DATA_DIR / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Total_Leakage"

H = 12        
D = 24 * H    
W = 7 * D     

VAL_START = "2018-10-01"   
TEST_START = "2019-01-01"  
FIT_END = VAL_START       
FIT_ROWS = "all"          


# In[2]:


files = [DATA_DIR / "2018_combined.csv"]
if (DATA_DIR / "2019_combined.csv").exists():
    files.append(DATA_DIR / "2019_combined.csv")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").set_index("Timestamp")

assert df.index.is_unique, "Duplicate timestamps found"
steps = df.index.to_series().diff().dropna().value_counts()
print("Time steps found:\n", steps)
assert (steps.index == pd.Timedelta(minutes=5)).all(), "Gaps in the 5-minute series: fix before making lags"

P  = [c for c in df.columns if c.startswith("pressure_")]
FL = [c for c in df.columns if c.startswith("flow_")]
DM = [c for c in df.columns if c.startswith("demand_")]
LV = [c for c in df.columns if c.startswith("level_")]
RAW = P + FL + DM + LV
print(f"rows={len(df)}  pressures={len(P)}  flows={len(FL)}  demands={len(DM)}  levels={len(LV)}")
print("Years:", sorted(df.index.year.unique()))


# In[3]:


FEATS, DICT = {}, []

def add(group, name, values, formula, unit):
    col = f"{group}__{name}"
    FEATS[col] = pd.Series(values, index=df.index).astype("float32")
    DICT.append({"feature": col, "group": group, "formula": formula, "unit": unit})

for c in RAW:
    unit = "m" if c.startswith(("pressure_", "level_")) else "m3/h"
    add("G0", c, df[c], "raw reading", unit)
print(len(FEATS), "raw features")


# In[4]:


train_mask = df.index < FIT_END

B = pd.DataFrame(index=df.index)
B["inflow"] = df["flow_p227"] + df["flow_p235"]          
B["pump"] = df["flow_PUMP_1"]
B["demand"] = df[DM].sum(axis=1)                         
B["pump_minus_demand"] = B["pump"] - B["demand"]
B["p_mean"] = df[P].mean(axis=1)
B["p_min"] = df[P].min(axis=1)
B["level"] = df["level_T1"]
UNITS = {"inflow": "m3/h", "pump": "m3/h", "demand": "m3/h", "pump_minus_demand": "m3/h",
         "p_mean": "m", "p_min": "m", "level": "m"}

# G4 water balance
add("G4", "inflow", B["inflow"], "flow_p227 + flow_p235", "m3/h")
add("G4", "demand_sum", B["demand"], "sum of 82 metered demands", "m3/h")
add("G4", "pump_minus_demand", B["pump_minus_demand"], "flow_PUMP_1 - demand_sum", "m3/h")
add("G4", "inflow_minus_demand", B["inflow"] - B["demand"], "inflow - demand_sum", "m3/h")
add("G4", "level_change_1h", B["level"].diff(H), "level_T1(t) - level_T1(t-1h)", "m")
pump_threshold = 0.05 * B.loc[train_mask, "pump"].quantile(0.99)   
add("G4", "pump_on", (B["pump"] > pump_threshold), f"flow_PUMP_1 > {pump_threshold:.2f}", "0/1")

# G3 cross-sensor pressure
add("G3", "p_mean", B["p_mean"], "mean of 33 pressures", "m")
add("G3", "p_std", df[P].std(axis=1), "std of 33 pressures", "m")
add("G3", "p_min", B["p_min"], "min of 33 pressures", "m")
for c in P:
    add("G3", f"{c}_minus_mean", df[c] - B["p_mean"], f"{c} - p_mean", "m")

print("Pump-on fraction:", round(FEATS["G4__pump_on"].mean(), 3))


# In[5]:


hour = df.index.hour + df.index.minute / 60
dow = df.index.dayofweek
add("G5", "hour_sin", np.sin(2 * np.pi * hour / 24), "sin(2*pi*hour/24)", "-")
add("G5", "hour_cos", np.cos(2 * np.pi * hour / 24), "cos(2*pi*hour/24)", "-")
add("G5", "hour_sin2", np.sin(4 * np.pi * hour / 24), "sin(4*pi*hour/24) (morning/evening peaks)", "-")
add("G5", "hour_cos2", np.cos(4 * np.pi * hour / 24), "cos(4*pi*hour/24)", "-")
add("G5", "dow_sin", np.sin(2 * np.pi * dow / 7), "sin(2*pi*dow/7)", "-")
add("G5", "dow_cos", np.cos(2 * np.pi * dow / 7), "cos(2*pi*dow/7)", "-")
add("G5", "weekend", dow >= 5, "Saturday or Sunday", "0/1")


# In[6]:


LAGS = {"5min": 1, "15min": 3, "1h": H}
for s in B.columns:
    for lname, k in LAGS.items():
        add("G1", f"{s}_lag_{lname}", B[s].shift(k), f"{s}(t-{lname})", UNITS[s])
    add("G1", f"{s}_chg_5min", B[s].diff(1), f"{s}(t) - {s}(t-5min)", UNITS[s])
    add("G1", f"{s}_chg_1h", B[s].diff(H), f"{s}(t) - {s}(t-1h)", UNITS[s])
    add("G1", f"{s}_chg_1d", B[s].diff(D), f"{s}(t) - {s}(same time yesterday)", UNITS[s])

# 1-hour change at every pressure sensor
for c in P:
    add("G1", f"{c}_chg_1h", df[c].diff(H), f"{c}(t) - {c}(t-1h)", "m")


# In[7]:


WINDOWS = {"15min": 3, "1h": H, "24h": D, "7d": W}
for s in B.columns:
    for wname, w in WINDOWS.items():
        r = B[s].rolling(w, min_periods=min(w, D))
        add("G2", f"{s}_mean_{wname}", r.mean(), f"rolling mean of {s} over past {wname}", UNITS[s])
        add("G2", f"{s}_std_{wname}", r.std(), f"rolling std of {s} over past {wname}", UNITS[s])
        if wname in ("1h", "24h"):
            add("G2", f"{s}_min_{wname}", r.min(), f"rolling min of {s} over past {wname}", UNITS[s])
            add("G2", f"{s}_max_{wname}", r.max(), f"rolling max of {s} over past {wname}", UNITS[s])
    add("G2", f"{s}_q05_24h", B[s].rolling(D).quantile(0.05),
        f"5th percentile of {s} over past 24h", UNITS[s])

# per pressure sensor: short-term variability and departure from its own 24 h mean
for c in P:
    add("G2", f"{c}_std_1h", df[c].rolling(H).std(), f"rolling std of {c} over past 1h", "m")
    add("G2", f"{c}_minus_mean_24h", df[c] - df[c].rolling(D).mean(), f"{c} - its 24h rolling mean", "m")


# In[8]:


idx = df.index
is_night = (idx.hour >= 2) & (idx.hour < 4)
night_date = idx.normalize()

for s in ["inflow", "pump", "demand", "pump_minus_demand"]:
    nightly = B.loc[is_night, s].groupby(night_date[is_night]).agg(["mean", "min"])
    nightly["vs_week"] = nightly["mean"] - nightly["mean"].shift(1).rolling(7, min_periods=3).median()
    nightly["chg_1d"] = nightly["mean"].diff(1)
    nightly.index = nightly.index + pd.Timedelta(hours=4)          # available from 04:00 onwards
    nf = nightly.reindex(idx, method="ffill")
    add("G6", f"night_{s}_mean", nf["mean"], f"mean {s} 02:00-04:00 of the most recent completed night", UNITS[s])
    add("G6", f"night_{s}_min", nf["min"], f"min {s} 02:00-04:00 of the most recent completed night", UNITS[s])
    add("G6", f"night_{s}_vs_week", nf["vs_week"], f"last night mean - median of previous 7 nights ({s})", UNITS[s])
    add("G6", f"night_{s}_chg_1d", nf["chg_1d"], f"last night mean - the night before ({s})", UNITS[s])


# In[9]:


for s in ["inflow", "pump", "demand", "pump_minus_demand", "p_mean", "p_min"]:
    gap = B[s].ewm(span=D, adjust=False).mean() - B[s].ewm(span=W, adjust=False).mean()
    add("G7", f"{s}_ewm_gap", gap, f"EWM(span 1d) - EWM(span 7d) of {s}", UNITS[s])
    m24 = B[s].rolling(D).mean()
    add("G7", f"{s}_mean24h_chg_1d", m24 - m24.shift(D), f"24h mean of {s} today - 24h mean yesterday", UNITS[s])


# In[10]:


fit_mask = df.index < FIT_END
if FIT_ROWS == "leak_free":
    fit_mask = fit_mask & (df[TARGET] == 0)
print("Rows used to fit G8:", int(fit_mask.sum()))

# 1. daily reference profile (288 slots x weekday/weekend)
slot = (idx.hour * 12 + idx.minute // 5) + 288 * (idx.dayofweek >= 5)
for s in ["inflow", "pump", "demand", "p_mean", "p_min"]:
    ref = B.loc[fit_mask, s].groupby(slot[fit_mask]).median()
    resid = B[s] - ref.reindex(slot).to_numpy()
    add("G8", f"{s}_vs_profile", resid, f"{s} - training median for this time slot", UNITS[s])
    add("G8", f"{s}_vs_profile_mean_24h", resid.rolling(D).mean(), f"24h mean of ({s} - profile)", UNITS[s])

# 2. demand-corrected pressure residuals
Q = pd.concat([df[FL], (df[FL] ** 2).add_suffix("_sq")], axis=1)
print("Missing values in flows:", int(Q.isna().sum().sum()),
      "| in pressures:", int(df[P].isna().sum().sum()))

fm = np.asarray(fit_mask)
Xq = Q.to_numpy(dtype=float)
Yp = df[P].to_numpy(dtype=float)
press_model = LinearRegression().fit(Xq[fm], Yp[fm])
p_resid = df[P] - press_model.predict(Xq)
z = p_resid / p_resid[fm].std()

# 3. PCA reconstruction error on standardised pressures
p_scaler = StandardScaler().fit(df.loc[fit_mask, P])
pca = PCA(n_components=0.95).fit(p_scaler.transform(df.loc[fit_mask, P]))
Zp = p_scaler.transform(df[P])
recon_err = ((Zp - pca.inverse_transform(pca.transform(Zp))) ** 2).mean(axis=1)
recon_err = pd.Series(recon_err, index=idx)
add("G8", "pca_recon_err", recon_err, f"PCA ({pca.n_components_} comps, 95% var) reconstruction MSE", "-")
add("G8", "pca_recon_err_mean_24h", recon_err.rolling(D).mean(), "24h mean of pca_recon_err", "-")
print("PCA components kept:", pca.n_components_)


# In[ ]:


X = pd.DataFrame(FEATS)
y = df[TARGET]
feature_dict = pd.DataFrame(DICT)

print("Total features:", X.shape[1])
print(feature_dict.groupby("group").size().rename("n_features"))

first_complete = X.notna().all(axis=1).idxmax()
warmup = X.index < first_complete
print(f"Warm-up rows removed: {warmup.sum()} (until {first_complete})")
X, y = X.loc[~warmup], y.loc[~warmup]

remaining_nan = int(X.isna().sum().sum())
print("Remaining NaN values:", remaining_nan)
assert remaining_nan == 0
assert np.isfinite(X.to_numpy()).all()

tr = X.index < VAL_START
va = (X.index >= VAL_START) & (X.index < TEST_START)
te = X.index >= TEST_START
X_train, y_train = X.loc[tr], y.loc[tr]
X_val, y_val = X.loc[va], y.loc[va]
X_test, y_test = X.loc[te], y.loc[te]
for name, a, b in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
    print(f"{name:5s}: X={a.shape}  y={b.shape}  leakage range {b.min() if len(b) else 0:.1f}-{b.max() if len(b) else 0:.1f} m3/h")


# In[ ]:


const_cols = X_train.columns[X_train.std() < 1e-8].tolist()
dup_cols = X_train.columns[X_train.T.duplicated()].tolist()
drop_cols = sorted(set(const_cols + dup_cols))
print("Constant:", const_cols)
print("Duplicates:", dup_cols)
X_train, X_val, X_test = (d.drop(columns=drop_cols) for d in (X_train, X_val, X_test))
feature_dict["kept"] = ~feature_dict["feature"].isin(drop_cols)
print("Features kept:", X_train.shape[1])


# In[ ]:


# In[47]: save features as pickle (.pkl) and CSV (.csv)

def save_both(obj, name):
    """Save a DataFrame or Series as .pkl and .csv, keeping the Timestamp index."""
    frame = obj.to_frame() if isinstance(obj, pd.Series) else obj
    frame.to_pickle(OUT_DIR / f"{name}.pkl")
    frame.to_csv(OUT_DIR / f"{name}.csv", index=True, index_label="Timestamp")

splits = [("X_train_fe", X_train), ("X_val_fe", X_val),
          ("y_train_fe", y_train), ("y_val_fe", y_val)]
if len(X_test):
    splits += [("X_test_fe", X_test), ("y_test_fe", y_test)]

for name, obj in splits:
    save_both(obj, name)
    print(f"{name:12s} shape={obj.shape}  -> {name}.pkl + {name}.csv")

# feature dictionary (CSV only, it is a small lookup table)
feature_dict.to_csv(OUT_DIR / "feature_dictionary.csv", index=False)

# fitted objects (scaler, PCA, models) cannot go to CSV, so they stay in joblib
joblib.dump({"press_model": press_model, "p_scaler": p_scaler, "pca": pca,
             "pump_threshold": pump_threshold, "fit_end": FIT_END, "fit_rows": FIT_ROWS,
             "dropped_columns": drop_cols, "warmup_rows": int(warmup.sum())},
            OUT_DIR / "fe_fitted_objects.joblib")

print("Saved to", OUT_DIR.resolve())
print(sorted(p.name for p in OUT_DIR.iterdir()))


# In[11]:


from pathlib import Path

print("Notebook is running in:", Path.cwd())

root = Path.home() / "Jupyter" / "ML-Water-Pipe-Leak-Detection"
found = [p for p in root.rglob("*combined*.csv") if "venv" not in p.parts]
for p in found:
    print(p.relative_to(root))


# In[12]:


files = [DATA_DIR / "2018_combined.csv"]
if (DATA_DIR / "2019_combined.csv").exists():
    files.append(DATA_DIR / "2019_combined.csv")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").set_index("Timestamp")

assert df.index.is_unique, "Duplicate timestamps found"
steps = df.index.to_series().diff().dropna().value_counts()
print("Time steps found:\n", steps)
assert (steps.index == pd.Timedelta(minutes=5)).all(), "Gaps in the 5-minute series: fix before making lags"

P  = [c for c in df.columns if c.startswith("pressure_")]
FL = [c for c in df.columns if c.startswith("flow_")]
DM = [c for c in df.columns if c.startswith("demand_")]
LV = [c for c in df.columns if c.startswith("level_")]
RAW = P + FL + DM + LV
print(f"rows={len(df)}  pressures={len(P)}  flows={len(FL)}  demands={len(DM)}  levels={len(LV)}")
print("Years:", sorted(df.index.year.unique()))

