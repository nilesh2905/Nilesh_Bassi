import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX


DATA_PATH = "tesla_deliveries_dataset_2015_2025.csv"
# LOAD & BUILD MONTHLY TIME SERIES

df_raw = pd.read_csv(DATA_PATH)

ts = (df_raw.groupby(["Year", "Month"])["Estimated_Deliveries"]
            .sum()
            .reset_index()
            .sort_values(["Year", "Month"]))

ts["Date"] = pd.to_datetime({"year": ts["Year"], "month": ts["Month"], "day": 1})
ts = ts.set_index("Date")[["Estimated_Deliveries"]].rename(
        columns={"Estimated_Deliveries": "Deliveries"})

print(f"Series shape : {ts.shape}")
print(ts.head())

# TIME SERIES COMPONENTS

decomp = seasonal_decompose(ts["Deliveries"], model="additive", period=12)

fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Time Series Decomposition — Monthly Tesla Deliveries", fontsize=13, fontweight="bold")

for ax, data, label, color in zip(
    axes,
    [ts["Deliveries"], decomp.trend, decomp.seasonal, decomp.resid],
    ["Observed", "Trend", "Seasonal", "Residual"],
    ["steelblue", "darkorange", "seagreen", "crimson"]
):
    ax.plot(data, color=color, linewidth=1.5)
    ax.set_ylabel(label, fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ts_decomposition.png", dpi=130, bbox_inches="tight")
plt.show()

# STATIONARITY — ADF Test
def adf_test(series, label="Series"):
    result = adfuller(series.dropna())
    print(f"\nADF Test — {label}")
    print(f"  Test Statistic : {result[0]:.4f}")
    print(f"  p-value        : {result[1]:.4f}")
    print(f"  Stationary     : {'YES ✔' if result[1] < 0.05 else 'NO ✘'}")

adf_test(ts["Deliveries"], "Original Series")

ts["Deliveries_diff"] = ts["Deliveries"].diff()
adf_test(ts["Deliveries_diff"].dropna(), "First Differenced")

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
axes[0].plot(ts["Deliveries"],      color="steelblue",  linewidth=1.5)
axes[0].set_title("Original Series"); axes[0].grid(True, alpha=0.3)
axes[1].plot(ts["Deliveries_diff"], color="darkorange", linewidth=1.5)
axes[1].set_title("First Differenced (Stationary)"); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("stationarity.png", dpi=130, bbox_inches="tight")
plt.show()

# LAG FEATURES + ROLLING STATISTICS

for lag in [1, 2, 3, 6, 12]:
    ts[f"lag_{lag}"] = ts["Deliveries"].shift(lag)

ts["rolling_mean_3"] = ts["Deliveries"].shift(1).rolling(3).mean()
ts["rolling_mean_6"] = ts["Deliveries"].shift(1).rolling(6).mean()
ts["rolling_std_3"]  = ts["Deliveries"].shift(1).rolling(3).std()
ts["rolling_std_6"]  = ts["Deliveries"].shift(1).rolling(6).std()

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(ts["Deliveries"],      label="Actual",            color="steelblue",  linewidth=1.5)
axes[0].plot(ts["rolling_mean_3"],  label="3-Month Roll Mean", color="darkorange", linewidth=1.5, linestyle="--")
axes[0].plot(ts["rolling_mean_6"],  label="6-Month Roll Mean", color="seagreen",   linewidth=1.5, linestyle="--")
axes[0].set_title("Rolling Mean vs Actual Deliveries")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(ts["rolling_std_3"], label="3-Month Roll Std", color="darkorange", linewidth=1.5)
axes[1].plot(ts["rolling_std_6"], label="6-Month Roll Std", color="seagreen",   linewidth=1.5)
axes[1].set_title("Rolling Standard Deviation")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rolling_stats.png", dpi=130, bbox_inches="tight")
plt.show()

# Lag correlation heatmap
lag_cols = ["Deliveries"] + [f"lag_{l}" for l in [1, 2, 3, 6, 12]]
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(ts[lag_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Lag Feature Correlation with Deliveries")
plt.tight_layout()
plt.savefig("lag_correlation.png", dpi=130, bbox_inches="tight")
plt.show()

# 5. CHRONOLOGICAL SPLIT

ml_df = ts.dropna().copy()

FEATURE_COLS = [f"lag_{l}" for l in [1, 2, 3, 6, 12]] + \
               ["rolling_mean_3", "rolling_mean_6", "rolling_std_3", "rolling_std_6"]
TARGET = "Deliveries"

X = ml_df[FEATURE_COLS]
y = ml_df[TARGET]

split_idx       = int(len(X) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\nChronological Split:")
print(f"  Train : {X_train.index[0].date()} → {X_train.index[-1].date()}  ({len(X_train)} months)")
print(f"  Test  : {X_test.index[0].date()}  → {X_test.index[-1].date()}   ({len(X_test)} months)")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(y_train.index, y_train.values/1e3, color="steelblue",  linewidth=2, label="Train")
ax.plot(y_test.index,  y_test.values/1e3,  color="darkorange", linewidth=2, label="Test")
ax.axvline(y_test.index[0], color="crimson", linestyle="--", linewidth=1.5, label="Split point")
ax.set(title="Chronological Train/Test Split", ylabel="Deliveries (k)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("chrono_split.png", dpi=130, bbox_inches="tight")
plt.show()

# 6. CROSS VALIDATION (TimeSeriesSplit)

tscv = TimeSeriesSplit(n_splits=5)

fig, ax = plt.subplots(figsize=(12, 4))
for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
    ax.barh(fold, len(tr_idx), color="steelblue",  alpha=0.6, label="Train" if fold==0 else "")
    ax.barh(fold, len(te_idx), left=len(tr_idx),   color="darkorange", alpha=0.8, label="Val" if fold==0 else "")
ax.set(title="TimeSeriesSplit — 5 Folds", xlabel="Number of samples", ylabel="Fold")
ax.legend(); ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("cv_folds.png", dpi=130, bbox_inches="tight")
plt.show()

print("\nCross-Validation R² — Ridge:")
scores = cross_val_score(Ridge(), X_train, y_train, cv=tscv, scoring="r2")
print(f"  Per fold : {np.round(scores, 3)}  |  mean={scores.mean():.4f}  std={scores.std():.4f}")

# HYPERPARAMETER TUNING — Ridge (GridSearchCV)

print("\nHyperparameter Tuning — Ridge...")

param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
gs = GridSearchCV(Ridge(), param_grid, cv=tscv, scoring="r2", n_jobs=-1)
gs.fit(X_train, y_train)

print(f"Best alpha : {gs.best_params_}")
print(f"Best CV R² : {gs.best_score_:.4f}")

# Tuning curve
cv_results = pd.DataFrame(gs.cv_results_)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(param_grid["alpha"], cv_results["mean_test_score"], marker="o", color="steelblue", linewidth=2)
ax.fill_between(param_grid["alpha"],
                cv_results["mean_test_score"] - cv_results["std_test_score"],
                cv_results["mean_test_score"] + cv_results["std_test_score"],
                alpha=0.2, color="steelblue")
ax.set_xscale("log")
ax.set(title="Ridge — Hyperparameter Tuning (alpha vs R²)",
       xlabel="Alpha (log scale)", ylabel="CV R²")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("hyperparameter_tuning.png", dpi=130, bbox_inches="tight")
plt.show()

# FORECASTING Ridge (tuned) + SARIMA

def evaluate(name, yt, yp):
    r2   = r2_score(yt, yp)
    rmse = mean_squared_error(yt, yp)**0.5
    mae  = mean_absolute_error(yt, yp)
    mape = mean_absolute_percentage_error(yt, yp)*100
    print(f"  {name:<30} R²={r2:.4f}  RMSE={rmse:,.0f}  MAE={mae:,.0f}  MAPE={mape:.2f}%")

print("\nTest Set Results:")

ridge_final = gs.best_estimator_
ridge_final.fit(X_train, y_train)
evaluate("Ridge (tuned)", y_test, ridge_final.predict(X_test))

sarima_fit  = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,12),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
sarima_fc   = sarima_fit.get_forecast(len(y_test))
sarima_mean = sarima_fc.predicted_mean.values
evaluate("SARIMA(1,1,1)(1,1,1,12)", y_test, sarima_mean)

# Forecast comparison plot
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(y_train.index, y_train.values/1e3, color="steelblue",  linewidth=1.5, label="Train")
ax.plot(y_test.index,  y_test.values/1e3,  color="black",      linewidth=2,   label="Actual (test)")
ax.plot(y_test.index,  ridge_final.predict(X_test)/1e3, color="darkorange", linestyle="--", linewidth=1.5, label="Ridge (tuned)")
ax.plot(y_test.index,  sarima_mean/1e3,                color="crimson",    linestyle=":",  linewidth=1.5, label="SARIMA")
ci = sarima_fc.conf_int()
ax.fill_between(y_test.index, ci.iloc[:,0]/1e3, ci.iloc[:,1]/1e3, alpha=0.15, color="crimson")
ax.axvline(y_test.index[0], color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.set(title="Forecast — Ridge vs SARIMA (k deliveries)", ylabel="Deliveries (k)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("forecast_comparison.png", dpi=130, bbox_inches="tight")
plt.show()

print("\n✔ Pipeline complete.")