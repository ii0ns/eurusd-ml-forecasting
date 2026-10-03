import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = 'target_return'
df = pd.read_csv('processed_data.csv', index_col=0, parse_dates=True)
X = df.drop(columns=['EUR_USD', TARGET])
y = df[TARGET]

def rf():
    return RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=10,
                                 min_samples_leaf=4, max_features='sqrt',
                                 n_jobs=-1, random_state=42)

def dir_report(label, p, t):
    n = len(t)
    hits = int((np.sign(p) == np.sign(t)).sum())
    up = np.mean(t > 0)
    p0 = max(up, 1 - up)                       # best naive guess (always up or always down)
    pv = binomtest(hits, n, p0, alternative='greater').pvalue
    print(f"{label:34} acc {hits/n*100:5.2f}%  naive {p0*100:5.2f}%  "
          f"pred-up {np.mean(p > 0)*100:5.1f}%  p={pv:.4f}")

def holdout(label, cols, model):
    s = int(len(X) * 0.8)
    model.fit(X[cols].iloc[:s], y.iloc[:s])
    dir_report(label, model.predict(X[cols].iloc[s:]), y.iloc[s:].values)

exo = [c for c in X.columns if c.startswith(('Gold', 'Oil'))]
fx_only = [c for c in X.columns if c not in exo]
lag_cols = [c for c in X.columns if c.startswith('return_lag_')]

print("--- Single holdout (same 80/20 split) ---")
holdout("RF, all features", list(X.columns), rf())
holdout("RF, without Gold/Oil", fx_only, rf())
holdout("RF, Gold/Oil features only", exo, rf())
holdout("Ridge, lagged returns only", lag_cols,
        make_pipeline(StandardScaler(), Ridge(alpha=10)))

print("\n--- Walk-forward (expanding window, 6 folds, all features) ---")
for i, (tr, te) in enumerate(TimeSeriesSplit(n_splits=6).split(X), 1):
    m = rf().fit(X.iloc[tr], y.iloc[tr])
    dir_report(f"fold {i} ({X.index[te[0]].date()})", m.predict(X.iloc[te]), y.iloc[te].values)