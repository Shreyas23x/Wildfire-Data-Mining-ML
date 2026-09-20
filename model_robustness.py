#Rerunning the matched 9-vs-9 final-size contest with three different model families.

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).parent
df, num, CATS = pickle.load(open(ROOT/"data"/"prepped.pkl", "rb"))

W9 = ["tmmx","rmin","vs","pr","vpd","erc","bi","fm100","fm1000"]
L9 = ["No_FireStation_5.0km","No_FireStation_10.0km","No_FireStation_20.0km",
      "Elevation","Slope","TRI","NDVI_mean__mean","Aridity_index","Population"]

sub = df.sample(80000, random_state=0).reset_index(drop=True)
y = sub["log_size"].values

MODELS = {
 "Gradient boosting (used for all headline numbers)":
    lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
              max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0, random_state=0),
 "Random forest":
    lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("m", RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                                  random_state=0, n_jobs=-1))]),
 "Ridge regression (linear)":
    lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
}

print(f"n = {len(sub):,}   matched 9 vs 9, predicting log10 final fire size\n")
rows = []
for name, mk in MODELS.items():
    cv = KFold(5, shuffle=True, random_state=0)
    rw = cross_val_score(mk(), sub[W9], y, cv=cv, scoring="r2").mean()
    rl = cross_val_score(mk(), sub[L9], y, cv=cv, scoring="r2").mean()
    rows.append([name, round(rw,4), round(rl,4), round(rl/rw,2), rl > rw])
    print(f"  {name:52s} weather {rw:+.4f}   landscape {rl:+.4f}   {rl/rw:.2f}x")

out = pd.DataFrame(rows, columns=["Model","Weather_9","Landscape_9","Ratio","Landscape_wins"])
out.to_csv(ROOT/"results"/"tables"/"table15_model_robustness.csv", index=False)
print(f"\nlandscape wins in {out.Landscape_wins.sum()}/{len(out)} model families")
print("wrote table15_model_robustness.csv")
