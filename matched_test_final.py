
#Emit the authoritative matched 9-vs-9 table (post NDVI bug-fix) + the three
#escape thresholds for BOTH feature sets, so every number on the poster is backed
#by a saved table. Using data/prepped.pkl written by accuracy_final.py.

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent
df, num, CATS = pickle.load(open(ROOT/"data"/"prepped.pkl", "rb"))
print(f"n = {len(df):,}")

W9 = ["tmmx","rmin","vs","pr","vpd","erc","bi","fm100","fm1000"]
L9 = ["No_FireStation_5.0km","No_FireStation_10.0km","No_FireStation_20.0km",
      "Elevation","Slope","TRI","NDVI_mean__mean","Aridity_index","Population"]
W9 = [c for c in W9 if c in df.columns]
L9 = [c for c in L9 if c in df.columns]
assert len(W9) == 9 and len(L9) == 9, (len(W9), len(L9))
print(f"matched sets: {len(W9)} weather vs {len(L9)} landscape/access")

sub = df.sample(min(250000, len(df)), random_state=0).reset_index(drop=True)

def r2(cols):
    return cross_val_score(
        HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
            max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
            random_state=0),
        sub[cols], sub["log_size"].values,
        cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean()

def auc(cols, y):
    kf = StratifiedKFold(5, shuffle=True, random_state=0); out=[]
    for tr, te in kf.split(sub[cols], y):
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                max_leaf_nodes=63, min_samples_leaf=40, random_state=0)
        m.fit(sub[cols].iloc[tr], y[tr])
        out.append(roc_auc_score(y[te], m.predict_proba(sub[cols].iloc[te])[:,1]))
    return float(np.mean(out))

rows = []
rw, rl = r2(W9), r2(L9)
rows.append(["final size (log10 acres)", "R2", np.nan, round(rw,4), round(rl,4),
             round(rl/rw, 2)])
print(f"\nFINAL SIZE      R2  weather {rw:.4f}  landscape {rl:.4f}  -> {rl/rw:.2f}x")

for label, thr in [("escapes (>=10 acres)",10), ("large (>=100 acres)",100),
                   ("major (>=1000 acres)",1000)]:
    y = (sub.FIRE_SIZE >= thr).astype(int).values
    aw, al = auc(W9, y), auc(L9, y)
    rows.append([label, "AUC", round(y.mean(),4), round(aw,4), round(al,4),
                 round(al/aw, 2)])
    print(f"{label:22s} AUC weather {aw:.4f}  landscape {al:.4f}  "
          f"(base rate {y.mean()*100:.1f}%)")

out = pd.DataFrame(rows, columns=["Outcome","Metric","Base_rate",
                                  "Weather_9","Landscape_9","Ratio"])
out.to_csv(ROOT/"results"/"tables"/"table12_matched_9v9_FINAL.csv", index=False)
print("\nwrote table12_matched_9v9_FINAL.csv  (authoritative, post NDVI fix)")
print(out.to_string(index=False))
