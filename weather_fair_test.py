#Maximizing weather features/test for fairness.

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent
FPA  = ROOT / "data" / "fpa"

W_NOW  = ["tmmx","rmin","vs","pr","vpd","erc","bi","fm100","fm1000"]
W_5D   = ["pr_5D_mean","tmmx_5D_mean","rmin_5D_mean","vs_5D_mean","fm100_5D_mean",
          "fm1000_5D_mean","bi_5D_mean","vpd_5D_mean","erc_5D_mean",
          "pr_5D_min","tmmx_5D_max","rmin_5D_min","vs_5D_max","fm100_5D_min",
          "fm1000_5D_min","bi_5D_max","vpd_5D_max","erc_5D_max"]
W_PCT  = ["tmmx_Percentile","vs_Percentile","fm100_Percentile","bi_Percentile",
          "vpd_Percentile","erc_Percentile"]
W_FULL = W_NOW + W_5D + W_PCT
NONW   = ["No_FireStation_5.0km","No_FireStation_10.0km","No_FireStation_20.0km",
          "Elevation","Slope","TRI","NDVI_mean","Aridity_index","Population"]
KEEP   = ["FIRE_SIZE"] + W_FULL + NONW

df = pd.concat([pd.read_csv(f, usecols=lambda c: c in KEEP, low_memory=False)
                for f in sorted(FPA.glob("*.csv"))], ignore_index=True)
have = [c for c in W_FULL if c in df.columns]
print(f"weather features available: {len(have)}/{len(W_FULL)}")
for c in have + NONW + ["FIRE_SIZE"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df[df.FIRE_SIZE > 0].reset_index(drop=True)
df["log_size"] = np.log10(df.FIRE_SIZE)
print(f"n = {len(df):,}\n")

sub = df.sample(min(150000, len(df)), random_state=0)

def r2(cols, label):
    sc = cross_val_score(HistGradientBoostingRegressor(max_iter=250, random_state=0),
                         sub[cols], sub["log_size"].values, cv=5, scoring="r2")
    print(f"  {label:44s} R2 = {sc.mean():+.4f} (+/-{sc.std():.4f})  [{len(cols)} feats]")
    return sc.mean()

print("FINAL SIZE (log10 acres)")
a = r2([c for c in W_NOW if c in df.columns],  "weather, ignition day only (9)")
b = r2(have,                                    f"weather, FULL fair shot ({len(have)})")
c = r2(NONW,                                    "landscape + firefighting access (9)")
d = r2(have + NONW,                             "everything combined")

print(f"\n  giving weather 5-day windows + percentiles: {a:.4f} -> {b:.4f} "
      f"(+{b-a:.4f})")
print(f"  landscape still ahead by {c-b:+.4f} R2  ({c/b:.2f}x)")

print("\nESCAPE (>= 10 acres), AUC")
y = (sub.FIRE_SIZE >= 10).astype(int).values
def auc(cols, label):
    kf = StratifiedKFold(5, shuffle=True, random_state=0); out=[]
    for tr, te in kf.split(sub[cols], y):
        m = HistGradientBoostingClassifier(max_iter=250, random_state=0)
        m.fit(sub[cols].iloc[tr], y[tr])
        out.append(roc_auc_score(y[te], m.predict_proba(sub[cols].iloc[te])[:,1]))
    print(f"  {label:44s} AUC = {np.mean(out):.4f}")
    return np.mean(out)
auc([c for c in W_NOW if c in df.columns], "weather, ignition day only")
auc(have,                                   "weather, FULL fair shot")
auc(NONW,                                   "landscape + firefighting access")

pd.DataFrame([{"features":"weather_ignition_only","n":9,"R2":a},
              {"features":"weather_full_5D_percentiles","n":len(have),"R2":b},
              {"features":"landscape_access","n":9,"R2":c},
              {"features":"combined","n":len(have)+9,"R2":d}]).to_csv(
    ROOT/"results"/"tables"/"table10_weather_fair_test.csv", index=False,
    float_format="%.4f")
print("\nwrote table10_weather_fair_test.csv")
