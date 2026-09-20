"""Finish the accuracy sweep: categoricals + the thesis check. Caches prepared data."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, re, pickle
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).parent
FPA, CACHE = ROOT/"data"/"fpa", ROOT/"data"/"prepped.pkl"

LEAK = ["FIRE_SIZE_CLASS","CONT_","MTBS","ICS_209","COMPLEX","Evacuation","GACC",
        "FIRE_NAME","FIRE_CODE","LOCAL_","FPA_ID","FOD_ID","SOURCE_",
        "NWCG_REPORTING","geometry","OBJECTID"]
W_NOW = ["tmmx","rmin","vs","pr","vpd","erc","bi","fm100","fm1000"]
W_EXT = W_NOW + ["pr_5D_mean","tmmx_5D_mean","rmin_5D_mean","vs_5D_mean","fm100_5D_mean",
    "fm1000_5D_mean","bi_5D_mean","vpd_5D_mean","erc_5D_mean","pr_5D_min","tmmx_5D_max",
    "rmin_5D_min","vs_5D_max","fm100_5D_min","fm1000_5D_min","bi_5D_max","vpd_5D_max",
    "erc_5D_max","tmmx_Percentile","vs_Percentile","fm100_Percentile","bi_Percentile",
    "vpd_Percentile","erc_Percentile","sph","th","srad","etr","tmmn",
    "sph_5D_mean","th_5D_mean","srad_5D_mean","etr_5D_mean","tmmn_5D_mean",
    "sph_5D_min","th_5D_max","srad_5D_max","etr_5D_max","tmmn_5D_max","sph_Percentile"]

if CACHE.exists():
    df, num, CATS = pickle.load(open(CACHE, "rb"))
    print(f"loaded cache: n={len(df):,}, {len(num)} numeric, {len(CATS)} categorical")
else:
    cols = list(pd.read_csv(FPA/"2020.csv", nrows=3, low_memory=False).columns)
    safe = [c for c in cols if not any(k in c for k in LEAK)]
    CATS = [c for c in ["Land_Cover","Land_Cover_1km","EVT","EVC","EVH","FRG",
            "NWCG_GENERAL_CAUSE","NWCG_CAUSE_CLASSIFICATION","OWNER_DESCR","Mang_Type",
            "Ecoregion_NA_L2CODE","Ecoregion_NA_L1CODE","STATE"] if c in safe]
    df = pd.concat([pd.read_csv(f, usecols=safe, low_memory=False)
                    for f in sorted(FPA.glob("*.csv"))], ignore_index=True)
    df = df[pd.to_numeric(df["FIRE_SIZE"], errors="coerce") > 0].reset_index(drop=True)
    df["log_size"] = np.log10(pd.to_numeric(df["FIRE_SIZE"], errors="coerce"))
    print(f"n = {len(df):,}")

    arr = [c for c in df.columns if df[c].dtype == object
           and df[c].astype(str).str.contains(r"'\s*'", regex=True, na=False).mean() > .5]
    print("parsing 12-month arrays:", arr)
    def ps(v):
        vals = [float(x) for x in re.findall(r"-?\d+\.?\d*", str(v))]
        return (np.mean(vals), np.min(vals), np.max(vals)) if vals else (np.nan,)*3
    for c in arr:
        t = df[c].map(ps)
        df[f"{c}__mean"] = [x[0] for x in t]
        df[f"{c}__min"]  = [x[1] for x in t]
        df[f"{c}__max"]  = [x[2] for x in t]
        df.drop(columns=[c], inplace=True)

    num = []
    for c in list(df.columns):
        if c in ("FIRE_SIZE","log_size") or c in CATS: continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > .55: df[c] = s; num.append(c)
        else: df.drop(columns=[c], inplace=True)
    num = [c for c in num if pd.api.types.is_numeric_dtype(df[c])]

    CATS = [c for c in CATS if c in df.columns and df[c].nunique(dropna=True) <= 250]
    for c in CATS: df[c] = df[c].astype("category")
    pickle.dump((df, num, CATS), open(CACHE, "wb"))
    print(f"cached: {len(num)} numeric, {len(CATS)} categorical -> {CATS}")

sub = df.sample(min(250000, len(df)), random_state=0).reset_index(drop=True)
y = sub["log_size"].values

def run(cols, label, cats=(), iters=600, lr=0.05, leaves=95):
    cols = [c for c in cols if c in sub.columns]
    mask = [c in cats for c in cols]
    m = HistGradientBoostingRegressor(max_iter=iters, learning_rate=lr,
        max_leaf_nodes=leaves, min_samples_leaf=40, l2_regularization=1.0,
        categorical_features=mask if any(mask) else None, random_state=0)
    sc = cross_val_score(m, sub[cols], y, cv=KFold(5, shuffle=True, random_state=0),
                         scoring="r2")
    print(f"  {label:48s} R2 = {sc.mean():+.4f} (+/-{sc.std():.4f})  [{len(cols)}]")
    return sc.mean()

print("\n--- BEST HONEST MODEL ---")
r_num = run(num, "all leak-free numeric")
r_cat = run(num + CATS, "+ categoricals (land cover, cause, ecoregion)", cats=CATS)
r_big = run(num + CATS, "+ categoricals, larger model", cats=CATS,
            iters=1000, lr=0.035, leaves=160)

print("\n--- THESIS CHECK: both sides fully expanded ---")
wx = [c for c in W_EXT if c in num]
lx = [c for c in num if c not in W_EXT] + CATS
r_w = run(wx, "weather, every weather feature available")
r_l = run(lx, "landscape / human, everything else", cats=CATS)
print(f"\n  landscape {r_l:.4f}  vs  weather {r_w:.4f}   ->  {r_l/max(r_w,1e-9):.2f}x")

pd.DataFrame([["all_numeric",r_num],["plus_categoricals",r_cat],["tuned_best",r_big],
              ["weather_full",r_w],["landscape_full",r_l]],
             columns=["features","CV_R2"]).to_csv(
    ROOT/"results"/"tables"/"table11_accuracy_ceiling.csv", index=False,
    float_format="%.4f")
print("\nwrote table11_accuracy_ceiling.csv")
