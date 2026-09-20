#Regenerate the per-region matched 9-vs-9 breakdown and figP1
#landscape feature set (NDVI parsed), so every number on the poster comes from one protocol
#Previously the regional row said 0.083/0.231

import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle, matplotlib as mpl, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).parent
FIG, TAB = ROOT/"results"/"figures", ROOT/"results"/"tables"
EMBER, SLATE, INK = "#DC5B18", "#1F4E5F", "#292524"
mpl.rcParams.update({"figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight",
    "font.size":10.5,"axes.titlesize":12,"axes.titleweight":"bold",
    "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
    "grid.color":"#EDE3D8","grid.linewidth":0.7,"axes.axisbelow":True,
    "text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,
    "figure.facecolor":"#FFFFFF","axes.facecolor":"#FFFFFF","savefig.facecolor":"#FFFFFF"})

df, num, CATS = pickle.load(open(ROOT/"data"/"prepped.pkl", "rb"))
W9 = ["tmmx","rmin","vs","pr","vpd","erc","bi","fm100","fm1000"]
L9 = ["No_FireStation_5.0km","No_FireStation_10.0km","No_FireStation_20.0km",
      "Elevation","Slope","TRI","NDVI_mean__mean","Aridity_index","Population"]
assert all(c in df.columns for c in W9+L9)

ECO = {"8.3":"Southeastern\nUSA Plains", "9.4":"South Central\nSemi-Arid Prairies",
       "6.2":"Western\nCordillera"}

def r2(d, cols, n=250000):
    s = d.sample(min(n, len(d)), random_state=0)
    return cross_val_score(
        HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0),
        s[cols], s["log_size"].values, cv=KFold(5, shuffle=True, random_state=0),
        scoring="r2").mean()

groups = [("All US fires", df)]
for code, nice in ECO.items():
    sub = df[df["Ecoregion_NA_L2CODE"].astype(str) == code]
    if len(sub) > 20000:
        groups.append((nice, sub))

rows, labels, wv, lv = [], [], [], []
for name, d in groups:
    a, b = r2(d, W9), r2(d, L9)
    rows.append([name.replace("\n"," "), len(d), round(a,4), round(b,4), round(b/a,2)])
    labels.append(f"{name}\n(n={len(d):,})"); wv.append(a); lv.append(b)
    print(f"  {name.replace(chr(10),' '):34s} weather {a:.4f}  landscape {b:.4f}  {b/a:.2f}x")

pd.DataFrame(rows, columns=["Region","N_fires","Weather_9","Landscape_9","Ratio"]).to_csv(
    TAB/"table14_regional_matched_FINAL.csv", index=False)

fig, ax = plt.subplots(figsize=(9.2, 4.3))
x = np.arange(len(labels)); w = 0.36
b1 = ax.bar(x-w/2, wv, w, color=SLATE, label="Weather (9 variables)")
b2 = ax.bar(x+w/2, lv, w, color=EMBER, label="Landscape + firefighting access (9 variables)")
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+.008, f"{b.get_height():.3f}",
            ha="center", fontsize=9.5, fontweight="bold", color=INK)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("Share of fire-size variation explained (R²)")
ax.set_title("Landscape and access lead inside every region tested\n"
             "matched 9 variables per side", fontsize=11.5, color="#9A3412")
ax.legend(frameon=False, fontsize=9.5); ax.set_ylim(0, max(lv)*1.22)
fig.tight_layout(); fig.savefig(FIG/"figP1_variance_explained.png"); plt.close(fig)
print("\nwrote table14_regional_matched_FINAL.csv + regenerated figP1")
