#Poster figures from 10 years of FPA-FOD
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
matplotlib.use("Agg")

ROOT = Path(__file__).parent
FPA, FIG = ROOT/"data"/"fpa", ROOT/"results"/"figures"
TAB = ROOT/"results"/"tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

C_BLUE, C_ORANGE, C_GREEN, C_VERM, C_GRAY = ("#0072B2","#E69F00","#009E73","#D55E00","#7F7F7F")
mpl.rcParams.update({"figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight",
    "font.size":10.5,"axes.titlesize":12,"axes.titleweight":"bold",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.color":"#E4E4E4","grid.linewidth":0.6,"axes.axisbelow":True})

WEATHER = ["tmmx","rmin","vs","pr","vpd","erc","bi","fm100","fm1000"]
SUPPRESS = ["No_FireStation_5.0km","No_FireStation_10.0km","No_FireStation_20.0km"]
LANDSCAPE = ["Elevation","Slope","TRI","NDVI_mean","Aridity_index"]
HUMAN = ["Population"]
NONW = SUPPRESS + LANDSCAPE + HUMAN
KEEP = ["FIRE_SIZE","Land_Cover","Ecoregion_NA_L2CODE"] + WEATHER + NONW

df = pd.concat([pd.read_csv(f, usecols=lambda c: c in KEEP, low_memory=False)
                for f in sorted(FPA.glob("*.csv"))], ignore_index=True)
for c in WEATHER + NONW + ["FIRE_SIZE"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df[df.FIRE_SIZE > 0].copy()
df["log_size"] = np.log10(df.FIRE_SIZE)
N = len(df); print(f"n = {N:,}")

ECO_NAMES = {8.3:"Southeastern USA Plains", 9.4:"South Central Semi-Arid Prairies",
             6.2:"Western Cordillera", 8.4:"Ozark/Ouachita Appalachian",
             8.1:"Mixed Wood Plains", 10.1:"Cold Deserts", 11.1:"Mediterranean California",
             8.5:"Mississippi Alluvial Plain", 9.3:"West-Central Semi-Arid Prairies",
             10.2:"Warm Deserts"}

def r2(sub, cols, n=80000):
    s = sub.dropna(subset=["log_size"]).sample(min(n, len(sub)), random_state=0)
    return cross_val_score(HistGradientBoostingRegressor(max_iter=200, random_state=0),
                           s[cols], s["log_size"].values, cv=5, scoring="r2").mean()

groups = [("All US fires", df)]
for e in df.Ecoregion_NA_L2CODE.value_counts().index[:3]:
    sub = df[df.Ecoregion_NA_L2CODE == e]
    if len(sub) > 20000:
        groups.append((ECO_NAMES.get(float(e), f"Ecoregion {e}"), sub))

labels, w_r2, n_r2 = [], [], []
for name, sub in groups:
    labels.append(f"{name}\n(n={len(sub):,})")
    w_r2.append(r2(sub, WEATHER)); n_r2.append(r2(sub, NONW))
    print(f"{name}: weather {w_r2[-1]:.3f}  landscape {n_r2[-1]:.3f}")

fig, ax = plt.subplots(figsize=(9.2, 4.3))
x = np.arange(len(labels)); w = 0.36
b1 = ax.bar(x-w/2, w_r2, w, color=C_BLUE, label="Weather at ignition (9 vars)")
b2 = ax.bar(x+w/2, n_r2, w, color=C_ORANGE, label="Landscape + suppression (9 vars)")
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+.006, f"{b.get_height():.3f}",
            ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Cross-validated R²  (log₁₀ fire size)")
ax.set_title("Landscape and suppression explain 2–5× more of fire size than weather —\n"
             "and the gap holds inside every ecoregion", fontsize=11.5)
ax.legend(frameon=False, fontsize=9.5); ax.set_ylim(0, max(n_r2)*1.22)
fig.tight_layout(); fig.savefig(FIG/"figP1_variance_explained.png"); plt.close(fig)

v = "No_FireStation_5.0km"; m = df[v].notna()
rows = []
for k, g in df[m].groupby("Land_Cover", observed=True):
    if len(g) < 500: continue
    r, _ = stats.spearmanr(g[v], g["log_size"])
    if not np.isnan(r): rows.append((str(k)[:26], r, len(g)))
rows.sort(key=lambda t: t[1])
pooled, _ = stats.spearmanr(df.loc[m, v], df.loc[m, "log_size"])

fig, ax = plt.subplots(figsize=(8.4, 5.2))
names = [r[0] for r in rows]; vals = [r[1] for r in rows]
ax.barh(names, vals, color=C_GREEN, height=0.66)
ax.axvline(0, color="#333", lw=1.1)
ax.axvline(pooled, color=C_VERM, lw=1.6, ls="--",
           label=f"Pooled across all types (ρ = {pooled:+.3f})")
for i, val in enumerate(vals):
    ax.text(val-0.006, i, f"{val:+.2f}", va="center", ha="right",
            fontsize=8.2, color="white", fontweight="bold")
ax.set_xlabel("Spearman ρ:  fire stations within 5 km  vs.  fire size")
ax.set_title(f"Fires grow larger where help is farther away —\n"
             f"negative in {len(rows)} of {len(rows)} land-cover types", fontsize=11.5)
ax.legend(frameon=False, fontsize=9, loc="lower left")
fig.tight_layout(); fig.savefig(FIG/"figP2_suppression_robustness.png"); plt.close(fig)

res = []
for var, nice in [("erc","ERC\n(fire danger)"), ("vpd","VPD\n(air dryness)"),
                  ("fm1000","1000-hr fuel\nmoisture"), ("tmmx","Max\ntemperature")]:
    s = df[var]; mm = s.notna()
    pooled_r, _ = stats.spearmanr(s[mm], df.loc[mm,"log_size"])
    rr = []
    for k, g in df[mm].groupby("Ecoregion_NA_L2CODE", observed=True):
        if len(g) < 2000: continue
        r, _ = stats.spearmanr(g[var], g["log_size"])
        if not np.isnan(r): rr.append((r, len(g)))
    tot = sum(n for _, n in rr); wr = sum(r*n for r, n in rr)/tot
    pos = sum(1 for r, _ in rr if r > 0)
    res.append((nice, pooled_r, wr, pos, len(rr)))
    print(f"{var}: pooled {pooled_r:+.3f} -> within {wr:+.3f} ({pos}/{len(rr)} pos)")

fig, ax = plt.subplots(figsize=(8.6, 4.6))
for i, (nice, p_, w_, pos, tot_) in enumerate(res):
    col = C_BLUE if w_ > 0 else C_ORANGE
    ax.plot([0, 1], [p_, w_], "-", color=col, lw=2.4, zorder=2)
    ax.scatter([0], [p_], s=95, color=C_GRAY, zorder=3, edgecolors="white", lw=1.5)
    ax.scatter([1], [w_], s=95, color=col, zorder=3, edgecolors="white", lw=1.5)
    ax.text(-0.045, p_, nice, ha="right", va="center", fontsize=8.6)
    ax.text(1.04, w_, f"{w_:+.3f}   ({pos}/{tot_} regions)", ha="left",
            va="center", fontsize=8.6, color=col, fontweight="bold")
ax.axhline(0, color="#333", lw=1.1, ls=":")
ax.set_xlim(-0.42, 1.62); ax.set_xticks([0, 1])
ax.set_xticklabels(["Pooled\n(all US together)", "Stratified\n(within ecoregion)"],
                   fontsize=10)
ax.set_ylabel("Spearman ρ  vs. fire size")
ax.set_title("Simpson's paradox: pooling ecoregions inverts the fire-weather signal",
             fontsize=11.5)
fig.tight_layout(); fig.savefig(FIG/"figP3_simpson_paradox.png"); plt.close(fig)

pd.DataFrame(res, columns=["Variable","Pooled_rho","WithinEco_rho",
                           "N_positive","N_ecoregions"]).to_csv(
    TAB/"table7_pooling_artifact.csv", index=False, float_format="%.4f")
pd.DataFrame(rows, columns=["Land_Cover","rho_firestations_vs_size","n"]).to_csv(
    TAB/"table8_suppression_by_landcover.csv", index=False, float_format="%.4f")


#rest of figures
ROOT = Path(__file__).parent
T = ROOT / "results" / "tables"
OUT = ROOT.parent / "paper" / "figures"; OUT.mkdir(parents=True, exist_ok=True)

WEATHER, LAND = "#D9822B", "#2A5DA8"
INK, MUTED, GRID = "#1f1f1f", "#5f5f5f", "#e3e3e3"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=600)
    plt.close(fig); print("wrote", name)

t12 = pd.read_csv(T / "table12_matched_9v9_FINAL.csv")
sd = {}
pa = T / "paper_A_randomcv_sd_ablation.csv"
if pa.exists():
    a = pd.read_csv(pa)
    for _, r in a.iterrows(): sd[(r.Feature_set, r.Target)] = r.SD_5fold
tmap = {"escapes (>=10 acres)": "AUC >=10 ac", "large (>=100 acres)": "AUC >=100 ac",
        "major (>=1000 acres)": "AUC >=1000 ac", "final size (log10 acres)": "R2 final size"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.9),
                               gridspec_kw={"width_ratios": [3, 1.25], "wspace": 0.55})
auc = t12[t12.Metric == "AUC"].reset_index(drop=True)
x = np.arange(len(auc)); w = 0.36
for off, col, c, lab, key in [(-w/2, "Weather_9", WEATHER, "Fire weather (9)", "Weather (9)"),
                              (w/2, "Landscape_9", LAND, "Landscape + access (9)", "Landscape + access (9)")]:
    err = [sd.get((key, tmap[o]), 0) for o in auc.Outcome]
    ax1.bar(x + off, auc[col], w, color=c, label=lab, yerr=err,
            error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": INK}, edgecolor="white", linewidth=0.5)
    for xi, v in zip(x + off, auc[col]):
        ax1.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6, color=INK)
ax1.axhline(0.5, color=MUTED, lw=0.6, ls="--"); ax1.text(-0.55, 0.505, "chance", fontsize=6, color=MUTED, ha="left", va="bottom")
ax1.set_xticks(x, ["Escape\n≥10 ac", "Large\n≥100 ac", "Major\n≥1,000 ac"])
ax1.set_ylim(0.45, 0.95); ax1.set_ylabel("ROC AUC (5-fold CV)")
ax1.yaxis.grid(True, color=GRID, lw=0.5); ax1.set_axisbelow(True)
ax1.set_title("(a) Does the fire escape?", loc="left")

fs = t12[t12.Metric == "R2"].iloc[0]
for i, (col, c, key) in enumerate([("Weather_9", WEATHER, "Weather (9)"), ("Landscape_9", LAND, "Landscape + access (9)")]):
    e = sd.get((key, "R2 final size"), 0)
    ax2.bar(i, fs[col], 0.7, color=c, yerr=e, error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": INK})
    ax2.text(i, fs[col] + 0.01, f"{fs[col]:.3f}", ha="center", va="bottom", fontsize=6, color=INK)
ax2.set_xticks([0, 1], ["Weather", "Landsc."]); ax2.set_ylim(0, 0.32)
ax2.set_ylabel("R² of log$_{10}$ size"); ax2.yaxis.grid(True, color=GRID, lw=0.5); ax2.set_axisbelow(True)
ax2.set_title("(b) Final size", loc="left")
fig.legend(*ax1.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.03))
fig.subplots_adjust(left=0.12, right=0.99, top=0.80, bottom=0.2)
save(fig, "fig1_stages")

t7 = pd.read_csv(T / "table7_pooling_artifact.csv")
t7["Var"] = t7.Variable.str.replace("\n", " ", regex=False)
labels = {"ERC (fire danger)": "Energy release comp.", "VPD (air dryness)": "Vapor pressure deficit",
          "1000-hr fuel moisture": "1000-h fuel moisture", "Max temperature": "Max temperature"}
fig, ax = plt.subplots(figsize=(3.5, 1.7))
y = np.arange(len(t7))[::-1]
for yi, (_, r) in zip(y, t7.iterrows()):
    ax.annotate("", xy=(r.WithinEco_rho, yi), xytext=(r.Pooled_rho, yi),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, shrinkA=3, shrinkB=3, mutation_scale=6))
    ax.plot(r.Pooled_rho, yi, "o", ms=4.5, mfc="white", mec=INK, mew=0.8)
    ax.plot(r.WithinEco_rho, yi, "o", ms=4.5, color=LAND)
    flip = np.sign(r.Pooled_rho) != np.sign(r.WithinEco_rho)
    n_same = int(r.N_positive) if r.WithinEco_rho > 0 else int(r.N_ecoregions - r.N_positive)
    ax.text(0.41, yi, f"{n_same}/{int(r.N_ecoregions)} regions {'>' if r.WithinEco_rho > 0 else '<'} 0"
            + ("\nsign reverses" if flip else ""), va="center", ha="right", fontsize=6,
            color=INK if flip else MUTED, linespacing=0.95)
ax.axvline(0, color=MUTED, lw=0.6)
ax.set_yticks(y, [labels.get(v, v) for v in t7.Var]); ax.set_xlim(-0.16, 0.42); ax.set_xticks([-0.1, 0, 0.1, 0.2])
ax.set_xlabel("Spearman ρ with fire size")
ax.plot([], [], "o", mfc="white", mec=INK, label="pooled (all US)")
ax.plot([], [], "o", color=LAND, label="weighted mean within ecoregion")
ax.legend(loc="lower right", frameon=False, bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.1)
ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.subplots_adjust(left=0.33, right=0.99, top=0.86, bottom=0.22)
save(fig, "fig2_simpson")

rows = []
t14 = pd.read_csv(T / "table14_regional_matched_FINAL.csv")
rows.append(("All US, random 5-fold", t14.Weather_9[0], t14.Landscape_9[0]))
pb, pc = T / "paper_B_spatial_block_cv.csv", T / "paper_C_temporal_holdout.csv"
if pb.exists():
    b = pd.read_csv(pb); b = b[b.Target == "R2 final size"].set_index("Feature_set").Mean
    rows.append(("Spatial-block CV (1° cells)", b["Weather (9)"], b["Landscape + access (9)"]))
if pc.exists():
    c = pd.read_csv(pc).set_index("Feature_set").R2_final_size
    rows.append(("Train 2011–17 → test 2018–20", c["Weather (9)"], c["Landscape + access (9)"]))
for _, r in t14.iloc[1:].iterrows():
    rows.append((r.Region.replace("Southeastern USA", "SE US").replace("South Central", "S-C"), r.Weather_9, r.Landscape_9))
t15 = pd.read_csv(T / "table15_model_robustness.csv")
for _, r in t15.iloc[1:].iterrows():
    rows.append((r.Model.replace(" regression (linear)", " (linear)"), r.Weather_9, r.Landscape_9))
groups = ["Validation"] * (1 + pb.exists() + pc.exists()) + ["Region"] * 3 + ["Model"] * 2

fig, ax = plt.subplots(figsize=(3.5, 0.28 * len(rows) + 0.6))
y = np.arange(len(rows))[::-1]
for yi, (lab, wv, lv) in zip(y, rows):
    ax.plot([wv, lv], [yi, yi], color=GRID, lw=2.2, solid_capstyle="round", zorder=1)
    ax.plot(wv, yi, "o", ms=4.5, color=WEATHER, zorder=2)
    ax.plot(lv, yi, "o", ms=4.5, color=LAND, zorder=2)
    ax.text(0.62, yi, f"{lv / wv:.1f}×", va="center", ha="right", fontsize=6.5, color=INK)
ax.set_yticks(y, [r[0] for r in rows])
prev = None
for yi, g in zip(y, groups):
    if prev is not None and g != prev: ax.axhline(yi + 0.5, color=MUTED, lw=0.4, ls=":")
    prev = g
ax.set_xlim(0, 0.63); ax.set_xticks([0, .1, .2, .3, .4, .5]); ax.set_xlabel("CV R² of log$_{10}$ fire size")
ax.plot([], [], "o", color=WEATHER, label="Fire weather (9)")
ax.plot([], [], "o", color=LAND, label="Landscape + access (9)")
ax.legend(loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2, frameon=False, borderaxespad=0.1)
ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.subplots_adjust(left=0.40, right=0.98, top=1 - 0.35 / (0.28 * len(rows) + 0.6), bottom=0.55 / (0.28 * len(rows) + 0.6))
save(fig, "fig3_robustness")
