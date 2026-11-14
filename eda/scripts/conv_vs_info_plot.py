import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_pickle(os.path.join(HERE, ".cache", "conv_info_df.pkl"))
labs = ["5–19", "20–59", "60–149", "150–299", "300+"]
d = d[d["info"] >= 5].copy()
d["tramo"] = pd.cut(d["info"], [5, 20, 60, 150, 300, 1e9], labels=labs, right=False)
COL = {"v2": "#1f6fb5", "v0": "#c8641b", "baseline": "#2f8f5b"}
NAME = {"v2": "tengo-aurilla-v2", "v0": "tengo-aurilla-v0", "baseline": "baseline (no usa al usuario)"}
INK, MUTED, GRID = "#17212b", "#5a6877", "#e3e8ee"
g = d[d.grupo.isin(COL)].groupby(["grupo", "tramo"], observed=True).agg(n=("buy", "size"), c=("buy", "sum")).reset_index()
g["p"] = g.c / g.n; g["se"] = np.sqrt(g.p * (1 - g.p) / g.n)
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
x = np.arange(len(labs))
for k in ("baseline", "v0", "v2"):
    s = g[g.grupo == k].set_index("tramo").reindex(labs)
    ax[0].errorbar(x, 100 * s.p, yerr=100 * s.se, color=COL[k], lw=2, marker="o", ms=7, capsize=3, elinewidth=1, label=NAME[k])
    ax[0].annotate(NAME[k].split(" ")[0], (x[-1], 100 * s.p.iloc[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
ax[0].set_ylabel("Listas que acaban en compra (%)", color=INK); ax[0].set_ylim(0, None)
ax[0].set_title("Conversión real según lo que sabíamos del usuario", loc="left", color=INK, fontsize=11, fontweight="bold")
ax[0].legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK)
b = g[g.grupo == "baseline"].set_index("tramo").reindex(labs)
for k in ("v0", "v2"):
    s = g[g.grupo == k].set_index("tramo").reindex(labs)
    r = s.p / b.p; se = r * np.sqrt((s.se / s.p) ** 2 + (b.se / b.p) ** 2)
    ax[1].errorbar(x, r, yerr=se, color=COL[k], lw=2, marker="o", ms=7, capsize=3, elinewidth=1)
    ax[1].annotate(NAME[k].split("-")[-1], (x[-1], r.iloc[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
ax[1].axhline(1, color=COL["baseline"], lw=1.2, ls=(0, (4, 3)))
ax[1].annotate("baseline", (x[0], 1), xytext=(0, 5), textcoords="offset points", color=MUTED, fontsize=9)
ax[1].set_ylabel("Veces la conversión del baseline", color=INK); ax[1].set_ylim(0, None)
ax[1].set_title("Ventaja sobre el baseline", loc="left", color=INK, fontsize=11, fontweight="bold")
for a in ax:
    a.set_xticks(x); a.set_xticklabels(labs, color=INK); a.set_xlim(-0.3, len(labs) - 0.2)
    a.set_xlabel("Eventos del usuario conocidos al llegar la petición (histórico + hoy)", color=MUTED, fontsize=9)
    a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): a.spines[sp].set_color(GRID)
    a.tick_params(colors=MUTED, length=0)
fig.text(0.01, 0.005, "Barras: ±1 error estándar. Sábado 14/11, hora UTC — v2: 19:23–20:55 · v0: 18:10–19:19 · baseline: peticiones no respondidas de todo el día.", color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
os.makedirs(os.path.join(HERE, "..", "fig"), exist_ok=True)
out = os.path.join(HERE, "..", "fig", "conversion_vs_info.png"); fig.savefig(out, dpi=130, facecolor="white"); print(out)
print(g.assign(conv=(100 * g.p).round(2), se=(100 * g.se).round(2))[["grupo", "tramo", "n", "c", "conv", "se"]].to_string(index=False))
