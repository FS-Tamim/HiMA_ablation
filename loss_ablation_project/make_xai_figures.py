# ============================================================
# Paper figures for the LIME / SHAP analysis, built from the saved results (no model re-run):
#   fig_top_words_shap_lime.png  - top words per predicted class, SHAP and LIME side by side
#   fig_method_agreement.png     - (a) LIME vs SHAP score of every word, (b) per-meme agreement,
#                                  words vs image regions
#   fig_examples_shap.png        - one correctly classified meme per class: meme, SHAP image map,
#                                  caption coloured by SHAP
# Figures with Bengali text are drawn by headless Chrome (matplotlib cannot shape Bengali).
#
# Usage:  python make_xai_figures.py      Options: --run-dir ablation_suite_outputs/xai/ce_seed42
#                                                  --top 10   --examples 20,44,...  (one per class)
# Output: <run-dir>/lime_shap/paper_figures/
# ============================================================
import os, argparse, base64, html, io, subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import spearmanr, pearsonr

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FONT = "'Kohinoor Bangla','Noto Sans Bengali','Bangla MN',sans-serif"
C_SHAP, C_LIME = "#2a78d6", "#eb6834"          # categorical slots 1 and 2; also used for words / regions
C_INK, C_INK2, C_MUTED, C_GRID, C_AXIS, C_SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
NICE = {"gendered aggression": "Gendered aggression", "political aggression": "Political aggression",
        "religious aggression": "Religious aggression", "others": "Others"}
ORDER = [("L1", "Aggressive"), ("L1", "Non-Aggressive"), ("L2", "gendered aggression"),
         ("L2", "political aggression"), ("L2", "religious aggression"), ("L2", "others")]

def nice(c):
    return NICE.get(c, c)

def shoot(page, png, width, height):
    tmp = png[:-4] + ".html"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(page)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=3",
                    f"--window-size={width},{height}", f"--screenshot={png}", "file://" + os.path.abspath(tmp)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(tmp)

def style_axes(ax):
    ax.set_facecolor(C_SURF)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(C_AXIS)
    ax.tick_params(colors=C_MUTED, length=0, labelsize=9)
    ax.grid(color=C_GRID, lw=0.6); ax.set_axisbelow(True)

# ---------------- figure 1: top words, SHAP vs LIME ----------------
def fig_top_words(W, out, top):
    W = W.copy()
    W["w"] = W["word"].astype(str).str.strip().str.lower()
    W = W[W["w"].map(lambda s: len(s) > 1 and any(ch.isalpha() for ch in s))]
    G = W.groupby(["level", "predicted_class", "w"]).agg(shap=("shap", "mean"), lime=("lime", "mean"),
                                                          n=("shap", "size")).reset_index()
    G = G[G["n"] >= 3]
    vmax = max(G["shap"].max(), G["lime"].max())
    panels = []
    for lv, cls in ORDER:
        g = G[(G["level"] == lv) & (G["predicted_class"] == cls)].sort_values("shap", ascending=False).head(top)
        rows = []
        for r in g.itertuples():
            rows.append(
                f'<div class="r"><div class="w">{html.escape(r.w)}</div><div class="bars">'
                f'<div class="b" style="width:{max(r.shap, 0) / vmax * 100:.1f}%;background:{C_SHAP}"></div>'
                f'<span class="v">{r.shap:.2f}</span>'
                f'<div class="b" style="width:{max(r.lime, 0) / vmax * 100:.1f}%;background:{C_LIME}"></div>'
                f'<span class="v">{r.lime:.2f}</span></div></div>')
        panels.append(f'<div class="p"><h3>{html.escape(nice(cls))}</h3><p>Level {lv[1]} · n = memes predicted as this class'
                      f'</p>{"".join(rows)}</div>')
    css = f"""
body{{margin:0;background:{C_SURF};font-family:system-ui,{FONT};color:{C_INK}}}
.wrap{{padding:14px 18px}} .lg{{font-size:13px;color:{C_INK2};margin:0 0 10px}}
.lg i{{display:inline-block;width:12px;height:12px;border-radius:2px;margin:0 5px -1px 12px}}
.grid{{display:grid;grid-template-columns:repeat(3,330px);gap:18px 26px}}
.p h3{{margin:0;font-size:15px}} .p p{{margin:0 0 6px;font-size:11px;color:{C_MUTED}}}
.r{{display:grid;grid-template-columns:96px 1fr;align-items:center;margin:3px 0}}
.w{{font-size:15px;text-align:right;padding-right:8px;white-space:nowrap;overflow:hidden}}
.bars{{display:grid;grid-template-columns:1fr 34px;row-gap:2px;align-items:center}}
.b{{height:8px;border-radius:0 4px 4px 0;min-width:1px}} .v{{font-size:10px;color:{C_INK2};padding-left:4px}}
"""
    page = (f'<!doctype html><html lang="bn"><meta charset="utf-8"><style>{css}</style><div class="wrap">'
            f'<p class="lg">Mean attribution to the predicted class (words seen ≥ 3 times; top {top} by SHAP)'
            f'<i style="background:{C_SHAP}"></i>SHAP<i style="background:{C_LIME}"></i>LIME</p>'
            f'<div class="grid">{"".join(panels)}</div></div></html>')
    shoot(page, os.path.join(out, "fig_top_words_shap_lime.png"), 3 * 330 + 2 * 26 + 36, 2 * (40 + top * 34) + 90)

# ---------------- figure 2: method agreement ----------------
def fig_agreement(W, A, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=C_SURF,
                                 gridspec_kw=dict(width_ratios=[1, 1.15]))
    x, y = W["shap"].to_numpy(), W["lime"].to_numpy()
    lim = np.quantile(np.abs(np.r_[x, y]), 0.995)
    style_axes(a1)
    a1.scatter(x, y, s=5, alpha=0.18, color=C_SHAP, linewidths=0)
    a1.plot([-lim, lim], [-lim, lim], color=C_AXIS, lw=1, ls="--")
    a1.axhline(0, color=C_AXIS, lw=0.8); a1.axvline(0, color=C_AXIS, lw=0.8)
    a1.set_xlim(-lim, lim); a1.set_ylim(-lim, lim)
    a1.set_xlabel("SHAP value of the word", color=C_INK, fontsize=10)
    a1.set_ylabel("LIME weight of the word", color=C_INK, fontsize=10)
    r_p, r_s = pearsonr(x, y)[0], spearmanr(x, y)[0]
    a1.set_title(f"(a) Every caption word, both levels (n = {len(x):,})", loc="left", fontsize=10.5, color=C_INK)
    a1.text(0.03, 0.95, f"Pearson r = {r_p:.2f}\nSpearman ρ = {r_s:.2f}", transform=a1.transAxes, va="top",
            fontsize=9.5, color=C_INK2)

    style_axes(a2); a2.grid(axis="x", visible=False)
    groups = [("L1", "text_spearman", "Words", C_SHAP), ("L1", "image_spearman", "Image regions", C_LIME),
              ("L2", "text_spearman", "Words", C_SHAP), ("L2", "image_spearman", "Image regions", C_LIME)]
    pos = [0, 0.75, 2, 2.75]
    for p, (lv, col, lab, c) in zip(pos, groups):
        v = A.loc[A["level"] == lv, col].dropna().to_numpy()
        bp = a2.boxplot(v, positions=[p], widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color=C_SURF, lw=2), whiskerprops=dict(color=C_AXIS),
                        capprops=dict(color=C_AXIS), boxprops=dict(facecolor=c, edgecolor=c))
        jit = np.random.default_rng(0).uniform(-0.18, 0.18, len(v))
        a2.scatter(p + jit, v, s=4, color=c, alpha=0.25, linewidths=0, zorder=0)
        a2.text(p, 1.04, f"{np.median(v):.2f}", ha="center", fontsize=9, color=C_INK2)
    a2.set_xticks([0.375, 2.375]); a2.set_xticklabels(["Level 1", "Level 2"], fontsize=10, color=C_INK)
    a2.set_ylim(-0.6, 1.12); a2.set_ylabel("Spearman ρ between LIME and SHAP (per meme)", color=C_INK, fontsize=10)
    a2.set_title("(b) Per-meme agreement (median above each box)", loc="left", fontsize=10.5, color=C_INK)
    from matplotlib.patches import Patch
    a2.legend(handles=[Patch(color=C_SHAP, label="Words"), Patch(color=C_LIME, label="Image regions")],
              frameon=False, loc="lower left", fontsize=9, labelcolor=C_INK)
    plt.tight_layout(w_pad=2.5)
    plt.savefig(os.path.join(out, "fig_method_agreement.png"), dpi=200); plt.close()
    return r_p, r_s

# ---------------- figure 3: worked examples ----------------
COLS = {"meme": (16, 315), "shap": (645, 945)}

def crop(fig_png, row, n_rows, col):
    im = Image.open(fig_png).convert("RGB")
    y0, y1 = (25, 325) if n_rows == 1 else ((35, 335), (371, 671))[row]
    x0, x1 = COLS[col]
    buf = io.BytesIO(); im.crop((x0, y0, x1, y1)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def pick_examples(P, W, l2_names):
    n_words = W[W["level"] == "L1"].groupby("idx").size()
    P = P.assign(nw=P["idx"].map(n_words).fillna(0))
    good = P[(P["nw"].between(5, 16))]
    picks = []
    g = good[(good["true1"] == 0) & (good["pred1"] == 0)].sort_values("conf1", ascending=False)
    picks.append(("Non-Aggressive", "L1", int(g.iloc[0]["idx"])))
    for c, name in enumerate(l2_names):
        g = good[(good["true1"] == 1) & (good["pred1"] == 1) & (good["true2"] == c) & (good["pred2"] == c)]
        g = g.sort_values("conf2", ascending=False)
        if len(g):
            picks.append((name, "L2", int(g.iloc[0]["idx"])))
    return picks

def colour_words(words, scores):
    scale = float(np.abs(scores).max()) or 1.0
    out = []
    for w, v in zip(words, scores):
        a = min(abs(v) / scale, 1.0) * 0.85
        rgb = "235,104,52" if v > 0 else "42,120,214"
        out.append(f'<span style="background:rgba({rgb},{a:.2f})">{html.escape(str(w))}</span>')
    return " ".join(out)

def fig_examples(P, W, ls, out, l2_names, forced):
    picks = pick_examples(P, W, l2_names)
    if forced:
        picks = [(p[0], p[1], i) for p, i in zip(picks, forced)]
    rows = []
    for name, lv, i in picks:
        r = P[P["idx"] == i].iloc[0]
        two = int(r["pred1"]) == 1
        fig_png = os.path.join(ls, "figures", f"sample_{i:04d}.png")
        row = 1 if (lv == "L2" and two) else 0
        g = W[(W["idx"] == i) & (W["level"] == lv)].sort_values("position")
        conf = r["conf2"] if lv == "L2" else r["conf1"]
        rows.append(
            f'<div class="ex"><div class="lab"><b>{html.escape(nice(name))}</b><br>Level {lv[1]} · p = {conf:.2f}'
            f'<br><span>test sample {i}</span></div>'
            f'<img src="{crop(fig_png, row, 2 if two else 1, "meme")}"><img src="{crop(fig_png, row, 2 if two else 1, "shap")}">'
            f'<div class="cap">{colour_words(g["word"].tolist(), g["shap"].to_numpy())}</div></div>')
    css = f"""
body{{margin:0;background:{C_SURF};font-family:system-ui,{FONT};color:{C_INK}}}
.wrap{{padding:14px 18px}} .lg{{font-size:13px;color:{C_INK2};margin:0 0 8px}}
.lg i{{display:inline-block;width:12px;height:12px;border-radius:2px;margin:0 5px -1px 12px}}
.hd,.ex{{display:grid;grid-template-columns:150px 170px 170px 1fr;gap:12px;align-items:center}}
.hd{{font-size:12px;color:{C_MUTED};margin-bottom:4px}} .ex{{border-top:1px solid {C_GRID};padding:8px 0}}
.ex img{{width:170px;height:170px;border-radius:4px}}
.lab{{font-size:13px;line-height:1.45}} .lab span{{color:{C_MUTED};font-size:11px}}
.cap{{font-size:16px;line-height:2.05}} .cap span{{padding:2px 3px;border-radius:3px}}
"""
    page = (f'<!doctype html><html lang="bn"><meta charset="utf-8"><style>{css}</style><div class="wrap">'
            f'<p class="lg">SHAP explanations of correctly classified test memes. Colour shows the effect on the '
            f'predicted class:<i style="background:rgba(235,104,52,.7)"></i>supports'
            f'<i style="background:rgba(42,120,214,.7)"></i>argues against</p>'
            f'<div class="hd"><div>Predicted class</div><div>Meme</div><div>SHAP: image regions</div>'
            f'<div>SHAP: caption words</div></div>{"".join(rows)}</div></html>')
    shoot(page, os.path.join(out, "fig_examples_shap.png"), 1180, 70 + len(rows) * 190)
    return picks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "ablation_suite_outputs", "xai", "ce_seed42"))
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--examples", default="", help="comma list of test indices, one per class in figure order")
    a = ap.parse_args()
    ls = os.path.join(a.run_dir, "lime_shap")
    out = os.path.join(ls, "paper_figures"); os.makedirs(out, exist_ok=True)
    W = pd.read_csv(os.path.join(ls, "word_attributions.csv"), encoding="utf-8-sig")
    A = pd.read_csv(os.path.join(ls, "agreement.csv"))
    P = pd.read_csv(os.path.join(a.run_dir, "per_sample.csv"))
    l2_names = [n for _, n in sorted(set(zip(P["pred2"], P["pred2_name"])))]

    fig_top_words(W, out, a.top); print("saved fig_top_words_shap_lime.png")
    r_p, r_s = fig_agreement(W, A, out)
    print(f"saved fig_method_agreement.png  (word-level Pearson {r_p:.3f}, Spearman {r_s:.3f})")
    forced = [int(x) for x in a.examples.split(",") if x]
    picks = fig_examples(P, W, ls, out, l2_names, forced)
    print("saved fig_examples_shap.png  examples:", ", ".join(f"{nice(n)}={i}" for n, _, i in picks))

if __name__ == "__main__":
    main()
