# ============================================================
# Builds the complete LIME + SHAP report from the files xai_lime_shap.py saved, without
# re-running the model:
#   lime_shap/word_attributions.csv, lime_shap/agreement.csv, lime_shap/figures/*.png,
#   per_sample.csv and explained_model_test_metrics.csv (from xai_analysis.py),
#   wordclouds/{shap,lime}_all_classes.png (from make_wordclouds.py, optional)
#
# Report sections: overview and test metrics, how to read it, LIME/SHAP agreement (overall and
# per class, with a chart), top words by both methods, word clouds, and every test meme with
# filters (level-1 / level-2 class, correct / misclassified, sample number).
#
# Usage:
#   python build_lime_shap_report.py --data-root <folder with hierarchical_aggression.csv>
# Options: --run-dir ablation_suite_outputs/xai/ce_seed42
# ============================================================
import os, sys, argparse, html, json

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="", help="only needed to show captions that have no word explanation")
    ap.add_argument("--run-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "ablation_suite_outputs", "xai", "ce_seed42"))
    return ap.parse_args()

ARGS = parse_args()
if ARGS.data_root:
    os.environ["ABLATION_DATA_ROOT"] = ARGS.data_root

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_WORD, C_IMG, C_INK, C_MUTED, C_GRID, C_SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
L1 = ["Non-Aggressive", "Aggressive"]
NICE = {"gendered aggression": "Gendered aggression", "political aggression": "Political aggression",
        "religious aggression": "Religious aggression", "others": "Others"}

def nice(c):
    return NICE.get(c, c)

def colour_words(words, scores):
    scale = float(np.abs(scores).max()) or 1.0
    out = []
    for w, v in zip(words, scores):
        a = min(abs(v) / scale, 1.0) * 0.85
        rgb = "235,104,52" if v > 0 else "42,120,214"
        out.append(f'<span style="background:rgba({rgb},{a:.2f})" title="{v:+.4f}">{html.escape(str(w))}</span>')
    return " ".join(out)

def table(df, fmt=None):
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                v = "–" if pd.isna(v) else (fmt or "{:.3f}").format(v)
            cells.append(f"<td>{html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'

def captions(run_dir, seed, n):
    """Test-set captions (same split as training) - only if the dataset is available."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import ablation_suite as S
        if not os.path.isfile(S.DATASET_PATH):
            return {}
        df, _ = S.load_dataset(S.DATASET_PATH, S.IMAGE_DIR)
        _, _, test = S.stratified_split(df, seed)
        return dict(zip(range(len(test)), test["text"])) if len(test) == n else {}
    except Exception as e:
        print("captions not loaded:", e)
        return {}

def main():
    run = ARGS.run_dir
    ls = os.path.join(run, "lime_shap")
    W = pd.read_csv(os.path.join(ls, "word_attributions.csv"), encoding="utf-8-sig")
    A = pd.read_csv(os.path.join(ls, "agreement.csv"))
    P = pd.read_csv(os.path.join(run, "per_sample.csv"))
    M = pd.read_csv(os.path.join(run, "explained_model_test_metrics.csv")).iloc[0]
    l2_names = [n for _, n in sorted(set(zip(P["pred2"], P["pred2_name"])))]
    n = len(P)
    cap = captions(run, int(M["seed"]), n)

    # ---------------- overview ----------------
    P["true1_name"] = P["true1"].map(lambda x: L1[x])
    P["true2_name"] = [l2_names[t] if t1 == 1 else "" for t1, t in zip(P["true1"], P["true2"])]
    P["ok1"] = P["true1"] == P["pred1"]
    P["ok"] = P["ok1"] & ((P["true1"] == 0) | ((P["pred1"] == 1) & (P["true2"] == P["pred2"])))
    metrics = pd.DataFrame([
        dict(level="Level 1 (aggressive or not)", accuracy=M["CG_A"], weighted_F1=M["CG_F1"], macro_F1=M["CG_MacroF1"]),
        dict(level="Level 2 (aggression type)*", accuracy=M["FG_A"], weighted_F1=M["FG_F1"], macro_F1=M["FG_MacroF1"])])

    # ---------------- agreement ----------------
    A["level_name"] = A["level"].map({"L1": "Level 1", "L2": "Level 2"})
    def summ(g):
        return pd.Series(dict(memes=len(g),
                              words_mean=g["text_spearman"].mean(), words_median=g["text_spearman"].median(),
                              regions_mean=g["image_spearman"].mean(), regions_median=g["image_spearman"].median()))
    over = A.groupby("level_name").apply(summ).reset_index().rename(columns={"level_name": "level"})
    per = A.groupby(["level_name", "predicted_class"]).apply(summ).reset_index()
    per["predicted_class"] = per["predicted_class"].map(nice)
    per = per.rename(columns={"level_name": "level", "predicted_class": "predicted class"})
    for d in (over, per):
        d["memes"] = d["memes"].astype(int)
    no_rho = int(A["text_spearman"].isna().sum())

    fig, ax = plt.subplots(figsize=(9.5, 4.4), facecolor=C_SURF); ax.set_facecolor(C_SURF)
    lab = [f"L{r['level'][-1]}\n{r['predicted class'].replace(' aggression', '')}\n(n={r['memes']})"
           for _, r in per.iterrows()]
    x = np.arange(len(per)); w = 0.36
    ax.bar(x - w / 2 - 0.01, per["words_mean"], w, color=C_WORD, label="Caption words")
    ax.bar(x + w / 2 + 0.01, per["regions_mean"], w, color=C_IMG, label="Image regions")
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8.5, color=C_INK)
    ax.set_ylim(0, 1); ax.set_ylabel("Mean Spearman correlation", color=C_INK)
    ax.set_title("How closely LIME and SHAP rank the same evidence, per predicted class", loc="left",
                 fontsize=11, color=C_INK)
    ax.grid(axis="y", color=C_GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7"); ax.tick_params(colors=C_MUTED, length=0)
    ax.legend(frameon=False, labelcolor=C_INK, loc="upper left", bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_ylim(0, 1.12); ax.set_yticks(np.arange(0, 1.01, 0.2))
    plt.tight_layout(); plt.savefig(os.path.join(ls, "agreement_per_class.png"), dpi=160); plt.close()

    # ---------------- top words (both methods) ----------------
    W["w"] = W["word"].astype(str).str.strip().str.lower()
    W = W[W["w"].map(lambda s: len(s) > 1 and any(ch.isalpha() for ch in s))]
    G = W.groupby(["level", "predicted_class", "w"]).agg(shap=("shap", "mean"), lime=("lime", "mean"),
                                                          count=("shap", "size")).reset_index()
    G = G[G["count"] >= 3]
    tops = []
    for (lv, cls), g in G.groupby(["level", "predicted_class"]):
        s = g.sort_values("shap", ascending=False).head(10).reset_index(drop=True)
        l = g.sort_values("lime", ascending=False).head(10).reset_index(drop=True)
        both = len(set(s["w"]) & set(l["w"]))
        rows = "".join(
            f"<tr><td>{k + 1}</td><td class='bn'>{html.escape(s.loc[k, 'w']) if k < len(s) else ''}</td>"
            f"<td>{s.loc[k, 'shap']:+.3f}</td><td class='bn'>{html.escape(l.loc[k, 'w']) if k < len(l) else ''}</td>"
            f"<td>{l.loc[k, 'lime']:+.3f}</td></tr>"
            for k in range(min(len(s), len(l))))
        tops.append(f'<div class="topcard"><h3>Level {lv[1]} — {html.escape(nice(cls))}</h3>'
                    f'<p class="note">{both} of the top 10 words are shared by both methods.</p>'
                    f'<table><thead><tr><th>#</th><th>SHAP word</th><th>SHAP</th><th>LIME word</th><th>LIME</th>'
                    f'</tr></thead><tbody>{rows}</tbody></table></div>')

    clouds = ""
    for src, name in [("shap", "SHAP"), ("lime", "LIME")]:
        f = os.path.join(run, "wordclouds", f"{src}_all_classes.png")
        if os.path.isfile(f):
            clouds += (f'<h3>{name}</h3><img loading="lazy" src="../wordclouds/{src}_all_classes.png" '
                       f'alt="{name} word clouds per class">')

    # ---------------- every meme ----------------
    words = {k: g.sort_values("position") for k, g in
             pd.read_csv(os.path.join(ls, "word_attributions.csv"), encoding="utf-8-sig").groupby(["idx", "level"])}
    ag = {(r.idx, r.level): r for r in A.itertuples()}
    cards = []
    for r in P.itertuples():
        i = int(r.idx)
        lines = []
        for lv in ["L1", "L2"]:
            if (i, lv) not in ag:
                continue
            a = ag[(i, lv)]
            name = nice(a.predicted_class)
            rho_t = "–" if pd.isna(a.text_spearman) else f"{a.text_spearman:.2f}"
            rho_i = "–" if pd.isna(a.image_spearman) else f"{a.image_spearman:.2f}"
            if (i, lv) in words:
                g = words[(i, lv)]
                for m in ["shap", "lime"]:
                    lines.append(f'<div class="row"><b>{m.upper()} — Level {lv[1]}</b> (why “{html.escape(name)}”)<br>'
                                 f'<span class="txt">{colour_words(g["word"].tolist(), g[m].to_numpy())}</span></div>')
            else:
                c = cap.get(i, "")
                lines.append(f'<div class="row"><b>Level {lv[1]}</b>: no word explanation — the caption has fewer '
                             f'than two words{": “" + html.escape(c) + "”" if c else ""}.</div>')
            lines.append(f'<p class="note">Level {lv[1]} agreement between LIME and SHAP: words {rho_t}, '
                         f'image regions {rho_i}.</p>')
        true_s = r.true1_name + (f" / {nice(r.true2_name)}" if r.true1 == 1 else "")
        pred_s = f"{L1[r.pred1]} ({r.conf1:.2f})" + (f" / {nice(r.pred2_name)} ({r.conf2:.2f})" if r.pred1 == 1 else "")
        c1 = L1[r.pred1]
        c2 = r.pred2_name if r.pred1 == 1 else "none"
        cards.append(
            f'<section class="card" data-idx="{i}" data-l1="{html.escape(c1)}" data-l2="{html.escape(c2)}" '
            f'data-ok="{"ok" if r.ok else "bad"}"><h3>Test sample {i} <span class="img">({html.escape(str(r.image))})</span> '
            f'<span class="{"ok" if r.ok else "bad"}">{"correct" if r.ok else "misclassified"}</span></h3>'
            f'<p><b>True:</b> {html.escape(true_s)} &nbsp; <b>Predicted:</b> {html.escape(pred_s)}</p>'
            f'<img loading="lazy" src="figures/sample_{i:04d}.png" alt="LIME and SHAP image maps for sample {i}">'
            f'{"".join(lines)}</section>')

    l2_opts = "".join(f'<option value="{html.escape(c)}">{html.escape(nice(c))}</option>' for c in l2_names)
    n_ok, n_bad = int(P["ok"].sum()), int((~P["ok"]).sum())
    page = f"""<!doctype html><html lang="bn"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>HRFDCA LIME and SHAP</title>
<style>
:root{{--ink:#0b0b0b;--ink2:#52514e;--line:#e1e0d9;--card:#fcfcfb;--page:#f9f9f7}}
body{{font-family:system-ui,'Noto Sans Bengali','Kohinoor Bangla',sans-serif;max-width:1100px;margin:24px auto;
padding:0 16px;color:var(--ink);background:var(--page);line-height:1.5}}
nav{{position:sticky;top:0;background:var(--page);padding:8px 0;border-bottom:1px solid var(--line);z-index:2;font-size:14px}}
nav a{{margin-right:14px;color:#256abf;text-decoration:none}}
section,.box{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 18px;margin:16px 0}}
img{{max-width:100%}} .txt{{font-size:17px;line-height:2}} .txt span{{padding:2px 3px;border-radius:3px}}
.row{{margin:10px 0}} .note{{color:var(--ink2);font-size:14px;margin:4px 0}} h3{{margin:0 0 6px;font-size:16px}}
.img{{color:var(--ink2);font-weight:400}}
.ok,.bad{{font-size:12px;font-weight:600;padding:2px 8px;border-radius:10px;border:1px solid}}
.ok{{color:#006300}} .bad{{color:#d03b3b}}
table{{border-collapse:collapse;font-size:14px;margin:6px 0}} td,th{{padding:4px 10px;border-bottom:1px solid var(--line);text-align:right}}
td:first-child,th:first-child,td.bn{{text-align:left}} .bn{{font-size:15px}}
.tops{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}}
.topcard table{{width:100%}} .topcard td,.topcard th{{padding:3px 6px}}
.topcard{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}}
.filters{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;font-size:14px}}
.filters select,.filters input{{font:inherit;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:#fff}}
.key span{{padding:2px 6px;border-radius:3px}} .hidden{{display:none}}
</style></head><body>
<h1>HRFDCA — LIME and SHAP explanations</h1>
<nav><a href="#overview">Overview</a><a href="#read">How to read</a><a href="#agreement">Agreement</a>
<a href="#words">Top words</a><a href="#clouds">Word clouds</a><a href="#memes">All memes ({n})</a></nav>

<h2 id="overview">Overview</h2>
<div class="box">
<p>Model: HRFDCA (MuRIL + Swin-T with dual cross-attention) trained with <b>{html.escape(str(M['loss_name']))}</b> loss,
seed {int(M['seed'])}, best epoch {int(M['best_epoch'])} of 15. Explained: all <b>{n}</b> test memes
({n_ok} fully correct, {n_bad} misclassified at either level).</p>
{table(metrics)}
<p class="note">* Level 2 is scored on memes that are truly aggressive, as in the ablation study.</p>
</div>

<h2 id="read">How to read this report</h2>
<div class="box">
<p>Both methods explain the probability of the class the model <b>predicted</b>, at Level 1 and — when the meme is
predicted aggressive — at Level 2. Each explanation uses 500 model evaluations.</p>
<ul>
<li><b>Caption words:</b> a word is removed by replacing it with [MASK]; the image is unchanged.</li>
<li><b>Image regions:</b> the image is split into about 50 SLIC superpixels; a region is removed by filling it with grey;
the caption is unchanged.</li>
<li><b>LIME</b> fits a local linear model to random removals; <b>SHAP</b> (KernelSHAP) estimates each word's or region's
Shapley value, its average contribution over many combinations.</li>
</ul>
<p class="key">Colours for words and image regions: <span style="background:rgba(235,104,52,.6)">supports the predicted class</span>
<span style="background:rgba(42,120,214,.6)">argues against it</span>. Stronger colour = larger effect. Hover a word for its value.</p>
</div>

<h2 id="agreement">Agreement between LIME and SHAP</h2>
<div class="box">
<p>Spearman rank correlation between the LIME and SHAP scores of the same meme (1 = identical ranking, 0 = unrelated).</p>
{table(over)}
<img loading="lazy" src="agreement_per_class.png" alt="Mean LIME-SHAP agreement per predicted class" style="max-width:900px">
{table(per)}
<p class="note">{no_rho} explanation(s) have no word correlation because the caption is too short or all word scores
were equal; sample 113 has a one-word caption and no word explanation.</p>
</div>

<h2 id="words">Most influential words per predicted class</h2>
<p class="note">Mean score over the memes predicted as that class; only words seen at least three times are listed.</p>
<div class="tops">{"".join(tops)}</div>

<h2 id="clouds">Word clouds</h2>
<div class="box">{clouds or '<p class="note">Run make_wordclouds.py --source shap / --source lime to add them.</p>'}</div>

<h2 id="memes">All test memes</h2>
<div class="box filters">
<label>Level 1 <select id="f1"><option value="">all</option><option>Aggressive</option><option>Non-Aggressive</option></select></label>
<label>Level 2 <select id="f2"><option value="">all</option>{l2_opts}</select></label>
<label>Result <select id="fo"><option value="">all</option><option value="ok">correct</option><option value="bad">misclassified</option></select></label>
<label>Sample # <input id="fi" type="number" min="0" max="{n - 1}" style="width:80px"></label>
<span id="count" class="note"></span>
</div>
{"".join(cards)}
<script>
const cards = [...document.querySelectorAll('section.card')];
function apply() {{
  const f1 = document.getElementById('f1').value, f2 = document.getElementById('f2').value,
        fo = document.getElementById('fo').value, fi = document.getElementById('fi').value;
  let shown = 0;
  for (const c of cards) {{
    const ok = (!f1 || c.dataset.l1 === f1) && (!f2 || c.dataset.l2 === f2) &&
               (!fo || c.dataset.ok === fo) && (fi === '' || c.dataset.idx === fi);
    c.classList.toggle('hidden', !ok); if (ok) shown++;
  }}
  document.getElementById('count').textContent = shown + ' of ' + cards.length + ' memes shown';
}}
for (const id of ['f1', 'f2', 'fo', 'fi']) document.getElementById(id).addEventListener('input', apply);
apply();
</script>
</body></html>"""
    out = os.path.join(ls, "lime_shap_report.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"Saved {out}\n  memes: {n} | cards: {len(cards)} | top-word tables: {len(tops)} | "
          f"captions loaded: {len(cap)}")

if __name__ == "__main__":
    main()
