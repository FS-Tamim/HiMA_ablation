# ============================================================
# One shareable, self-contained HTML report that combines every XAI result for HRFDCA.
# All images are embedded, so the single .html file can be emailed or uploaded on its own.
#
# Inputs (all already saved in <run-dir>):
#   xai_analysis.py   : explained_model_test_metrics.csv, modality_ablation.csv, modality_shapley.csv/.png,
#                       faithfulness_deletion.csv, xai_report.html, samples/*.png, cm_*.png, loss_*.png
#   xai_lime_shap.py  : lime_shap/agreement.csv, word_attributions.csv, figures/*.png
#   build_lime_shap_report.py : lime_shap/agreement_per_class.png
#   make_xai_figures.py       : lime_shap/paper_figures/*.png
#   make_wordclouds.py        : wordclouds/{ig,shap,lime}_all_classes.png
#
# Usage:  python build_xai_report.py      Options: --run-dir ablation_suite_outputs/xai/ce_seed42
# Output: <run-dir>/HRFDCA_XAI_report.html
# ============================================================
import os, re, argparse, base64, html, io
import numpy as np
import pandas as pd
from PIL import Image

NICE = {"gendered aggression": "Gendered aggression", "political aggression": "Political aggression",
        "religious aggression": "Religious aggression", "others": "Others"}

def nice(c):
    return NICE.get(c, c)

def embed(path, max_w=1800, photo=False):
    """Image file -> data URI, downscaled to max_w; photos as JPEG to keep the report small."""
    im = Image.open(path)
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    if photo:
        im.convert("RGB").save(buf, "JPEG", quality=82, optimize=True); mime = "jpeg"
    else:
        im.save(buf, "PNG", optimize=True); mime = "png"
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode()

def img(path, alt, max_w=1800, photo=False, style=""):
    if not os.path.isfile(path):
        return f'<p class="note">Missing: {html.escape(os.path.basename(path))}</p>'
    return f'<img src="{embed(path, max_w, photo)}" alt="{html.escape(alt)}" style="{style}">'

def table(df, fmt="{:.3f}"):
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = ""
    for _, r in df.iterrows():
        cells = "".join(f"<td>{fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else html.escape(str(v))}</td>"
                        for v in r)
        body += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

def colour_words(words, scores):
    scale = float(np.abs(scores).max()) or 1.0
    out = []
    for w, v in zip(words, scores):
        a = min(abs(v) / scale, 1.0) * 0.85
        rgb = "235,104,52" if v > 0 else "42,120,214"
        out.append(f'<span style="background:rgba({rgb},{a:.2f})" title="{v:+.4f}">{html.escape(str(w))}</span>')
    return " ".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "ablation_suite_outputs", "xai", "ce_seed42"))
    a = ap.parse_args()
    R = a.run_dir; LS = os.path.join(R, "lime_shap"); PF = os.path.join(LS, "paper_figures")
    WC = os.path.join(R, "wordclouds")

    M = pd.read_csv(os.path.join(R, "explained_model_test_metrics.csv")).iloc[0]
    MA = pd.read_csv(os.path.join(R, "modality_ablation.csv"))
    MS = pd.read_csv(os.path.join(R, "modality_shapley.csv"))
    FD = pd.read_csv(os.path.join(R, "faithfulness_deletion.csv"))
    AG = pd.read_csv(os.path.join(LS, "agreement.csv"))
    FL_path = os.path.join(LS, "faithfulness_lime_shap.csv")
    FL = pd.read_csv(FL_path) if os.path.isfile(FL_path) else pd.DataFrame()
    WA = pd.read_csv(os.path.join(LS, "word_attributions.csv"), encoding="utf-8-sig")
    P = pd.read_csv(os.path.join(R, "per_sample.csv"))
    n = len(P)

    # ---------------- numbers for the summary ----------------
    full, no_txt, no_img = (MA.iloc[k] for k in range(3))
    share = {r.predicted_class: r.text_share for r in MS[MS["level"] == "L2"].itertuples()}
    f_txt = FD[(FD["level"] == "L2") & FD["attribution"].str.startswith("Text")].iloc[0]
    f_img = FD[(FD["level"] == "L2") & FD["attribution"].str.startswith("Image")].iloc[0]
    ag = AG.groupby("level")[["text_spearman", "image_spearman"]].median()
    r_word = np.corrcoef(WA["shap"], WA["lime"])[0, 1]

    metrics = pd.DataFrame([
        dict(level="Level 1: aggressive or not", accuracy=M["CG_A"], weighted_F1=M["CG_F1"], macro_F1=M["CG_MacroF1"]),
        dict(level="Level 2: aggression type *", accuracy=M["FG_A"], weighted_F1=M["FG_F1"], macro_F1=M["FG_MacroF1"])])
    MA2 = MA.rename(columns={"input": "input given to the model", "CG_A": "L1 accuracy", "CG_F1": "L1 weighted F1",
                             "CG_MacroF1": "L1 macro F1", "FG_A": "L2 accuracy", "FG_F1": "L2 weighted F1",
                             "FG_MacroF1": "L2 macro F1"})
    MS2 = MS.assign(predicted_class=MS["predicted_class"].map(nice)).rename(columns={
        "predicted_class": "predicted class", "shap_text": "text contribution", "shap_image": "image contribution",
        "text_share": "text share"})
    FL2 = FL.assign(method=FL["method"].str.upper(), modality=FL["modality"].map({"text": "words", "image": "image regions"})
                    ).rename(columns={"removed_frac": "fraction removed", "drop_top": "drop: top-ranked removed",
                                      "drop_random": "drop: random removed",
                                      "top_beats_random": "top > random (share of memes)"}) if len(FL) else FL
    fl_txt = FL[(FL["level"] == "L2") & (FL["modality"] == "text")].set_index("method") if len(FL) else None
    FD2 = FD.rename(columns={"removed_frac": "fraction removed", "drop_top": "drop: top-ranked removed",
                             "drop_random": "drop: random removed", "top_beats_random": "top > random (share of memes)"})
    AGs = AG.groupby(["level", "predicted_class"]).agg(memes=("idx", "size"),
                                                        words=("text_spearman", "median"),
                                                        image_regions=("image_spearman", "median")).reset_index()
    AGs["predicted_class"] = AGs["predicted_class"].map(nice)
    AGs = AGs.rename(columns={"predicted_class": "predicted class", "words": "median ρ: words",
                              "image_regions": "median ρ: image regions"})

    # ---------------- example memes: IG / Grad-CAM card + LIME / SHAP ----------------
    old = open(os.path.join(R, "xai_report.html"), encoding="utf-8").read()
    words = {k: g.sort_values("position") for k, g in WA.groupby(["idx", "level"])}
    RA_path = os.path.join(LS, "region_attributions.csv")
    regions = ({k: g for k, g in pd.read_csv(RA_path).groupby(["idx", "level"])} if os.path.isfile(RA_path) else {})

    # per-meme text vs image split from LIME / SHAP: summed |attribution| of words vs image regions
    def split(i, lv, m):
        w = words.get((i, lv)); r = regions.get((i, lv))
        if r is None:
            return None
        wa = float(w[m].abs().sum()) if w is not None else 0.0
        ra = float(r[m].abs().sum())
        wp = float(np.clip(w[m], 0, None).sum()) if w is not None else 0.0
        rp = float(np.clip(r[m], 0, None).sum())
        return wa / max(wa + ra, 1e-12), wp, rp

    def split_line(i):
        out = []
        for m in ["shap", "lime"]:
            parts = []
            for lv in ["L1", "L2"]:
                v = split(i, lv, m)
                if v:
                    parts.append(f"Level {lv[1]}: text share {v[0]:.2f} (words +{v[1]:.3f}, image regions +{v[2]:.3f})")
            if parts:
                out.append(f'<p class="note"><b>{m.upper()} text vs image:</b> ' + " &nbsp;|&nbsp; ".join(parts) + "</p>")
        return "".join(out)

    def where(cx, cy):
        v = "top" if cy < 1 / 3 else ("bottom" if cy > 2 / 3 else "middle")
        h = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "centre")
        return "centre" if (v, h) == ("middle", "centre") else f"{v} {h}"

    def value_tables(i, lv):
        """Top words and top image regions for one meme and level, with their LIME and SHAP values."""
        parts = []
        if (i, lv) in words:
            g = words[(i, lv)].assign(a=lambda d: d["shap"].abs()).sort_values("a", ascending=False).head(6)
            rows = "".join(f"<tr><td class='bn'>{html.escape(str(r.word))}</td><td>{r.shap:+.3f}</td><td>{r.lime:+.3f}</td></tr>"
                           for r in g.itertuples())
            parts.append(f"<div><b>Top words (by |SHAP|)</b><table><thead><tr><th>word</th><th>SHAP</th><th>LIME</th>"
                         f"</tr></thead><tbody>{rows}</tbody></table></div>")
        if (i, lv) in regions:
            g = regions[(i, lv)]
            pos = np.clip(g["shap"].to_numpy(), 0, None); k = max(1, int(round(0.2 * len(g))))
            conc = np.sort(pos)[::-1][:k].sum() / max(pos.sum(), 1e-12)
            t = g.sort_values("shap", ascending=False).head(5)
            rows = "".join(f"<tr><td>{where(r.cx, r.cy)}</td><td>{r.area_frac:.1%}</td><td>{r.shap:+.3f}</td>"
                           f"<td>{r.lime:+.3f}</td></tr>" for r in t.itertuples())
            parts.append(f"<div><b>Top image regions (by SHAP)</b><table><thead><tr><th>location</th><th>area</th>"
                         f"<th>SHAP</th><th>LIME</th></tr></thead><tbody>{rows}</tbody></table>"
                         f"<p class='note'>{len(g)} regions; the top 20% hold {conc:.0%} of the positive SHAP value.</p></div>")
        return f'<div class="vals"><h5>Level {lv[1]} values</h5>{"".join(parts)}</div>' if parts else ""
    cards = []
    for sec in re.findall(r"<section>.*?</section>", old, re.S):
        m = re.search(r"test sample (\d+)", sec)
        if not m:
            continue
        i = int(m.group(1))
        sec = re.sub(r'<img src="(samples/[^"]+)"[^>]*>',
                     lambda mm: img(os.path.join(R, mm.group(1)), "Grad-CAM and attention maps", 1400, True), sec)
        sec = sec.replace("<section>", f'<section class="card" id="m{i}">', 1)
        sec = re.sub(r'(<p class="note"><b>Modality Shapley:</b>.*?</p>)', lambda mm: mm.group(1) + split_line(i), sec,
                     count=1, flags=re.S)
        sec = sec.replace("<b>Level 1 words</b>", "<b>Integrated Gradients — Level 1</b>")
        sec = sec.replace("<b>Level 2 words</b>", "<b>Integrated Gradients — Level 2</b>")
        extra = ['<h4>LIME and SHAP</h4>',
                 img(os.path.join(LS, "figures", f"sample_{i:04d}.png"), "LIME and SHAP image maps", 960, True)]
        for lv in ["L1", "L2"]:
            if (i, lv) in words:
                g = words[(i, lv)]
                for meth in ["shap", "lime"]:
                    extra.append(f'<div class="row"><b>{meth.upper()} — Level {lv[1]}</b><br><span class="txt">'
                                 f'{colour_words(g["word"].tolist(), g[meth].to_numpy())}</span></div>')
        extra.append('<div class="valgrid">' + "".join(value_tables(i, lv) for lv in ["L1", "L2"]) + "</div>")
        cards.append(sec.replace("</section>", "".join(extra) + "</section>"))

    css = """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--card:#fcfcfb;--page:#f9f9f7;--link:#256abf}
*{box-sizing:border-box}
body{font-family:system-ui,'Noto Sans Bengali','Kohinoor Bangla',sans-serif;max-width:1120px;margin:0 auto;
padding:0 16px 40px;color:var(--ink);background:var(--page);line-height:1.55}
header{padding:28px 0 8px} h1{margin:0 0 4px;font-size:28px} .sub{color:var(--ink2);margin:0}
nav{position:sticky;top:0;background:var(--page);padding:10px 0;border-bottom:1px solid var(--line);z-index:5;
font-size:14px;display:flex;flex-wrap:wrap;gap:6px 16px} nav a{color:var(--link);text-decoration:none}
h2{margin:34px 0 10px;font-size:22px} h3{margin:18px 0 6px;font-size:17px} h4{margin:14px 0 4px;font-size:15px}
.box,section.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 20px;margin:14px 0}
.warn{background:#fff6ec;border-color:#f3c9a4}
img{max-width:100%;height:auto;display:block;margin:8px 0}
table{border-collapse:collapse;font-size:14px;margin:8px 0;display:block;overflow-x:auto}
td,th{padding:5px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
td:first-child,th:first-child{text-align:left} th{font-weight:600}
.note{color:var(--ink2);font-size:14px} .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px} .kpi span{font-size:13px;color:var(--ink2)}
ul.find li{margin:6px 0}
.txt{font-size:17px;line-height:2.05} .txt span{padding:2px 3px;border-radius:3px} .row{margin:10px 0}
.ok,.bad{font-size:12px;font-weight:600;padding:2px 8px;border-radius:10px;border:1px solid}
.ok{color:#006300} .bad{color:#d03b3b} .key span{padding:2px 6px;border-radius:3px}
section.card h3{margin-top:0}
.valgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:12px;margin-top:10px}
.vals{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:#fff}
.vals h5{margin:0 0 6px;font-size:14px} .vals>div{display:inline-block;vertical-align:top;margin-right:14px}
.vals table{font-size:13px} .vals td,.vals th{padding:3px 8px} td.bn{text-align:left;font-size:15px}
"""
    key = ('<p class="note">Under each meme, <b>Modality Shapley</b> gives the text and image contribution to the predicted '
           'class (probability units). The <b>SHAP / LIME text vs image</b> lines give the caption\'s share of the meme\'s '
           'total attribution (|words| / (|words| + |image regions|)) and the summed positive attribution of words and '
           'of image regions.</p>'
           '<p class="key">Word colours: <span style="background:rgba(235,104,52,.6)">supports the predicted class</span> '
           '<span style="background:rgba(42,120,214,.6)">argues against it</span>. Hover a word for its value.</p>')

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>HRFDCA XAI Report</title>
<style>{css}</style></head><body>
<header><h1>Explaining HRFDCA: XAI report</h1>
<p class="sub">Hierarchical multimodal aggression detection in Bengali memes — what the model relies on, and how
reliable the explanations are.</p></header>
<nav><a href="#summary">Summary</a><a href="#setup">Model &amp; methods</a><a href="#modality">Text vs image</a>
<a href="#faith">Faithfulness</a><a href="#words">Influential words</a><a href="#agree">LIME vs SHAP</a>
<a href="#examples">Examples</a><a href="#memes">Meme gallery</a><a href="#notes">Limitations</a></nav>

<div class="box warn"><b>Content note.</b> This report contains memes with offensive, political, religious and
gendered content, and images of real people, taken from the test set. Share it only with people who should see the data.</div>

<h2 id="summary">Summary</h2>
<div class="kpis">
<div class="kpi"><b>{n}</b><span>test memes explained (full test set)</span></div>
<div class="kpi"><b>{full['FG_F1']:.3f} → {no_txt['FG_F1']:.3f}</b><span>Level-2 F1 when the caption is removed</span></div>
<div class="kpi"><b>{full['FG_F1']:.3f} → {no_img['FG_F1']:.3f}</b><span>Level-2 F1 when the image is removed</span></div>
<div class="kpi"><b>{ag.loc['L1', 'text_spearman']:.2f} / {ag.loc['L1', 'image_spearman']:.2f}</b><span>median LIME–SHAP agreement, words / image regions (Level 1)</span></div>
</div>
<div class="box"><ul class="find">
<li><b>The caption drives most decisions.</b> Without the caption, Level-2 weighted F1 falls from {full['FG_F1']:.3f} to
{no_txt['FG_F1']:.3f}; without the image it only falls to {no_img['FG_F1']:.3f}.</li>
<li><b>The image matters most for gendered aggression.</b> Text accounts for {share.get('gendered aggression', np.nan):.0%} of the
contribution there, against {share.get('political aggression', np.nan):.0%} for political and
{share.get('religious aggression', np.nan):.0%} for religious aggression.</li>
<li><b>The explanations are faithful.</b> Removing the top 20% of words ranked by Integrated Gradients lowers the Level-2
confidence by {f_txt['drop_top']:.2f}, against {f_txt['drop_random']:.2f} for random words; for Grad-CAM image regions
the drop is {f_img['drop_top']:.3f} against {f_img['drop_random']:.3f}.{(" LIME and SHAP pass the same test: removing their top 20% of words lowers Level-2 confidence by " + f"{fl_txt.loc['lime', 'drop_top']:.2f} (LIME) and {fl_txt.loc['shap', 'drop_top']:.2f} (SHAP), against {fl_txt.loc['shap', 'drop_random']:.2f} at random.") if fl_txt is not None else ""}</li>
<li><b>Influential words match each class.</b> Party and leader names for political aggression, religion words for
religious aggression, women-related words for gendered aggression, study and family words for non-aggressive memes.</li>
<li><b>LIME and SHAP tell the same story.</b> Across {len(WA):,} word scores the two methods correlate at r = {r_word:.2f};
per meme the median rank agreement is {ag.loc['L1', 'text_spearman']:.2f} for words and
{ag.loc['L1', 'image_spearman']:.2f} for image regions.</li>
</ul></div>

<h2 id="setup">Model and methods</h2>
<div class="box">
<p><b>Model.</b> HRFDCA — MuRIL text encoder and Swin-T image encoder joined by bidirectional cross-attention, with a
hierarchical residual head (Level 1: aggressive or not; Level 2: gendered, political, religious, others). Trained with
{html.escape(str(M['loss_name']))} loss, seed {int(M['seed'])}, 15 epochs; best epoch {int(M['best_epoch'])} by validation loss.</p>
{table(metrics)}
<p class="note">* Level 2 is scored on memes that are truly aggressive, as in the ablation study. This model was
retrained for the XAI analysis, so its numbers differ slightly from the ablation table.</p>
<h3>Explanation methods</h3>
<ul>
<li><b>Modality ablation and Shapley values</b> — the model is run with the caption, the image, both, or neither; the
exact two-player Shapley value splits each prediction between text and image.</li>
<li><b>Integrated Gradients</b> (32 steps, [PAD] baseline) — word-level attribution through the MuRIL embeddings.</li>
<li><b>Grad-CAM</b> on the 7×7 Swin-T tokens, and the model's own <b>cross-attention</b> maps.</li>
<li><b>LIME</b> and <b>KernelSHAP</b> — model-agnostic; words removed with [MASK], image split into ~50 superpixels
filled with grey; 500 model evaluations per explanation.</li>
<li><b>Deletion test</b> — remove the top 20% of attributed words or image cells and measure the confidence drop,
compared with removing the same amount at random.</li>
</ul></div>

<h2 id="modality">Text vs image</h2>
<div class="box">
<h3>Performance with one modality removed</h3>{table(MA2)}
<h3>Contribution of each modality to the predicted class (two-player Shapley)</h3>
{img(os.path.join(R, 'modality_shapley.png'), 'Modality Shapley chart', 1400)}
{table(MS2)}
<p class="note">With no input the model defaults to "aggressive", so for memes predicted aggressive neither modality
adds much at Level 1, while for memes predicted non-aggressive the caption does almost all the work.</p>
</div>

<h2 id="faith">Faithfulness of the explanations</h2>
<div class="box"><h3>Integrated Gradients and Grad-CAM</h3>{table(FD2)}
{"<h3>LIME and SHAP</h3>" + table(FL2) if len(FL) else ""}
<p class="note">A larger confidence drop for top-ranked than for random removal means the explanation points at
evidence the model really uses.</p></div>

<h2 id="words">Most influential words per class</h2>
<div class="box">
<h3>SHAP and LIME, top 10 words per predicted class</h3>
{img(os.path.join(PF, 'fig_top_words_shap_lime.png'), 'Top words SHAP vs LIME', 1700)}
<h3>Word clouds — Integrated Gradients</h3>{img(os.path.join(WC, 'ig_all_classes.png'), 'IG word clouds', 1500)}
<h3>Word clouds — SHAP</h3>{img(os.path.join(WC, 'shap_all_classes.png'), 'SHAP word clouds', 1500)}
<h3>Word clouds — LIME</h3>{img(os.path.join(WC, 'lime_all_classes.png'), 'LIME word clouds', 1500)}
<p class="note">Words seen at least three times in the test set; bigger and darker means more influential.</p></div>

<h2 id="agree">Agreement between LIME and SHAP</h2>
<div class="box">
{img(os.path.join(PF, 'fig_method_agreement.png'), 'LIME vs SHAP agreement', 1700)}
{img(os.path.join(LS, 'agreement_per_class.png'), 'Agreement per class', 1400)}
{table(AGs)}
<p class="note">Spearman ρ between the LIME and SHAP scores of the same meme (1 = identical ranking).</p></div>

<h2 id="examples">Worked examples (SHAP)</h2>
<div class="box">{img(os.path.join(PF, 'fig_examples_shap.png'), 'SHAP examples', 1700)}</div>

<h2 id="memes">Meme gallery: every method side by side</h2>
<div class="box"><p>{len(cards)} test memes (correct and misclassified, all classes). For each: the Grad-CAM and
cross-attention maps, Integrated Gradients words, then the LIME and SHAP image maps and words, and tables with the
actual LIME and SHAP values of the strongest words and image regions. The values for every word and region of all
485 memes are in <code>word_attributions.csv</code> and <code>region_attributions.csv</code>.</p>{key}</div>
{"".join(cards)}

<h2 id="notes">Limitations</h2>
<div class="box"><ul>
<li>One model (CE loss, seed 42) is explained; findings describe this model, not every training run.</li>
<li>Removing a modality or a word changes the input distribution; the ablation and deletion numbers are relative
comparisons, not exact causal effects.</li>
<li>Grad-CAM works on a 7×7 grid, so image maps are coarse and often land on text printed inside the meme.</li>
<li>Word statistics use words seen at least three times; rarer but decisive words are not listed.</li>
<li>Level-2 metrics are computed on truly aggressive memes only.</li>
</ul></div>
<p class="note">Generated from the saved outputs of xai_analysis.py, xai_lime_shap.py, make_xai_figures.py and
make_wordclouds.py.</p>
</body></html>"""
    out = os.path.join(R, "HRFDCA_XAI_report.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"Saved {out}  ({os.path.getsize(out) / 1e6:.1f} MB, {len(cards)} gallery memes)")

if __name__ == "__main__":
    main()
