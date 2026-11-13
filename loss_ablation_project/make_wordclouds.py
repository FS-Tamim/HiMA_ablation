# ============================================================
# Word clouds of the most influential words per predicted class (compact, circular).
#
# Bengali needs complex text shaping, which matplotlib / PIL do not do here. The clouds are therefore
# laid out and drawn by headless Chrome: each word is measured with the browser's own (shaped) ink
# box and packed along a spiral inside a circle, at the largest scale at which every word fits.
#
# Usage:
#   python make_wordclouds.py                                  # Integrated Gradients top words
#   python make_wordclouds.py --source shap                    # SHAP top words (after xai_lime_shap.py)
# Options: --run-dir ablation_suite_outputs/xai/ce_seed42   --top 20   --size 420
# Output: <run-dir>/wordclouds/<source>_*.png and <source>_all_classes.png
# ============================================================
import os, argparse, html, json, subprocess
import pandas as pd

FONT_CSS = "'Kohinoor Bangla','Noto Sans Bengali','Bangla MN',sans-serif"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
RAMP = ["#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]       # light -> dark = weak -> strong
SHORT = {"gendered aggression": "Gendered aggression", "political aggression": "Political aggression",
         "religious aggression": "Religious aggression", "others": "Others"}

def load(run_dir, source, top):
    """-> [(level, class, [(word, weight)])] in the order the classes appear in the tables."""
    out = []
    if source == "ig":
        for lvl in ["L1", "L2"]:
            d = pd.read_csv(os.path.join(run_dir, f"top_words_{lvl}.csv"), encoding="utf-8-sig")
            d = d.rename(columns={"mean_share": "weight"}); d["level"] = lvl
            out.append(d)
    else:
        d = pd.read_csv(os.path.join(run_dir, "lime_shap", "top_words_lime_shap.csv"), encoding="utf-8-sig")
        out.append(d.rename(columns={source: "weight"}))
    d = pd.concat(out)
    d = d[d["weight"] > 0]
    res = []
    for (lvl, cls), g in d.groupby(["level", "predicted_class"], sort=False):
        g = g.sort_values("weight", ascending=False).head(top)
        res.append((lvl, cls, list(zip(g["word"], g["weight"].astype(float)))))
    return res

CSS = """
body{margin:0;background:#fcfcfb;font-family:system-ui,FONT;color:#0b0b0b}
.grid{display:grid;grid-template-columns:repeat(COLS,SIZEpx);gap:14px 18px;padding:16px}
.panel{text-align:center} .panel h3{margin:0;font-size:17px;font-weight:600}
.panel p{margin:0 0 5px;font-size:12px;color:#52514e}
canvas{width:SIZEpx;height:SIZEpx;border-radius:50%;background:#fff;box-shadow:0 0 0 1px #e1e0d9;display:block}
"""

# Runs in the browser: spiral packing with shaped-ink collision boxes, largest scale that fits all words.
JS = """
const FONT = "FONTCSS", RAMP = RAMPJSON, S = SIZE, PAD = 3, R = S / 2 - 7;
function layout(ctx, words, scale) {
  const top = words[0][1], placed = [];
  for (const [w, v] of words) {
    const rel = v / top, size = scale * (15 + 62 * Math.pow(rel, 0.75));
    ctx.font = `500 ${size}px ${FONT}`;
    const m = ctx.measureText(w);
    const L = m.actualBoundingBoxLeft, Rt = m.actualBoundingBoxRight;
    const A = m.actualBoundingBoxAscent, D = m.actualBoundingBoxDescent;
    const bw = L + Rt + 2 * PAD, bh = A + D + 2 * PAD;
    let ok = false;
    for (let t = 0; t < 900; t += 0.18) {
      const r = 0.9 * t, cx = S / 2 + r * Math.cos(t), cy = S / 2 + 0.85 * r * Math.sin(t);
      const x0 = cx - bw / 2, y0 = cy - bh / 2, x1 = x0 + bw, y1 = y0 + bh;
      const far = Math.max(Math.hypot(x0 - S / 2, y0 - S / 2), Math.hypot(x1 - S / 2, y0 - S / 2),
                           Math.hypot(x0 - S / 2, y1 - S / 2), Math.hypot(x1 - S / 2, y1 - S / 2));
      if (far > R) { if (r > R) break; continue; }
      if (placed.some(p => x0 < p.x1 && x1 > p.x0 && y0 < p.y1 && y1 > p.y0)) continue;
      placed.push({x0, y0, x1, y1, w, size, rel, bx: x0 + PAD + L, by: y0 + PAD + A});
      ok = true; break;
    }
    if (!ok) return null;
  }
  return placed;
}
for (const cv of document.querySelectorAll("canvas")) {
  const dpr = window.devicePixelRatio || 1;
  cv.width = S * dpr; cv.height = S * dpr;
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr); ctx.textBaseline = "alphabetic";
  const words = JSON.parse(cv.dataset.words);
  let placed = null;
  for (let scale = 2.2; scale > 0.3 && !placed; scale *= 0.96) placed = layout(ctx, words, scale);
  for (const p of placed || []) {
    ctx.font = `500 ${p.size}px ${FONT}`;
    ctx.fillStyle = RAMP[Math.min(Math.floor(p.rel * RAMP.length), RAMP.length - 1)];
    ctx.fillText(p.w, p.bx, p.by);
  }
}
"""

def page(panels, cols, size):
    css = CSS.replace("FONT", FONT_CSS).replace("COLS", str(cols)).replace("SIZE", str(size))
    js = JS.replace("FONTCSS", FONT_CSS).replace("RAMPJSON", json.dumps(RAMP)).replace("SIZE", str(size))
    return (f'<!doctype html><html lang="bn"><meta charset="utf-8"><style>{css}</style>'
            f'<div class="grid">{"".join(panels)}</div><script>{js}</script></html>')

def shoot(page_html, png, width, height):
    tmp = png[:-4] + ".html"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(page_html)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=3",
                    f"--window-size={width},{height}", f"--screenshot={png}", "file://" + os.path.abspath(tmp)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(tmp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "ablation_suite_outputs", "xai", "ce_seed42"))
    ap.add_argument("--source", default="ig", choices=["ig", "shap", "lime"])
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--size", type=int, default=420, help="diameter of each cloud in px")
    a = ap.parse_args()
    if not os.path.isfile(CHROME):
        raise SystemExit("Google Chrome is needed to draw Bengali text correctly.")
    out = os.path.join(a.run_dir, "wordclouds"); os.makedirs(out, exist_ok=True)
    method = {"ig": "Integrated Gradients", "shap": "SHAP", "lime": "LIME"}[a.source]
    head = 46                                                           # title + subtitle height

    panels = []
    for lvl, cls, words in load(a.run_dir, a.source, a.top):
        data = html.escape(json.dumps(words, ensure_ascii=False), quote=True)
        panel = (f'<div class="panel"><h3>{html.escape(SHORT.get(cls, cls))}</h3>'
                 f'<p>Level {lvl[1]} · top {len(words)} words · {method}</p><canvas data-words="{data}"></canvas></div>')
        panels.append(panel)
        fn = os.path.join(out, f"{a.source}_{lvl}_{cls.replace(' ', '_').replace('-', '_').lower()}.png")
        shoot(page([panel], 1, a.size), fn, a.size + 32, a.size + head + 32)
        print("saved", fn)
    cols = 3 if len(panels) > 4 else 2
    rows = -(-len(panels) // cols)
    fn = os.path.join(out, f"{a.source}_all_classes.png")
    shoot(page(panels, cols, a.size), fn, cols * a.size + (cols - 1) * 18 + 32,
          rows * (a.size + head) + (rows - 1) * 14 + 32)
    print("saved", fn)

if __name__ == "__main__":
    main()
