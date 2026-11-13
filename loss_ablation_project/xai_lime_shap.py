# ============================================================
# LIME + SHAP explanations for the trained HRFDCA (uses the checkpoint saved by xai_analysis.py)
#
# For each selected test meme and for both hierarchy levels (Level 2 only if predicted aggressive):
#   Text  : LIME (LimeTextExplainer) and SHAP (KernelSHAP) over the caption words; a removed word
#           is replaced by [MASK], the image stays unchanged.
#   Image : LIME (LimeImageExplainer) and SHAP (KernelSHAP) over SLIC superpixels; a removed
#           superpixel is filled with the dataset-mean colour, the caption stays unchanged.
# Both methods explain the probability of the class the model predicted.
#
# Output (next to the checkpoint, in lime_shap/):
#   lime_shap_report.html, figures/sample_XXXX.png, word_attributions.csv, agreement.csv,
#   region_attributions.csv (LIME/SHAP value, area and centre of every superpixel), segments.npz (SLIC maps),
#   faithfulness_lime_shap.csv (+ _per_meme): deletion test of the top-ranked words / regions vs random
#
# Usage:
#   python xai_lime_shap.py --data-root <folder with hierarchical_aggression.csv>
# Options: --ckpt path.pt   --samples 19,20,21 (test indices; default = the memes in xai_report.html)
#          --all (every test meme)   --n-samples 500 (model evaluations per explanation)   --segments 50
#          --shard 0/2 --no-report (split the memes over several GPUs; a final run without --shard
#          builds the report). Finished memes are cached in lime_shap/cache and skipped on restart.
# ============================================================

import os, sys, argparse, glob, html, re, pickle, time, contextlib

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="")
    ap.add_argument("--ckpt", default="", help="default: ablation_suite_outputs/xai/ce_seed42/hrfdca_ce_seed42.pt")
    ap.add_argument("--samples", default="", help="comma list of test indices")
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--segments", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--all", action="store_true", help="explain every test meme")
    ap.add_argument("--delete-frac", type=float, default=0.2, help="fraction removed in the deletion test")
    ap.add_argument("--shard", default="", help="k/n: only every n-th meme starting at k (parallel GPUs)")
    ap.add_argument("--no-report", action="store_true", help="only compute and cache (used with --shard)")
    return ap.parse_args()

ARGS = parse_args()
if ARGS.data_root:
    os.environ["ABLATION_DATA_ROOT"] = ARGS.data_root

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from lime.lime_text import LimeTextExplainer
from lime.lime_image import LimeImageExplainer
from lime.wrappers.scikit_image import SegmentationAlgorithm
import shap

import ablation_suite as S
import xai_analysis as X

device = S.device
DIVERGING = LinearSegmentedColormap.from_list("bgo", ["#2a78d6", "#f0efec", "#eb6834"])   # against / neutral / for
MEAN_T = torch.tensor(X.MEAN, dtype=torch.float32).view(1, 3, 1, 1)
STD_T = torch.tensor(X.STD, dtype=torch.float32).view(1, 3, 1, 1)

def shap_matrix(sv, n_feat, n_out):
    """KernelExplainer output -> (n_feat, n_out), whatever the shap version returns."""
    sv = np.array(sv)
    if sv.shape == (n_out, 1, n_feat):
        return sv[:, 0, :].T
    if sv.shape == (1, n_feat, n_out):
        return sv[0]
    raise ValueError(f"Unexpected SHAP output shape {sv.shape}")

def colour_words(words, scores):
    scale = float(np.abs(scores).max()) or 1.0
    out = []
    for w, v in zip(words, scores):
        a = min(abs(v) / scale, 1.0) * 0.85
        rgb = "235,104,52" if v > 0 else "42,120,214"
        out.append(f'<span style="background:rgba({rgb},{a:.2f})" title="{v:+.4f}">{html.escape(w)}</span>')
    return " ".join(out)

def rho(a, b):
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return float(spearmanr(a, b)[0])

def main():
    a = ARGS
    ckpt = a.ckpt or os.path.join(S.OUT_ROOT, "xai", "ce_seed42", "hrfdca_ce_seed42.pt")
    if not os.path.isfile(ckpt):
        sys.exit(f"Checkpoint not found: {ckpt}  (run xai_analysis.py first)")
    base = os.path.dirname(ckpt)
    out = os.path.join(base, "lime_shap"); os.makedirs(os.path.join(out, "figures"), exist_ok=True)

    df, le2 = S.load_dataset(S.DATASET_PATH, S.IMAGE_DIR)
    l2_names = le2.classes_.tolist(); n_l2 = len(l2_names); n_out = 2 + n_l2
    cfg = S.HRFDCA_CFG
    ck = torch.load(ckpt, map_location="cpu")
    if ck["l2_classes"] != l2_names:
        sys.exit("Checkpoint Level-2 classes do not match the dataset.")
    S.set_seed(ck["seed"])
    _, _, test_df = S.stratified_split(df, ck["seed"])

    if a.all:
        idxs = list(range(len(test_df)))
    elif a.samples:
        idxs = [int(x) for x in a.samples.split(",") if x]
    else:                                             # the memes already shown in xai_report.html
        idxs = sorted(int(re.search(r"sample_(\d+)", f).group(1))
                      for f in glob.glob(os.path.join(base, "samples", "sample_*.png")))
    if not idxs:
        sys.exit("No samples selected (use --samples).")
    if a.shard:
        k, m = (int(x) for x in a.shard.split("/"))
        idxs = idxs[k::m]
    cache = os.path.join(out, "cache"); os.makedirs(cache, exist_ok=True)
    amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if device.type == "cuda" else contextlib.nullcontext
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(cfg["text_model"])
    model = S.HRFDCA(n_l2, cfg); model.load_state_dict(ck["state_dict"]); model.to(device)
    ex = X.Explainer(model, tok, cfg)
    ds = S.AggressionDataset(test_df, tok, cfg["max_len"], S.get_transforms(False))
    print(f"Device: {device} | {len(idxs)} memes | {a.n_samples} model evaluations per explanation\n")

    seg_fn = SegmentationAlgorithm("slic", n_segments=a.segments, compactness=10, sigma=1, start_label=0)

    rows, agree, cards = [], [], []
    region_rows, faith_rows, seg_store = [], [], {}
    for n, i in enumerate(idxs, 1):
        it = ds[i]
        ids, mask = it["input_ids"][None].to(device), it["attention_mask"][None].to(device)
        pix = it["pixel_values"][None].to(device)
        img = X.denorm(pix)                                   # (224, 224, 3) in [0, 1]
        text = test_df.loc[i, "text"]
        words = text.split()
        y1, y2 = int(it["label1"]), int(it["label2"])

        cf = os.path.join(cache, f"sample_{i:04d}.pkl")

        def compute():
            rs = ck["seed"] + i                                   # per-meme seed: reproducible, shard-independent
            lime_text = LimeTextExplainer(split_expression=r"\s+", bow=False, mask_string=tok.mask_token, random_state=rs)
            lime_img = LimeImageExplainer(random_state=rs)
            @torch.no_grad()
            def f_text(texts):                                    # list of captions -> (n, 2 + n_l2) probabilities
                res = []
                for k in range(0, len(texts), a.batch):
                    enc = tok(list(texts[k:k + a.batch]), max_length=cfg["max_len"], padding="max_length",
                              truncation=True, return_tensors="pt").to(device)
                    ex._vis_expand = enc["input_ids"].shape[0]    # the image is encoded once per batch
                    try:
                        l1, l2, _ = model({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                                           "pixel_values": pix})
                    finally:
                        ex._vis_expand = 0
                    res.append(torch.cat([l1.float().softmax(-1), l2.float().softmax(-1)], 1).cpu().numpy())
                return np.concatenate(res)

            @torch.no_grad()
            def f_img(imgs):                                      # (n, 224, 224, 3) in [0, 1] -> probabilities
                res = []
                imgs = np.asarray(imgs, dtype=np.float32)
                for k in range(0, len(imgs), a.batch):
                    x = torch.from_numpy(imgs[k:k + a.batch]).permute(0, 3, 1, 2)
                    x = ((x - MEAN_T) / STD_T).to(device)
                    l1, l2, _ = model({"input_ids": ids.expand(len(x), -1), "attention_mask": mask.expand(len(x), -1),
                                       "pixel_values": x})
                    res.append(torch.cat([l1.float().softmax(-1), l2.float().softmax(-1)], 1).cpu().numpy())
                return np.concatenate(res)

            p = f_text([text])[0]
            pred1, pred2 = int(p[:2].argmax()), int(p[2:].argmax())
            targets = [(1, pred1, X.L1_NAMES[pred1])] + ([(2, 2 + pred2, l2_names[pred2])] if pred1 == 1 else [])
            labels = [t[1] for t in targets]

            # ---- text: LIME and SHAP over words ----
            T = {}
            if len(words) >= 2:
                e = lime_text.explain_instance(text, f_text, labels=labels, num_features=len(words),
                                               num_samples=a.n_samples)
                for _, lab, _ in targets:
                    w = np.zeros(len(words))
                    for j, v in e.as_map()[lab]:
                        w[j] = v
                    T[("lime", lab)] = w
                def f_words(z):
                    return f_text([" ".join(wd if on else tok.mask_token for wd, on in zip(words, r)) for r in z])
                ke = shap.KernelExplainer(f_words, np.zeros((1, len(words))))
                np.random.seed(rs)
                sv = shap_matrix(ke.shap_values(np.ones((1, len(words))), nsamples=a.n_samples, l1_reg=False,
                                                silent=True), len(words), n_out)
                for _, lab, _ in targets:
                    T[("shap", lab)] = sv[:, lab]

            # ---- image: LIME and SHAP over superpixels ----
            e = lime_img.explain_instance(img.astype(np.double), f_img, labels=labels, top_labels=None, hide_color=0.45,
                                          num_samples=a.n_samples, batch_size=a.batch, segmentation_fn=seg_fn)
            seg = e.segments; n_seg = int(seg.max()) + 1
            grey = np.full_like(img, 0.45)
            def f_seg(z):
                return f_img(np.stack([np.where((r[seg] > 0)[..., None], img, grey) for r in z]))
            ke = shap.KernelExplainer(f_seg, np.zeros((1, n_seg)))
            np.random.seed(rs)
            sv = shap_matrix(ke.shap_values(np.ones((1, n_seg)), nsamples=a.n_samples, l1_reg=False, silent=True),
                             n_seg, n_out)
            I = {}
            for _, lab, _ in targets:
                w = np.zeros(n_seg)
                for j, v in e.local_exp[lab]:
                    w[j] = v
                I[("lime", lab)], I[("shap", lab)] = w, sv[:, lab]

            # ---- deletion test: remove the top-ranked words / regions vs the same number at random ----
            rng = np.random.default_rng(rs)
            D = {}
            for _, lab, _ in targets:
                base = float(p[lab])
                parts = [("image", n_seg, f_seg, I)] + ([("text", len(words), f_words, T)] if T else [])
                for mod, nf, fn_, A_ in parts:
                    k = max(1, int(round(a.delete_frac * nf)))
                    rnd = rng.choice(nf, k, replace=False)
                    for m in ["lime", "shap"]:
                        top = np.argsort(-A_[(m, lab)])[:k]
                        z = np.ones((2, nf)); z[0, top] = 0; z[1, rnd] = 0
                        q = fn_(z)[:, lab]
                        D[(mod, m, lab)] = (base - float(q[0]), base - float(q[1]))
            return dict(p=p, T=T, I=I, D=D, seg=seg.astype(np.int16), n_samples=a.n_samples)

        if os.path.isfile(cf):
            with open(cf, "rb") as fh:
                d = pickle.load(fh)
        else:
            with amp():
                d = compute()
            with open(cf, "wb") as fh:
                pickle.dump(d, fh)
        el = time.time() - t0
        print(f"  {n}/{len(idxs)}  sample {i} | elapsed {el / 60:.1f} min | "
              f"left ~{el / n * (len(idxs) - n) / 60:.0f} min", flush=True)
        if a.no_report:
            continue
        p, T, I, seg = d["p"], d["T"], d["I"], d["seg"].astype(int)
        n_seg = int(seg.max()) + 1
        pred1, pred2 = int(p[:2].argmax()), int(p[2:].argmax())
        targets = [(1, pred1, X.L1_NAMES[pred1])] + ([(2, 2 + pred2, l2_names[pred2])] if pred1 == 1 else [])

        # ---- image-region values, segment maps and deletion results ----
        seg_store[f"sample_{i:04d}"] = d["seg"]
        cnt = np.bincount(seg.ravel(), minlength=n_seg).astype(float)
        yy, xx = np.indices(seg.shape)
        cy = np.bincount(seg.ravel(), weights=yy.ravel(), minlength=n_seg) / np.maximum(cnt, 1) / seg.shape[0]
        cx = np.bincount(seg.ravel(), weights=xx.ravel(), minlength=n_seg) / np.maximum(cnt, 1) / seg.shape[1]
        for lvl, lab, name in targets:
            for sg in range(n_seg):
                region_rows.append(dict(idx=i, level=f"L{lvl}", predicted_class=name, segment=sg,
                                        area_frac=cnt[sg] / seg.size, cx=cx[sg], cy=cy[sg],
                                        lime=I[("lime", lab)][sg], shap=I[("shap", lab)][sg]))
            for (mod, m, lb), (dt, dr) in d.get("D", {}).items():
                if lb == lab:
                    faith_rows.append(dict(idx=i, level=f"L{lvl}", predicted_class=name, modality=mod, method=m,
                                           drop_top=dt, drop_random=dr))

        # ---- figure ----
        fig, axes = plt.subplots(len(targets), 3, figsize=(9.6, 3.5 * len(targets)), facecolor=X.C_SURF, squeeze=False)
        for r, (lvl, lab, name) in enumerate(targets):
            axes[r, 0].imshow(img); axes[r, 0].set_title(f"Meme (Level {lvl}: {name})", fontsize=9)
            for c, m in [(1, "lime"), (2, "shap")]:
                h = I[(m, lab)][seg]; lim = float(np.abs(h).max()) or 1.0
                axes[r, c].imshow(img); axes[r, c].imshow(h, cmap=DIVERGING, vmin=-lim, vmax=lim, alpha=0.62)
                axes[r, c].set_title(f"{m.upper()}: image regions (Level {lvl})", fontsize=9)
            for ax in axes[r]:
                ax.axis("off")
        plt.tight_layout(); fn = f"figures/sample_{i:04d}.png"
        plt.savefig(os.path.join(out, fn), dpi=100); plt.close()

        # ---- tables + report card ----
        lines = []
        for lvl, lab, name in targets:
            if T:
                for j, wd in enumerate(words):
                    rows.append(dict(idx=i, level=f"L{lvl}", predicted_class=name, position=j, word=wd,
                                     lime=T[("lime", lab)][j], shap=T[("shap", lab)][j]))
                for m in ["lime", "shap"]:
                    lines.append(f'<div class="row"><b>{m.upper()} — Level {lvl}</b> (why “{html.escape(name)}”)<br>'
                                 f'<span class="txt">{colour_words(words, T[(m, lab)])}</span></div>')
            agree.append(dict(idx=i, level=f"L{lvl}", predicted_class=name, n_words=len(words), n_segments=n_seg,
                              text_spearman=rho(T[("lime", lab)], T[("shap", lab)]) if T else np.nan,
                              image_spearman=rho(I[("lime", lab)], I[("shap", lab)])))
        true_s = X.L1_NAMES[y1] + (f" / {l2_names[y2]}" if y1 == 1 else "")
        pred_s = f"{X.L1_NAMES[pred1]} ({p[pred1]:.2f})" + (f" / {l2_names[pred2]} ({p[2 + pred2]:.2f})" if pred1 == 1 else "")
        ok = (y1 == pred1) and (y1 == 0 or y2 == pred2)
        cards.append(f'<section><h3>Test sample {i} ({html.escape(str(test_df.loc[i, "image_name"]))}) '
                     f'<span class="{"ok" if ok else "bad"}">{"correct" if ok else "misclassified"}</span></h3>'
                     f'<p><b>True:</b> {html.escape(true_s)} &nbsp; <b>Predicted:</b> {html.escape(pred_s)}</p>'
                     f'<img src="{fn}" alt="LIME and SHAP image maps">{"".join(lines)}</section>')

    if a.no_report:
        print(f"\nShard done: {len(idxs)} memes cached in {cache}"); return
    pd.DataFrame(rows).round(5).to_csv(os.path.join(out, "word_attributions.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(region_rows).round(5).to_csv(os.path.join(out, "region_attributions.csv"), index=False)
    np.savez_compressed(os.path.join(out, "segments.npz"), **seg_store)     # SLIC map per meme (224 x 224)
    Fm = pd.DataFrame(faith_rows)
    if len(Fm):
        Fm.round(5).to_csv(os.path.join(out, "faithfulness_lime_shap_per_meme.csv"), index=False)
        Fs = Fm.groupby(["level", "modality", "method"]).apply(lambda g: pd.Series(dict(
            n=len(g), removed_frac=a.delete_frac, drop_top=g["drop_top"].mean(), drop_random=g["drop_random"].mean(),
            top_beats_random=(g["drop_top"] > g["drop_random"]).mean()))).reset_index()
        Fs["n"] = Fs["n"].astype(int)
        Fs.round(4).to_csv(os.path.join(out, "faithfulness_lime_shap.csv"), index=False)
        print("\nDeletion test for LIME / SHAP (mean drop in predicted-class probability):")
        print(Fs.round(4).to_string(index=False))
    W = pd.DataFrame(rows)
    top_html = ""
    if len(W):                                        # words that matter most per predicted class
        W["w"] = W["word"].map(X.clean_word)
        G = W.dropna(subset=["w"]).groupby(["level", "predicted_class", "w"]).agg(
            shap=("shap", "mean"), lime=("lime", "mean"), count=("shap", "size")).reset_index()
        G = G[G["count"] >= min(3, max(1, len(idxs) // 20))]
        G = G.sort_values(["level", "predicted_class", "shap"], ascending=[True, True, False])
        G = G.groupby(["level", "predicted_class"]).head(15).rename(columns={"w": "word"})
        G.round(4).to_csv(os.path.join(out, "top_words_lime_shap.csv"), index=False, encoding="utf-8-sig")
        for (lv, cn), g in G.groupby(["level", "predicted_class"], sort=False):
            top_html += (f"<p><b>{lv} — {html.escape(cn)}:</b> "
                         + ", ".join(f"{html.escape(w)} ({v:+.3f})" for w, v in zip(g["word"], g["shap"])) + "</p>")
    A = pd.DataFrame(agree)
    A.round(4).to_csv(os.path.join(out, "agreement.csv"), index=False)
    summ = A.groupby("level")[["text_spearman", "image_spearman"]].agg(["mean", "median", "count"]).round(3)
    print("\nLIME vs SHAP rank agreement (Spearman) per level:"); print(summ.to_string())

    page = f"""<!doctype html><html lang="bn"><head><meta charset="utf-8"><title>HRFDCA LIME and SHAP</title>
<style>
body{{font-family:system-ui,'Noto Sans Bengali','Kohinoor Bangla',sans-serif;max-width:1100px;margin:24px auto;
padding:0 16px;color:#0b0b0b;background:#f9f9f7;line-height:1.5}}
section{{background:#fcfcfb;border:1px solid #e1e0d9;border-radius:8px;padding:14px 18px;margin:18px 0}}
img{{max-width:100%}} .txt{{font-size:17px;line-height:2}} .txt span{{padding:2px 3px;border-radius:3px}}
.row{{margin:10px 0}} h3{{margin:0 0 6px;font-size:16px}} .key span{{padding:2px 6px;border-radius:3px}}
.ok,.bad{{font-size:12px;font-weight:600;padding:2px 8px;border-radius:10px;border:1px solid}}
.ok{{color:#006300}} .bad{{color:#d03b3b}}
table{{border-collapse:collapse;font-size:14px}} td,th{{padding:4px 10px;border-bottom:1px solid #e1e0d9;text-align:right}}
</style></head><body>
<h1>HRFDCA — LIME and SHAP explanations</h1>
<p>Model: HRFDCA trained with <b>{html.escape(str(ck['loss']))}</b>, seed {ck['seed']} (best epoch {ck['best_epoch']}).
{len(idxs)} test memes, {a.n_samples} model evaluations per explanation. Words are removed by replacing them with
{html.escape(tok.mask_token)}; image regions are SLIC superpixels removed by filling with grey. Each method explains
the probability of the class the model predicted.</p>
<p class="key">Colours (words and image regions): <span style="background:rgba(235,104,52,.6)">supports the predicted class</span>
<span style="background:rgba(42,120,214,.6)">argues against it</span>. Hover a word for its value.</p>
<h2>Agreement between LIME and SHAP (Spearman rank correlation)</h2>{summ.to_html(border=0)}
<h2>Most influential words per predicted class (mean SHAP value)</h2>{top_html or "<p>n/a</p>"}
<h2>Examples</h2>{"".join(cards)}
</body></html>"""
    with open(os.path.join(out, "lime_shap_report.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"\nSaved to {out}:\n  lime_shap_report.html, figures/*.png, word_attributions.csv, agreement.csv, "
          f"top_words_lime_shap.csv,\n  region_attributions.csv, segments.npz, faithfulness_lime_shap(.csv, _per_meme.csv)")

if __name__ == "__main__":
    main()
