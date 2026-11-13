# ============================================================
# XAI ANALYSIS for the proposed HRFDCA (MuRIL + Swin-T, dual cross-attention)
#
# What it produces (ablation_suite_outputs/xai/<loss>_seed<seed>/):
#   1. Modality contribution  - test metrics with text or image removed, and exact
#                               two-player Shapley values (text vs image) per sample
#   2. Text attribution       - Integrated Gradients over the MuRIL input embeddings,
#                               summed to word level, for the Level-1 and Level-2 prediction
#   3. Image attribution      - Grad-CAM on the Swin-T patch tokens (7x7), Level-1 and Level-2
#   4. Cross-attention maps   - text->image and image->text attention of the fusion module
#   5. Faithfulness           - deletion test: remove the top-attributed words / patches and
#                               compare the confidence drop against random removal
#   6. Global word importance - most influential words per predicted class
#   7. xai_report.html        - per-sample explanations (Bengali text renders correctly in a browser)
#
# The loss-ablation runs did not keep model weights, so the first call trains one HRFDCA
# (same pipeline as ablation_suite.py, ~13 min on an RTX 4090) and saves a checkpoint;
# later calls reuse it.
#
# Usage (inside the project folder, with the .venv python):
#   python xai_analysis.py                           # CE, seed 42
#   python xai_analysis.py --loss focal --seed 123
#   python xai_analysis.py --smoke                   # tiny end-to-end check
#   python xai_analysis.py --light                   # laptop / Apple-silicon run (4 epochs, 100 test samples)
# Options: --data-root D:\datasets  --ckpt path.pt  --ig-steps 32  --n-correct 3  --n-wrong 1
#          --max-samples 0 (0 = whole test set)
# ============================================================

import os, sys, argparse, html, json

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="ce", help="training loss of the explained model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", default="", help="folder that contains hierarchical_aggression.csv")
    ap.add_argument("--ckpt", default="", help="existing checkpoint (skips training)")
    ap.add_argument("--ig-steps", type=int, default=32, help="Integrated Gradients steps")
    ap.add_argument("--ig-batch", type=int, default=16)
    ap.add_argument("--n-correct", type=int, default=3, help="correct examples per class in the report")
    ap.add_argument("--n-wrong", type=int, default=1, help="misclassified examples per class in the report")
    ap.add_argument("--max-samples", type=int, default=0, help="limit the test samples analysed (0 = all)")
    ap.add_argument("--delete-frac", type=float, default=0.2, help="fraction removed in the deletion test")
    ap.add_argument("--smoke", action="store_true", help="1-epoch model on 96 samples, 12 test samples")
    ap.add_argument("--epochs", type=int, default=0, help="training epochs (0 = the suite's 15)")
    ap.add_argument("--light", action="store_true",
                    help="for a laptop: 4 epochs, 100 test samples, 16 IG steps (unless set explicitly)")
    a = ap.parse_args()
    if a.light:
        a.epochs = a.epochs or 4
        a.max_samples = a.max_samples or 100
        a.ig_steps = 16 if a.ig_steps == 32 else a.ig_steps
    return a

ARGS = parse_args() if __name__ == "__main__" else None      # None when imported by xai_lime_shap.py
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # Apple GPU: run unsupported ops on the CPU
if ARGS and ARGS.data_root:
    os.environ["ABLATION_DATA_ROOT"] = ARGS.data_root      # must be set before the import below

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

import ablation_suite as S

device = S.device
L1_NAMES = ["Non-Aggressive", "Aggressive"]
MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
# chart colours: text = blue, image = orange (fixed), neutral ink for everything else
C_TEXT, C_IMAGE, C_INK, C_MUTED, C_GRID, C_SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

# ============================================================
# 1. MODEL + HOOKS
# ============================================================
class Explainer:
    """Wraps a trained HRFDCA. Hooks give access to (a) the text input embeddings, (b) the Swin-T
    patch tokens and (c) both cross-attention maps, without duplicating the forward pass."""

    def __init__(self, model, tok, cfg):
        self.model, self.tok, self.cfg = model.eval(), tok, cfg
        for p in model.parameters():
            p.requires_grad_(False)
        self.emb_layer = model.text_encoder.get_input_embeddings()
        self.stride = model.stride
        self._emb_override, self._vis, self._vis_leaf, self._vis_expand = None, None, False, 0
        self.attn = {}
        self.emb_layer.register_forward_hook(self._emb_hook)
        model.vis_norm.register_forward_hook(self._vis_hook)
        model.t2v.register_forward_hook(lambda m, i, o: self.attn.__setitem__("t2v", o[1].detach()))
        model.v2t.register_forward_hook(lambda m, i, o: self.attn.__setitem__("v2t", o[1].detach()))
        self.special = torch.tensor(sorted({t for t in [tok.cls_token_id, tok.sep_token_id, tok.pad_token_id]
                                            if t is not None}), device=device)
        enc = tok("", max_length=cfg["max_len"], padding="max_length", truncation=True, return_tensors="pt")
        self.null_ids, self.null_mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def _emb_hook(self, m, i, out):
        return self._emb_override if self._emb_override is not None else None

    def _vis_hook(self, m, i, out):
        if self._vis_expand:                       # one image, many interpolated texts (IG)
            out = out.expand(self._vis_expand, -1, -1, -1)
        if self._vis_leaf:                         # Grad-CAM: gradient w.r.t. the patch tokens
            out = out.detach().requires_grad_(True)
        self._vis = out
        return out

    @torch.no_grad()
    def probs(self, ids, mask, pix):
        """Softmax probabilities of both heads, shapes (B, 2) and (B, n_l2)."""
        l1, l2, _ = self.model({"input_ids": ids, "attention_mask": mask, "pixel_values": pix})
        return l1.softmax(-1), l2.softmax(-1)

    # ---- text: Integrated Gradients on the input embeddings ----
    def ig_text(self, ids, mask, pix, level, target, steps, bs):
        """ids/mask (1, T), pix (1, 3, H, W). Baseline = every real token replaced by [PAD].
        Returns per-token attribution (T,) and the completeness error |sum(attr) - (f(x) - f(base))|."""
        with torch.no_grad():
            emb = self.emb_layer(ids)
            keep = torch.isin(ids, self.special)
            base = self.emb_layer(torch.where(keep, ids, torch.full_like(ids, self.tok.pad_token_id)))
        total = torch.zeros_like(emb)
        alphas = (torch.arange(steps, device=device, dtype=emb.dtype) + 0.5) / steps
        try:
            for a in alphas.split(bs):
                n = len(a)
                x = (base + a.view(-1, 1, 1) * (emb - base)).requires_grad_(True)
                self._emb_override, self._vis_expand = x, n
                with torch.enable_grad():
                    out = self.model({"input_ids": ids.expand(n, -1), "attention_mask": mask.expand(n, -1),
                                      "pixel_values": pix})
                    g, = torch.autograd.grad(out[level][:, target].sum(), x)
                total += g.sum(0, keepdim=True)
            ends = torch.cat([base, emb])
            self._emb_override, self._vis_expand = ends, 2
            with torch.no_grad():
                out = self.model({"input_ids": ids.expand(2, -1), "attention_mask": mask.expand(2, -1),
                                  "pixel_values": pix})
            f0, f1 = out[level][:, target].tolist()
        finally:
            self._emb_override, self._vis_expand = None, 0
        attr = ((emb - base) * total / steps).sum(-1).squeeze(0)
        return attr.cpu().numpy(), abs(float(attr.sum()) - (f1 - f0))

    # ---- image: Grad-CAM on the 7x7 Swin-T tokens ----
    def gradcam(self, ids, mask, pix, level, target):
        self._vis_leaf = True
        try:
            with torch.enable_grad():
                out = self.model({"input_ids": ids, "attention_mask": mask, "pixel_values": pix})
                g, = torch.autograd.grad(out[level][0, target], self._vis)
            A = self._vis.detach()
        finally:
            self._vis_leaf = False
        cam = F.relu((A * g.mean((1, 2), keepdim=True)).sum(-1))[0]          # (7, 7)
        return (cam / cam.max().clamp_min(1e-12)).cpu().numpy()

    # ---- fusion module: cross-attention maps (head-averaged) ----
    @torch.no_grad()
    def cross_attention(self, ids, mask, pix):
        self.model({"input_ids": ids, "attention_mask": mask, "pixel_values": pix})
        kept = mask[0, ::self.stride].bool()                                  # real tokens after stride-s
        t2v = self.attn["t2v"][0][kept].mean(0)                               # (49,) text -> image
        v2t = self.attn["v2t"][0].mean(0)                                     # (T/s,) image -> text
        side = int(round(t2v.numel() ** 0.5))
        tok_score = np.zeros(mask.shape[1]); tok_score[::self.stride] = (v2t * kept).cpu().numpy()
        return t2v.view(side, side).cpu().numpy(), tok_score

# ============================================================
# 2. HELPERS
# ============================================================
def word_spans(tok, text, max_len):
    """[(char_start, char_end, [token positions])] for every word that survives truncation."""
    enc = tok(text, max_length=max_len, padding="max_length", truncation=True, return_offsets_mapping=True)
    spans = {}
    for pos, (wid, (s, e)) in enumerate(zip(enc.word_ids(0), enc["offset_mapping"])):
        if wid is None:
            continue
        if wid not in spans:
            spans[wid] = [s, e, [pos]]
        else:
            spans[wid][1] = e; spans[wid][2].append(pos)
    return [tuple(v) for _, v in sorted(spans.items())]

def to_words(token_scores, spans):
    return np.array([float(np.sum(token_scores[p])) for _, _, p in spans])

def denorm(pix):
    return np.clip(pix[0].permute(1, 2, 0).cpu().numpy() * STD + MEAN, 0, 1)

def upsample(m, size):
    return F.interpolate(torch.tensor(m)[None, None].float(), size=(size, size), mode="bilinear",
                         align_corners=False)[0, 0].numpy()

def mask_words(ids, spans, word_idx, mask_id):
    out = ids.clone()
    for w in word_idx:
        out[0, spans[w][2]] = mask_id
    return out

def mask_cells(pix, cells, side):
    """Grey out (set to the dataset mean = 0 after normalisation) the chosen cells of a side x side grid."""
    out = pix.clone(); step = pix.shape[-1] // side
    for c in cells:
        r, q = divmod(int(c), side)
        out[..., r * step:(r + 1) * step, q * step:(q + 1) * step] = 0
    return out

def colour_text(text, spans, scores, diverging=True):
    """HTML with a background per word. Diverging: orange = pushes towards the predicted class,
    blue = pushes away. Sequential: darker blue = more attention."""
    scale = float(np.abs(scores).max()) or 1.0
    parts, cur = [], 0
    for (s, e, _), v in zip(spans, scores):
        parts.append(html.escape(text[cur:s]))
        a = min(abs(v) / scale, 1.0) * 0.85
        rgb = ("235,104,52" if v > 0 else "42,120,214") if diverging else "42,120,214"
        parts.append(f'<span style="background:rgba({rgb},{a:.2f})" title="{v:+.4f}">{html.escape(text[s:e])}</span>')
        cur = e
    parts.append(html.escape(text[cur:]))
    return "".join(parts)

def clean_word(w):
    w = w.strip().lower()
    return w if (len(w) > 1 and any(ch.isalpha() for ch in w)) else None

# ============================================================
# 3. MAIN
# ============================================================
def main():
    a = ARGS
    tag = f"{a.loss}_seed{a.seed}" + (f"_ep{a.epochs}" if a.epochs else "") + ("_smoke" if a.smoke else "")
    out = os.path.join(S.OUT_ROOT, "xai", tag)
    os.makedirs(os.path.join(out, "samples"), exist_ok=True)
    print(f"Device: {device} | data: {S.DATASET_PATH} | output: {out}")

    df, le2 = S.load_dataset(S.DATASET_PATH, S.IMAGE_DIR)
    l2_names = le2.classes_.tolist(); n_l2 = len(l2_names)
    cfg = S.HRFDCA_CFG

    # ---- trained model (train once, then reuse the checkpoint) ----
    ckpt_path = a.ckpt or os.path.join(out, f"hrfdca_{tag}.pt")
    if not os.path.isfile(ckpt_path):
        print(f"\nNo checkpoint at {ckpt_path} -> training HRFDCA ({a.loss}, seed {a.seed}) first.")
        row = S.run_one("hrfdca", "hrfdca", a.loss, a.seed, df, le2, out, smoke=a.smoke, save_ckpt=ckpt_path,
                        epochs=a.epochs or None)
        pd.DataFrame([row]).to_csv(os.path.join(out, "explained_model_test_metrics.csv"), index=False)
    ck = torch.load(ckpt_path, map_location="cpu")
    if ck["l2_classes"] != l2_names:
        raise ValueError("Checkpoint Level-2 classes do not match the dataset.")
    if (ck["loss"], ck["seed"]) != (a.loss, a.seed):
        print(f"NOTE: checkpoint was trained with loss={ck['loss']} seed={ck['seed']}; using its seed for the split.")
    S.set_seed(ck["seed"])
    _, _, test_df = S.stratified_split(df, ck["seed"])          # same test split as training
    if a.smoke:
        test_df = test_df.iloc[:12].reset_index(drop=True)
    elif a.max_samples:
        test_df = test_df.iloc[:a.max_samples].reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(cfg["text_model"])
    if not tok.is_fast:
        raise RuntimeError("A fast tokenizer is needed for word-level attribution.")
    model = S.HRFDCA(n_l2, cfg)
    model.load_state_dict(ck["state_dict"]); model.to(device)
    ex = Explainer(model, tok, cfg)
    ds = S.AggressionDataset(test_df, tok, cfg["max_len"], S.get_transforms(False))
    rng = np.random.default_rng(ck["seed"])
    steps = 8 if a.smoke else a.ig_steps
    N = len(ds)
    print(f"\nAnalysing {N} test samples (IG steps = {steps})")

    # ---- per-sample analysis ----
    rec, store = [], {}
    abl = {k: {"p1": [], "p2": []} for k in ["full", "no_text", "no_image"]}
    words_global = {1: [], 2: []}
    for i in range(N):
        it = ds[i]
        ids, mask = it["input_ids"][None].to(device), it["attention_mask"][None].to(device)
        pix = it["pixel_values"][None].to(device)
        y1, y2 = int(it["label1"]), int(it["label2"])
        text = test_df.loc[i, "text"]
        spans = word_spans(tok, text, cfg["max_len"])
        zero = torch.zeros_like(pix)

        # the four modality coalitions: (text, image), (no text, image), (text, no image), (neither)
        P = [ex.probs(*c) for c in [(ids, mask, pix), (ex.null_ids, ex.null_mask, pix),
                                    (ids, mask, zero), (ex.null_ids, ex.null_mask, zero)]]
        pred1, pred2 = int(P[0][0].argmax()), int(P[0][1].argmax())
        for k, j in [("full", 0), ("no_text", 1), ("no_image", 2)]:
            abl[k]["p1"].append(int(P[j][0].argmax())); abl[k]["p2"].append(int(P[j][1].argmax()))
        r = dict(idx=i, image=test_df.loc[i, "image_name"], true1=y1, pred1=pred1, true2=y2, pred2=pred2,
                 conf1=float(P[0][0][0, pred1]), conf2=float(P[0][1][0, pred2]), n_words=len(spans))
        st = dict(text=text, spans=spans)

        for lvl, pred in [(1, pred1), (2, pred2)]:
            if lvl == 2 and pred1 == 0:            # the fine-grained head only matters for "aggressive"
                continue
            h = lvl - 1
            v = [float(p[h][0, pred]) for p in P]  # v[0]=both, v[1]=image only, v[2]=text only, v[3]=none
            r[f"shap_text_L{lvl}"] = 0.5 * ((v[0] - v[1]) + (v[2] - v[3]))
            r[f"shap_image_L{lvl}"] = 0.5 * ((v[0] - v[2]) + (v[1] - v[3]))

            attr, err = ex.ig_text(ids, mask, pix, h, pred, steps, a.ig_batch)
            w_attr = to_words(attr, spans)
            cam = ex.gradcam(ids, mask, pix, h, pred)
            st[f"ig{lvl}"], st[f"cam{lvl}"] = w_attr, cam
            r[f"ig_err_L{lvl}"] = err

            # deletion test (faithfulness)
            if len(spans) >= 2:
                k = max(1, int(round(a.delete_frac * len(spans))))
                top = np.argsort(-w_attr)[:k]; rnd = rng.choice(len(spans), k, replace=False)
                p_top = float(ex.probs(mask_words(ids, spans, top, tok.mask_token_id), mask, pix)[h][0, pred])
                p_rnd = float(ex.probs(mask_words(ids, spans, rnd, tok.mask_token_id), mask, pix)[h][0, pred])
                r[f"text_drop_top_L{lvl}"], r[f"text_drop_rand_L{lvl}"] = v[0] - p_top, v[0] - p_rnd
            side = cam.shape[0]; k = max(1, int(round(a.delete_frac * cam.size)))
            top = np.argsort(-cam.ravel())[:k]; rnd = rng.choice(cam.size, k, replace=False)
            p_top = float(ex.probs(ids, mask, mask_cells(pix, top, side))[h][0, pred])
            p_rnd = float(ex.probs(ids, mask, mask_cells(pix, rnd, side))[h][0, pred])
            r[f"img_drop_top_L{lvl}"], r[f"img_drop_rand_L{lvl}"] = v[0] - p_top, v[0] - p_rnd

            # global word statistics (attribution share of each word within its caption)
            denom = float(np.abs(w_attr).sum()) or 1.0
            for (s, e, _), val in zip(spans, w_attr):
                w = clean_word(text[s:e])
                if w:
                    words_global[lvl].append((pred, w, val / denom))

        st["t2v"], tok_att = ex.cross_attention(ids, mask, pix)
        st["v2t"] = to_words(tok_att, spans)
        st["img"] = denorm(pix)
        rec.append(r); store[i] = st
        if (i + 1) % 25 == 0 or i + 1 == N:
            print(f"  {i + 1}/{N}")

    R = pd.DataFrame(rec)
    R["pred1_name"] = R["pred1"].map(lambda x: L1_NAMES[x])
    R["pred2_name"] = R["pred2"].map(lambda x: l2_names[x])
    R.to_csv(os.path.join(out, "per_sample.csv"), index=False)

    # ---- (1) modality ablation metrics ----
    rows = []
    agg = (R["true1"] == 1).to_numpy()
    for k, name in [("full", "Text + Image"), ("no_text", "Image only (text removed)"),
                    ("no_image", "Text only (image removed)")]:
        p1, p2 = np.array(abl[k]["p1"]), np.array(abl[k]["p2"])
        row = dict(input=name, CG_A=accuracy_score(R["true1"], p1),
                   CG_F1=f1_score(R["true1"], p1, average="weighted", zero_division=0),
                   CG_MacroF1=f1_score(R["true1"], p1, average="macro", zero_division=0))
        if agg.any():
            row.update(FG_A=accuracy_score(R["true2"][agg], p2[agg]),
                       FG_F1=f1_score(R["true2"][agg], p2[agg], average="weighted", zero_division=0),
                       FG_MacroF1=f1_score(R["true2"][agg], p2[agg], average="macro", zero_division=0))
        rows.append(row)
    abl_df = pd.DataFrame(rows).round(4)
    abl_df.to_csv(os.path.join(out, "modality_ablation.csv"), index=False)
    print("\nModality ablation at test time (fine-grained on truly aggressive samples):")
    print(abl_df.to_string(index=False))

    # ---- (1b) modality Shapley values per predicted class ----
    shap_rows = []
    for lvl, col, names in [(1, "pred1", L1_NAMES), (2, "pred2", l2_names)]:
        sub = R.dropna(subset=[f"shap_text_L{lvl}"])
        for c, nm in enumerate(names):
            g = sub[sub[col] == c]
            if len(g) == 0:
                continue
            t, v = g[f"shap_text_L{lvl}"], g[f"shap_image_L{lvl}"]
            shap_rows.append(dict(level=f"L{lvl}", predicted_class=nm, n=len(g), shap_text=t.mean(),
                                  shap_image=v.mean(),
                                  text_share=(t.abs() / (t.abs() + v.abs()).clip(lower=1e-12)).mean()))
    shap_df = pd.DataFrame(shap_rows).round(4)
    shap_df.to_csv(os.path.join(out, "modality_shapley.csv"), index=False)
    print("\nMean modality Shapley value on the predicted-class probability:")
    print(shap_df.to_string(index=False))
    if len(shap_df):
        fig, ax = plt.subplots(figsize=(9, 4), facecolor=C_SURF); ax.set_facecolor(C_SURF)
        x = np.arange(len(shap_df)); w = 0.36
        ax.bar(x - w / 2 - 0.01, shap_df["shap_text"], w, color=C_TEXT, label="Text")
        ax.bar(x + w / 2 + 0.01, shap_df["shap_image"], w, color=C_IMAGE, label="Image")
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r.level}: {S.SHORT_L2.get(r.predicted_class, r.predicted_class)}\n(n={r.n})"
                            for r in shap_df.itertuples()], color=C_INK, fontsize=9)
        ax.set_ylabel("Mean Shapley value (probability)", color=C_INK)
        ax.set_title("Modality contribution to the predicted class", color=C_INK, loc="left", fontsize=11)
        ax.grid(axis="y", color=C_GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7"); ax.tick_params(colors=C_MUTED, length=0)
        ax.legend(frameon=False, labelcolor=C_INK)
        plt.tight_layout(); plt.savefig(os.path.join(out, "modality_shapley.png"), dpi=160); plt.close()

    # ---- (5) faithfulness ----
    f_rows = []
    for lvl in [1, 2]:
        for mod, nm in [("text", "Text (IG words)"), ("img", "Image (Grad-CAM cells)")]:
            ct, cr = f"{mod}_drop_top_L{lvl}", f"{mod}_drop_rand_L{lvl}"
            if ct not in R.columns:
                continue
            g = R.dropna(subset=[ct, cr])
            if len(g):
                f_rows.append(dict(level=f"L{lvl}", attribution=nm, n=len(g), removed_frac=a.delete_frac,
                                   drop_top=g[ct].mean(), drop_random=g[cr].mean(),
                                   top_beats_random=(g[ct] > g[cr]).mean()))
    f_df = pd.DataFrame(f_rows).round(4)
    f_df.to_csv(os.path.join(out, "faithfulness_deletion.csv"), index=False)
    print("\nDeletion test (mean drop in predicted-class probability):")
    print(f_df.to_string(index=False))
    ig_err = {f"L{l}": float(R[f"ig_err_L{l}"].mean()) for l in [1, 2] if f"ig_err_L{l}" in R.columns}
    print("Mean IG completeness error (logit units):", {k: round(v, 4) for k, v in ig_err.items()})

    # ---- (6) global word importance ----
    top_words = {}
    min_count = 1 if a.smoke else 3
    for lvl, names in [(1, L1_NAMES), (2, l2_names)]:
        W = pd.DataFrame(words_global[lvl], columns=["pred", "word", "share"])
        if W.empty:
            continue
        G = W.groupby(["pred", "word"])["share"].agg(mean_share="mean", count="size").reset_index()
        G = G[G["count"] >= min_count]
        G["predicted_class"] = G["pred"].map(lambda c: names[c])
        G = G.sort_values(["pred", "mean_share"], ascending=[True, False]).groupby("pred").head(20)
        G[["predicted_class", "word", "mean_share", "count"]].round(4).to_csv(
            os.path.join(out, f"top_words_L{lvl}.csv"), index=False, encoding="utf-8-sig")
        top_words[lvl] = G

    # ---- (7) report: pick examples ----
    picks = []
    cats = [("Non-Aggressive", (R["true1"] == 0), (R["pred1"] == 0))]
    for c, nm in enumerate(l2_names):
        cats.append((nm, (R["true1"] == 1) & (R["true2"] == c), (R["pred1"] == 1) & (R["pred2"] == c)))
    for nm, is_cls, is_right in cats:
        for pool, n in [(R[is_cls & is_right], a.n_correct), (R[is_cls & ~is_right], a.n_wrong)]:
            if len(pool) and n:
                picks += [(nm, int(j)) for j in rng.choice(pool["idx"].to_numpy(), min(n, len(pool)), replace=False)]

    cards = []
    for nm, i in picks:
        st, r = store[i], R.loc[i]
        has2 = "cam2" in st
        panels = [("Meme", None), ("Grad-CAM: Level 1", st["cam1"])]
        if has2:
            panels.append(("Grad-CAM: Level 2", st["cam2"]))
        panels.append(("Text→image attention", st["t2v"]))
        fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.5), facecolor=C_SURF)
        for ax, (title, m) in zip(axes, panels):
            ax.imshow(st["img"])
            if m is not None:
                m = upsample(m, st["img"].shape[0]); m = (m - m.min()) / max(m.max() - m.min(), 1e-12)
                ax.imshow(m, cmap="magma", alpha=0.55, vmin=0, vmax=1)
            ax.set_title(title, fontsize=9, color=C_INK); ax.axis("off")
        plt.tight_layout(); fn = f"samples/sample_{i:04d}.png"
        plt.savefig(os.path.join(out, fn), dpi=130); plt.close()

        true_s = L1_NAMES[r.true1] + (f" / {l2_names[r.true2]}" if r.true1 == 1 else "")
        pred_s = f"{L1_NAMES[r.pred1]} ({r.conf1:.2f})" + (f" / {l2_names[r.pred2]} ({r.conf2:.2f})" if has2 else "")
        ok = (r.true1 == r.pred1) and (r.true1 == 0 or r.true2 == r.pred2)
        lines = [f'<div class="row"><b>Level 1 words</b> (why “{L1_NAMES[r.pred1]}”)<br>'
                 f'<span class="txt">{colour_text(st["text"], st["spans"], st["ig1"])}</span></div>']
        if has2:
            lines.append(f'<div class="row"><b>Level 2 words</b> (why “{l2_names[r.pred2]}”)<br>'
                         f'<span class="txt">{colour_text(st["text"], st["spans"], st["ig2"])}</span></div>')
        lines.append(f'<div class="row"><b>Image→text attention</b><br><span class="txt">'
                     f'{colour_text(st["text"], st["spans"], st["v2t"], diverging=False)}</span></div>')
        shap = f'Level 1: text {r.shap_text_L1:+.3f}, image {r.shap_image_L1:+.3f}'
        if has2:
            shap += f' &nbsp;|&nbsp; Level 2: text {r.shap_text_L2:+.3f}, image {r.shap_image_L2:+.3f}'
        cards.append(f'<section><h3>{html.escape(nm)} — test sample {i} ({html.escape(str(r.image))}) '
                     f'<span class="{"ok" if ok else "bad"}">{"correct" if ok else "misclassified"}</span></h3>'
                     f'<p><b>True:</b> {html.escape(true_s)} &nbsp; <b>Predicted:</b> {html.escape(pred_s)}</p>'
                     f'<img src="{fn}" alt="attribution maps">{"".join(lines)}'
                     f'<p class="note"><b>Modality Shapley:</b> {shap}</p></section>')

    def table(d):
        return d.to_html(index=False, border=0, classes="t") if len(d) else "<p>n/a</p>"
    words_html = ""
    for lvl, G in top_words.items():
        for cname, g in G.groupby("predicted_class", sort=False):
            words_html += (f"<p><b>Level {lvl} — {html.escape(cname)}:</b> "
                           + ", ".join(f"{html.escape(w)} ({s:+.3f})" for w, s in zip(g["word"], g["mean_share"]))
                           + "</p>")
    page = f"""<!doctype html><html lang="bn"><head><meta charset="utf-8"><title>HRFDCA XAI report</title>
<style>
body{{font-family:system-ui,'Noto Sans Bengali','Kohinoor Bangla',sans-serif;max-width:1100px;margin:24px auto;
padding:0 16px;color:#0b0b0b;background:#f9f9f7;line-height:1.5}}
section{{background:#fcfcfb;border:1px solid #e1e0d9;border-radius:8px;padding:14px 18px;margin:18px 0}}
img{{max-width:100%}} .txt{{font-size:17px;line-height:2}} .txt span{{padding:2px 1px;border-radius:3px}}
.row{{margin:10px 0}} .note{{color:#52514e;font-size:14px}} h3{{margin:0 0 6px;font-size:16px}}
.ok,.bad{{font-size:12px;font-weight:600;padding:2px 8px;border-radius:10px;border:1px solid}}
.ok{{color:#006300}} .bad{{color:#d03b3b}}
table.t{{border-collapse:collapse;font-size:14px}} table.t td,table.t th{{padding:4px 10px;border-bottom:1px solid #e1e0d9;text-align:right}}
table.t td:first-child,table.t th:first-child{{text-align:left}}
.key span{{padding:2px 6px;border-radius:3px}}
</style></head><body>
<h1>HRFDCA explanation report</h1>
<p>Model: HRFDCA trained with <b>{html.escape(str(ck['loss']))}</b>, seed {ck['seed']} (best epoch {ck['best_epoch']}).
Analysed {N} test samples. Word attributions use Integrated Gradients ({steps} steps, [PAD] baseline);
image maps use Grad-CAM on the 7×7 Swin-T tokens.</p>
<p class="key">Word colours: <span style="background:rgba(235,104,52,.6)">supports the predicted class</span>
<span style="background:rgba(42,120,214,.6)">argues against it</span>. Hover a word for its value.
The image→text row is attention strength only; because the model keeps every {ex.stride}nd token, words whose
tokens were all dropped show no colour there.</p>
<h2>Modality contribution</h2>{table(abl_df)}<br>{table(shap_df)}
<p><img src="modality_shapley.png" alt="modality Shapley chart" style="max-width:760px"></p>
<h2>Faithfulness (deletion test)</h2>{table(f_df)}
<p class="note">Larger drop for “top” than for “random” means the attribution points at evidence the model really uses.
Mean IG completeness error: {json.dumps({k: round(v, 4) for k, v in ig_err.items()})} logit units.</p>
<h2>Most influential words per predicted class</h2>{words_html or "<p>n/a</p>"}
<h2>Examples</h2>{"".join(cards)}
</body></html>"""
    with open(os.path.join(out, "xai_report.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"\nSaved to {out}:\n  xai_report.html, per_sample.csv, modality_ablation.csv, modality_shapley.csv/.png,"
          f"\n  faithfulness_deletion.csv, top_words_L1.csv, top_words_L2.csv, samples/*.png")

if __name__ == "__main__":
    main()
