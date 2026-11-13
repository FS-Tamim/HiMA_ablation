# ============================================================
# Redraws the LIME / SHAP image-region figures (lime_shap/figures/sample_XXXX.png) from the saved
# numbers - region_attributions.csv + segments.npz - without running the model again.
# Same layout as xai_lime_shap.py: per level, the meme, the LIME map and the SHAP map.
#
# Usage:  python redraw_lime_shap_figures.py --data-root <folder with hierarchical_aggression.csv>
# Options: --run-dir ablation_suite_outputs/xai/ce_seed42
# ============================================================
import os, argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="")
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
from matplotlib.colors import LinearSegmentedColormap
import torch

import ablation_suite as S
import xai_analysis as X

DIVERGING = LinearSegmentedColormap.from_list("bgo", ["#2a78d6", "#f0efec", "#eb6834"])   # same as xai_lime_shap.py

def main():
    ls = os.path.join(ARGS.run_dir, "lime_shap")
    out = os.path.join(ls, "figures"); os.makedirs(out, exist_ok=True)
    R = pd.read_csv(os.path.join(ls, "region_attributions.csv"))
    segs = np.load(os.path.join(ls, "segments.npz"))
    P = pd.read_csv(os.path.join(ARGS.run_dir, "per_sample.csv"))
    seed = int(pd.read_csv(os.path.join(ARGS.run_dir, "explained_model_test_metrics.csv")).iloc[0]["seed"])

    df, le2 = S.load_dataset(S.DATASET_PATH, S.IMAGE_DIR)
    S.set_seed(seed)
    _, _, test_df = S.stratified_split(df, seed)                  # same test split as training
    ds = S.AggressionDataset(test_df, None, 0, S.get_transforms(False))
    groups = {k: g.sort_values("segment") for k, g in R.groupby(["idx", "level"])}

    for i in range(len(test_df)):
        img = X.denorm(ds[i]["pixel_values"][None])
        seg = segs[f"sample_{i:04d}"].astype(int)
        levels = [lv for lv in ["L1", "L2"] if (i, lv) in groups]
        fig, axes = plt.subplots(len(levels), 3, figsize=(9.6, 3.5 * len(levels)), facecolor=X.C_SURF, squeeze=False)
        for r, lv in enumerate(levels):
            g = groups[(i, lv)]
            axes[r, 0].imshow(img)
            axes[r, 0].set_title(f"Meme (Level {lv[1]}: {g['predicted_class'].iloc[0]})", fontsize=9)
            for c, m in [(1, "lime"), (2, "shap")]:
                h = g[m].to_numpy()[seg]; lim = float(np.abs(h).max()) or 1.0
                axes[r, c].imshow(img); axes[r, c].imshow(h, cmap=DIVERGING, vmin=-lim, vmax=lim, alpha=0.62)
                axes[r, c].set_title(f"{m.upper()}: image regions (Level {lv[1]})", fontsize=9)
            for ax in axes[r]:
                ax.axis("off")
        plt.tight_layout(); plt.savefig(os.path.join(out, f"sample_{i:04d}.png"), dpi=100); plt.close()
        if (i + 1) % 100 == 0 or i + 1 == len(test_df):
            print(f"  {i + 1}/{len(test_df)}", flush=True)
    print("Saved", out)

if __name__ == "__main__":
    main()
