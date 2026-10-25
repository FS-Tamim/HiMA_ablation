import os, numpy as np, pandas as pd
src = os.path.join("loss_ablation_outputs", "loss_ablation_all_results.csv")
dst = os.path.join("loss_ablation_outputs", "loss_ablation_key_results.csv")
df = pd.read_csv(src)
names = {"ce": "CE (baseline)", "weighted_ce": "Weighted CE",
         "label_smooth_ce": "Label-Smoothing CE (eps=0.1)", "focal": "Focal (gamma=2)",
         "weighted_focal": "Weighted Focal (gamma=2)"}
def gmean(row, cols):
    vals = [row[c] for c in cols if c in row and pd.notna(row[c])]
    return float(np.prod(vals) ** (1.0 / len(vals))) if len(vals) == len(cols) else np.nan
cg_rec = ["l1_aggressive_recall", "l1_non_aggressive_recall"]
fg_rec = ["l2_political_aggression_recall", "l2_religious_aggression_recall",
          "l2_gendered_aggression_recall", "l2_others_recall"]
out = pd.DataFrame({"Loss_Function": df["loss"].map(names).fillna(df["loss"]), "Model": df["model"]})
for k, v in [("l1_acc","CG_A"),("l1_weighted_precision","CG_P"),("l1_weighted_recall","CG_R"),("l1_weighted_f1","CG_F1")]:
    out[v] = df.get(k)
out["CG_GMean"] = df.apply(lambda r: gmean(r, cg_rec), axis=1)
for k, v in [("l2_acc","FG_A"),("l2_weighted_precision","FG_P"),("l2_weighted_recall","FG_R"),("l2_weighted_f1","FG_F1")]:
    out[v] = df.get(k)
out["FG_GMean"] = df.apply(lambda r: gmean(r, fg_rec), axis=1)
for k, v in [("l1_aggressive_f1","Ag_F1"),("l1_non_aggressive_f1","NAg_F1"),
             ("l2_political_aggression_f1","PAg_F1"),("l2_gendered_aggression_f1","GAg_F1"),
             ("l2_religious_aggression_f1","RAg_F1"),("l2_others_f1","OT_F1")]:
    out[v] = df.get(k)
if "ce" in df["loss"].values:
    base = out[df["loss"] == "ce"].set_index("Model")
    out["Delta_CG_F1_vs_CE"] = out["CG_F1"] - out["Model"].map(base["CG_F1"])
    out["Delta_FG_F1_vs_CE"] = out["FG_F1"] - out["Model"].map(base["FG_F1"])
out = out.round(4)
out.to_csv(dst, index=False)
print(out.to_string(index=False))
print("\nBest CG_F1:", out.loc[out["CG_F1"].idxmax(), "Loss_Function"])
print("Best FG_F1:", out.loc[out["FG_F1"].idxmax(), "Loss_Function"])
print("Saved:", os.path.abspath(dst))
