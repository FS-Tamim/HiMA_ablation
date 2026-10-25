import glob, os, re, sys
import pandas as pd

HERE = os.getcwd()
OUT_DIR = os.path.join(HERE, "loss_ablation_outputs")
OUT_CSV = os.path.join(OUT_DIR, "loss_ablation_all_results.csv")

def read_log(path):
    raw = open(path, "rb").read()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    elif raw.startswith(b"\xef\xbb\xbf"):
        text = raw.decode("utf-8-sig")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-16", errors="replace") if raw[1:2] == b"\x00" else raw.decode("cp1252", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")

def slug(s):
    return re.sub(r"[^0-9a-zA-Z]+", "_", s.strip()).strip("_").lower()

HEADER_RE = re.compile(r"^(?P<tag>\S+) \| Level-(?P<lvl>[12]) \((?:Binary|Fine-Grained)\)\s*$")
METRIC_RE = re.compile(r"^(Accuracy|Precision|Recall|F1|Macro-F1)\s*:\s*([\d.]+)\s*$")
ROW_RE    = re.compile(r"^\s*(?P<name>.+?)\s+(?P<p>\d\.\d+)\s+(?P<r>\d\.\d+)\s+(?P<f>\d\.\d+)\s+(?P<s>\d+)\s*$")
LOSS_RE   = re.compile(r"^LOSS:\s*(?P<loss>\S+)")
BEST_RE   = re.compile(r"^\s*Best epoch:\s*(?P<ep>\d+)\s*\(val loss\s*(?P<vl>[\d.]+)\)")
KEYS = {"Accuracy": "acc", "Precision": "weighted_precision", "Recall": "weighted_recall",
        "F1": "weighted_f1", "Macro-F1": "macro_f1"}

def parse(text):
    rows, order, best = {}, [], {}
    cur_loss, cur = None, None
    for line in text.split("\n"):
        m = LOSS_RE.match(line)
        if m:
            cur_loss, cur = m.group("loss"), None; continue
        m = BEST_RE.match(line)
        if m and cur_loss:
            best[cur_loss] = (int(m.group("ep")), float(m.group("vl"))); continue
        m = HEADER_RE.match(line.strip())
        if m:
            tag = m.group("tag"); loss, _, rest = tag.partition("__")
            key = (loss, rest.split("_split")[0])
            if key not in rows:
                rows[key] = {"loss": loss, "model": key[1]}; order.append(key)
            cur = (rows[key], "l" + m.group("lvl")); continue
        if cur is None:
            continue
        row, pre = cur
        m = METRIC_RE.match(line.strip())
        if m:
            row[f"{pre}_{KEYS[m.group(1)]}"] = float(m.group(2)); continue
        m = ROW_RE.match(line)
        if m:
            n = slug(m.group("name"))
            row[f"{pre}_{n}_precision"] = float(m.group("p"))
            row[f"{pre}_{n}_recall"]    = float(m.group("r"))
            row[f"{pre}_{n}_f1"]        = float(m.group("f"))
            row[f"{pre}_{n}_support"]   = int(m.group("s"))
    out = []
    for k in order:
        r = rows[k]
        if r["loss"] in best:
            r["best_epoch"], r["best_val_loss"] = best[r["loss"]]
        out.append(r)
    return out

logs = sorted(glob.glob(os.path.join(HERE, "ablation_log_*.txt")), key=os.path.getmtime)
if not logs:
    sys.exit("No ablation_log_*.txt found in this folder.")
print("Reading log:", logs[-1])
res = parse(read_log(logs[-1]))
if not res:
    sys.exit("No finished runs in the log yet.")

df = pd.DataFrame(res)
front = ["loss", "model", "best_epoch", "best_val_loss",
         "l1_acc", "l1_weighted_precision", "l1_weighted_recall", "l1_weighted_f1", "l1_macro_f1",
         "l2_acc", "l2_weighted_precision", "l2_weighted_recall", "l2_weighted_f1", "l2_macro_f1"]
front = [c for c in front if c in df.columns]
df = df[front + [c for c in df.columns if c not in front]]

if "ce" in df["loss"].values:
    for c in ["l1_weighted_f1", "l1_macro_f1", "l2_weighted_f1", "l2_macro_f1"]:
        if c in df.columns:
            base = df[df["loss"] == "ce"].set_index("model")[c]
            df["delta_vs_ce_" + c] = df[c] - df["model"].map(base)

os.makedirs(OUT_DIR, exist_ok=True)
df.round(4).to_csv(OUT_CSV, index=False)
show = [c for c in ["loss", "best_epoch", "l1_weighted_f1", "l1_macro_f1", "l1_non_aggressive_recall",
                    "l2_weighted_f1", "l2_macro_f1"] if c in df.columns]
print()
print(df[show].round(4).to_string(index=False))
print(f"\n{len(df)} runs, {df.shape[1]} columns")
print("Saved:", OUT_CSV)
