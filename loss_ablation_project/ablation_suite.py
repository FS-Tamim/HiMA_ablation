# ============================================================
# LOSS-FUNCTION ABLATION SUITE - HiMA / BMAD
#   Part "hrfdca" : proposed HRFDCA (MuRIL + Swin-T + bidirectional cross-attention
#                   + hierarchical residual refinement), built from Section V (Eqs. 2-24)
#   Part "text"   : text-only hierarchical baselines (mBERT, BanglaBERT, XLM-R, MuRIL)
#   Part "image"  : image-only hierarchical baselines (EfficientNet, ResNet50, ViT, ConvNeXt, Swin-T)
# Losses (applied to both hierarchy heads):
#   CE (baseline), Weighted CE, Label-Smoothing CE, Focal, Weighted Focal
#
# Shared with the original pipeline: data loading, 80-10-10 stratified split, transforms,
# hierarchical L1 + L2 (L2 on aggressive samples only), best-val-loss checkpoint, metrics.
#
# Usage (inside the project folder, with the .venv python):
#   python ablation_suite.py --part smoke      # 2-3 min per model: checks everything works
#   python ablation_suite.py --part hrfdca     # 5 losses x 3 seeds
#   python ablation_suite.py --part text       # 4 models x 5 losses
#   python ablation_suite.py --part image      # 5 models x 5 losses
#   python ablation_suite.py --part summary    # rebuild all tables from saved results
# Optional: --models muril,xlmr   --losses ce,focal   --seeds 42,123
# Runs that already finished are skipped, so you can stop and restart at any time.
# ============================================================

import os, gc, sys, argparse, warnings, traceback, random, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW, Adam
import torchvision.transforms as T
from torchvision import models as tvm

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ============================================================
# 1. CONFIGURATION
# ============================================================
DATA_ROOT    = os.environ.get("ABLATION_DATA_ROOT", r"D:\datasets")
DATASET_PATH = os.path.join(DATA_ROOT, "hierarchical_aggression.csv")
IMAGE_DIR    = os.path.join(DATA_ROOT, "Img", "Img")
if not os.path.isdir(IMAGE_DIR) and os.path.isdir(os.path.join(DATA_ROOT, "Img")):
    IMAGE_DIR = os.path.join(DATA_ROOT, "Img")

SPLIT      = (0.8, 0.1, 0.1)
IMAGE_SIZE = 224

# ---- Proposed model: HRFDCA (paper Sec. V; same training setup as the HRF notebook) ----
HRFDCA_CFG = dict(
    text_model="google/muril-base-cased", image_model="swin_t",
    max_len=128, batch_size=8, epochs=15, lr=2e-5, weight_decay=1e-2, warmup_ratio=0.1,
    optimizer="adamw",
    fusion_dim=256,        # d_f  (shared fusion space)
    num_heads=8,           # heads in each multi-head cross-attention
    token_stride=2,        # s    (Eq. 3 text-token downsampling)
    residual_hidden=256,   # hidden size of the residual refinement block
    dropout=0.3, attn_dropout=0.1,
    lambda_r=0.01,         # lambda in Eq. 24
    seeds=[42, 123, 2024],
)

# ---- Text-only baselines (paper Table 7: AdamW, lr 2e-5, batch 16, 15 epochs, warmup 500, wd 1e-2) ----
TEXT_MODELS = {
    "mbert":      "bert-base-multilingual-cased",
    "banglabert": "csebuetnlp/banglabert",
    "xlmr":       "FacebookAI/xlm-roberta-base",
    "muril":      "google/muril-base-cased",
}
TEXT_CFG = dict(max_len=128, batch_size=16, epochs=15, lr=2e-5, weight_decay=1e-2,
                warmup_steps=500, optimizer="adamw", dropout=0.3, seeds=[123])
# NOTE: Table 7 lists max length 512. Captions are short (max 111 words), so 128 tokens covers
# almost all of them and is ~4x faster. Set max_len=512 above to match Table 7 exactly.

# ---- Image-only baselines (paper Table 6: Adam, lr 2e-5, batch 32, 15 epochs, wd 5e-4) ----
IMAGE_MODELS = ["efficientnet", "resnet50", "vit_b_16", "convnext_tiny", "swin_t"]
IMAGE_CFG = dict(batch_size=32, epochs=15, lr=2e-5, weight_decay=5e-4,
                 optimizer="adam", dropout=0.3, seeds=[123])

# ---- Loss functions (the ablation) ----
LOSS_CONFIGS = {
    "ce":              dict(type="ce"),
    "weighted_ce":     dict(type="ce", class_weight=True),
    "label_smooth_ce": dict(type="ce", label_smoothing=0.1),
    "focal":           dict(type="focal", gamma=2.0),
    "weighted_focal":  dict(type="focal", gamma=2.0, class_weight=True),
}
LOSS_NAMES = {"ce": "CE", "weighted_ce": "W.CE", "label_smooth_ce": "L.S.CE",
              "focal": "Focal", "weighted_focal": "W.Focal"}
MODEL_NAMES = {"hrfdca": "HRFDCA", "mbert": "mBERT", "banglabert": "b-BERT", "xlmr": "XLM-R",
               "muril": "MuRIL", "efficientnet": "EfficientNet", "resnet50": "ResNet",
               "vit_b_16": "ViT", "convnext_tiny": "ConvNext", "swin_t": "Swin-T"}

_IN_NOTEBOOK = "ipykernel" in sys.modules
NUM_WORKERS  = int(os.environ.get("ABLATION_NUM_WORKERS", 0 if _IN_NOTEBOOK else 4))
_BASE   = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT_ROOT = os.path.join(_BASE, "ablation_suite_outputs")

# ============================================================
# 2. DATA (same as the original pipeline)
# ============================================================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_transforms(train=True):
    if train:
        return T.Compose([
            T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            T.RandomCrop(IMAGE_SIZE),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def resolve_image_path(image_dir, filename):
    if pd.isna(filename):
        return None
    filename = str(filename).strip()
    if filename == "":
        return None
    for cand in (os.path.join(image_dir, filename), os.path.join(image_dir, os.path.basename(filename))):
        if os.path.isfile(cand):
            return cand
    return None

def load_dataset(path, image_dir):
    df = pd.read_csv(path)
    for col in ["image_name", "Captions", "Label", "binary"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.copy()
    df["binary"] = df["binary"].astype(str).str.strip().str.lower()
    df = df[df["binary"].isin(["yes", "no"])].copy()
    df["level1"] = (df["binary"] == "yes").astype(int)
    df["Label"] = df["Label"].astype(str).str.strip()
    agg = df["level1"] == 1
    le2 = LabelEncoder()
    df["level2"] = -1
    df.loc[agg, "level2"] = le2.fit_transform(df.loc[agg, "Label"])
    df["text"] = df["Captions"].astype(str).fillna("").str.strip()
    df["image_name"] = df["image_name"].astype(str).str.strip()
    df["image_path"] = df["image_name"].apply(lambda x: resolve_image_path(image_dir, x))
    print(f"Missing image paths: {df['image_path'].isna().sum()}")
    df = df[df["image_path"].notna()].dropna(subset=["text", "level1"]).reset_index(drop=True)
    print(f"Usable samples: {len(df)} | L1: {df['level1'].value_counts().to_dict()} | "
          f"L2 classes: {le2.classes_.tolist()}")
    return df, le2

def stratified_split(df, seed):
    tr, vr, ter = SPLIT
    train_df, temp_df = train_test_split(df, test_size=vr + ter,
                                         stratify=df["level1"].astype(str), random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=ter / (vr + ter),
                                       stratify=temp_df["level1"].astype(str), random_state=seed)
    print(f"  Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

class AggressionDataset(Dataset):
    """Returns text tokens and/or image tensor depending on the model type."""
    def __init__(self, df, tokenizer=None, max_len=128, transform=None):
        self.texts = df["text"].tolist()
        self.paths = df["image_path"].tolist()
        self.l1 = df["level1"].tolist()
        self.l2 = df["level2"].tolist()
        self.tok, self.max_len, self.transform = tokenizer, max_len, transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        item = {"label1": torch.tensor(self.l1[i], dtype=torch.long),
                "label2": torch.tensor(self.l2[i], dtype=torch.long)}
        if self.tok is not None:
            enc = self.tok(self.texts[i], max_length=self.max_len, padding="max_length",
                           truncation=True, return_tensors="pt")
            item["input_ids"] = enc["input_ids"].squeeze(0)
            item["attention_mask"] = enc["attention_mask"].squeeze(0)
        if self.transform is not None:
            try:
                img = Image.open(self.paths[i]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
            item["pixel_values"] = self.transform(img)
        return item

# ============================================================
# 3. MODELS
# ============================================================
def build_pooled_image_backbone(name):
    """torchvision backbone returning one pooled feature vector per image."""
    if name == "efficientnet":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
        dim = m.classifier[-1].in_features; m.classifier = nn.Identity()
    elif name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        dim = m.fc.in_features; m.fc = nn.Identity()
    elif name == "vit_b_16":
        m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.DEFAULT)
        dim = m.heads.head.in_features; m.heads = nn.Identity()
    elif name == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
        dim = m.classifier[-1].in_features
        m.classifier = nn.Sequential(m.classifier[0], m.classifier[1])   # keep LayerNorm2d + Flatten
    elif name == "swin_t":
        m = tvm.swin_t(weights=tvm.Swin_T_Weights.DEFAULT)
        dim = m.head.in_features; m.head = nn.Identity()
    else:
        raise ValueError(f"Unknown image model: {name}")
    return m, dim

class TextOnlyHierarchical(nn.Module):
    """Text encoder -> [CLS] -> dropout -> Level-1 head and Level-2 head."""
    def __init__(self, model_name, n_l2, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head1, self.head2 = nn.Linear(h, 2), nn.Linear(h, n_l2)

    def forward(self, b):
        out = self.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        z = self.drop(out.last_hidden_state[:, 0, :])
        return self.head1(z), self.head2(z), None

class ImageOnlyHierarchical(nn.Module):
    """Image encoder -> pooled feature -> dropout -> Level-1 head and Level-2 head."""
    def __init__(self, name, n_l2, dropout=0.3):
        super().__init__()
        self.encoder, h = build_pooled_image_backbone(name)
        self.drop = nn.Dropout(dropout)
        self.head1, self.head2 = nn.Linear(h, 2), nn.Linear(h, n_l2)

    def forward(self, b):
        z = self.encoder(b["pixel_values"])
        if z.dim() > 2:
            z = z.flatten(1)
        z = self.drop(z)
        return self.head1(z), self.head2(z), None

def masked_mean(x, mask):
    """x: (B, T, d), mask: (B, T) with 1 = real token."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp_min(1.0)

class HRFDCA(nn.Module):
    """Hierarchical Residual Fusion with Dual Cross-Attention (paper Sec. V).

    Text  : MuRIL token embeddings H_t (Eq. 2) -> stride-s downsampling (Eq. 3)
            -> Linear + LayerNorm -> Z_t (Eq. 4)
    Image : Swin-T patch tokens H_v (Eq. 5, 7x7 = 49 tokens) -> Linear + LayerNorm -> Z_v (Eq. 6)
    Cross : A_t->v = MHCA(Q=Z_t, K=V=Z_v) (Eq. 7);  A_v->t = MHCA(Q=Z_v, K=V=Z_t) (Eq. 8)
    Pool  : p_t, p_v, p_ta, p_va (Eqs. 10-13) -> f_in = concat (Eq. 14)
    Fuse  : F = sigma(W_f f_in + b_f) (Eq. 15)
    L1    : y_1 = head1(F) (Eq. 16)
    Resid : r_in = [F; y_1] (Eq. 17) -> R = phi(W_r r_in + b_r) (Eq. 18) -> F' = F + R (Eq. 19)
    L2    : y_2 = head2(F') (Eq. 20)
    """
    def __init__(self, n_l2, cfg):
        super().__init__()
        d, s = cfg["fusion_dim"], cfg["token_stride"]
        self.stride = s
        self.text_encoder = AutoModel.from_pretrained(cfg["text_model"])
        swin = tvm.swin_t(weights=tvm.Swin_T_Weights.DEFAULT)
        self.vis_features, self.vis_norm = swin.features, swin.norm
        d_t = self.text_encoder.config.hidden_size
        d_v = swin.head.in_features

        self.text_proj = nn.Sequential(nn.Linear(d_t, d), nn.LayerNorm(d))
        self.vis_proj = nn.Sequential(nn.Linear(d_v, d), nn.LayerNorm(d))
        self.t2v = nn.MultiheadAttention(d, cfg["num_heads"], dropout=cfg["attn_dropout"], batch_first=True)
        self.v2t = nn.MultiheadAttention(d, cfg["num_heads"], dropout=cfg["attn_dropout"], batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(4 * d, d), nn.ReLU(), nn.Dropout(cfg["dropout"]))
        self.head1 = nn.Linear(d, 2)
        self.residual_block = nn.Sequential(
            nn.Linear(d + 2, cfg["residual_hidden"]), nn.ReLU(), nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["residual_hidden"], d))
        self.head2 = nn.Linear(d, n_l2)

    def forward(self, b):
        # text tokens (Eqs. 2-4)
        H_t = self.text_encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        H_t, t_mask = H_t[:, ::self.stride, :], b["attention_mask"][:, ::self.stride]
        Z_t = self.text_proj(H_t)
        # visual tokens (Eqs. 5-6)
        H_v = self.vis_norm(self.vis_features(b["pixel_values"]))      # (B, 7, 7, 768)
        Z_v = self.vis_proj(H_v.flatten(1, 2))                          # (B, 49, d)
        # bidirectional cross-attention (Eqs. 7-9)
        A_tv, _ = self.t2v(Z_t, Z_v, Z_v)                                # text queries image
        A_vt, _ = self.v2t(Z_v, Z_t, Z_t, key_padding_mask=(t_mask == 0))  # image queries text
        # pooling + fusion (Eqs. 10-15)
        f_in = torch.cat([masked_mean(Z_t, t_mask), Z_v.mean(1),
                          masked_mean(A_tv, t_mask), A_vt.mean(1)], dim=1)
        Fz = self.fusion(f_in)
        # hierarchical residual prediction (Eqs. 16-20)
        logits1 = self.head1(Fz)
        R = self.residual_block(torch.cat([Fz, logits1], dim=1))
        logits2 = self.head2(Fz + R)
        return logits1, logits2, R

# ============================================================
# 4. LOSSES
# ============================================================
class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017) with optional class weights."""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma, self.weight, self.ls = gamma, weight, label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, label_smoothing=self.ls, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.weight is not None:
            return loss.sum() / self.weight[targets].sum().clamp_min(1e-12)
        return loss.mean()

def inverse_freq_weights(labels, n_classes):
    counts = np.maximum(np.bincount(np.asarray(labels, dtype=int), minlength=n_classes), 1).astype(float)
    return torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float)

def build_criterion(cfg, w=None):
    w = w.to(device) if (cfg.get("class_weight") and w is not None) else None
    ls = cfg.get("label_smoothing", 0.0)
    if cfg["type"] == "ce":
        return nn.CrossEntropyLoss(weight=w, label_smoothing=ls)
    return FocalLoss(gamma=cfg.get("gamma", 2.0), weight=w, label_smoothing=ls)

def hierarchical_loss(out, l1, l2, crit1, crit2, lambda_r=0.0):
    logits1, logits2, R = out
    loss = crit1(logits1, l1)
    agg = (l1 == 1) & (l2 >= 0)
    if agg.any():
        loss = loss + crit2(logits2[agg], l2[agg])
    if R is not None and lambda_r > 0:
        loss = loss + lambda_r * R.pow(2).mean()          # Eq. 23-24
    return loss, agg

# ============================================================
# 5. TRAIN / EVAL
# ============================================================
def to_device(b):
    return {k: v.to(device, non_blocking=True) for k, v in b.items()}

def train_epoch(model, loader, opt, sched, crit1, crit2, lambda_r):
    model.train(); total = 0.0
    for b in loader:
        b = to_device(b)
        opt.zero_grad()
        loss, _ = hierarchical_loss(model(b), b["label1"], b["label2"], crit1, crit2, lambda_r)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        total += loss.item()
    return total / max(len(loader), 1)

@torch.no_grad()
def eval_epoch(model, loader, crit1, crit2, lambda_r):
    model.eval(); total = 0.0
    y1t, y1p, y2t, y2p = [], [], [], []
    for b in loader:
        b = to_device(b)
        out = model(b)
        loss, agg = hierarchical_loss(out, b["label1"], b["label2"], crit1, crit2, lambda_r)
        total += loss.item()
        y1t += b["label1"].cpu().tolist(); y1p += out[0].argmax(-1).cpu().tolist()
        if agg.any():
            y2t += b["label2"][agg].cpu().tolist(); y2p += out[1][agg].argmax(-1).cpu().tolist()
    return total / max(len(loader), 1), y1t, y1p, y2t, y2p

SHORT_L2 = {"gendered aggression": "GAg", "others": "OT",
            "political aggression": "PAg", "religious aggression": "RAg"}

def level_metrics(y_true, y_pred, n, short, long, prefix, title):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(n))
    m = {f"{prefix}_A": accuracy_score(y_true, y_pred),
         f"{prefix}_P": precision_score(y_true, y_pred, average="weighted", zero_division=0),
         f"{prefix}_R": recall_score(y_true, y_pred, average="weighted", zero_division=0),
         f"{prefix}_F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
         f"{prefix}_MacroF1": f1_score(y_true, y_pred, average="macro", zero_division=0),
         f"{prefix}_MCC": matthews_corrcoef(y_true, y_pred)}
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    pre = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1c = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    m[f"{prefix}_GMean"] = float(np.prod(rec) ** (1.0 / n))
    for i, s in enumerate(short):
        m[f"{s}_P"], m[f"{s}_R"], m[f"{s}_F1"] = pre[i], rec[i], f1c[i]
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print("  ".join(f"{k}={m[f'{prefix}_{k}']:.4f}" for k in ["A", "P", "R", "F1", "MacroF1", "GMean", "MCC"]))
    print(classification_report(y_true, y_pred, labels=labels, target_names=long, zero_division=0))
    return m

def save_cm(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(max(5, len(labels) + 1), max(4, len(labels))))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()

def save_curve(tr, va, title, path):
    plt.figure(figsize=(7, 3.5))
    plt.plot(range(1, len(tr) + 1), tr, "o-", label="Train"); plt.plot(range(1, len(va) + 1), va, "s-", label="Val")
    plt.title(title, fontsize=9); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()

# ============================================================
# 6. ONE EXPERIMENT
# ============================================================
def run_one(part, model_key, loss_key, seed, df, le2, out_dir, smoke=False):
    loss_cfg = LOSS_CONFIGS[loss_key]
    n_l2 = len(le2.classes_)
    l2_long = le2.classes_.tolist()
    l2_short = [SHORT_L2.get(c, c.replace(" ", "_")) for c in l2_long]
    print(f"\n{'#' * 80}\n[{part}] model={model_key} | loss={loss_key} {loss_cfg} | seed={seed}\n{'#' * 80}")
    set_seed(seed)
    train_df, val_df, test_df = stratified_split(df, seed)

    if part == "hrfdca":
        cfg = HRFDCA_CFG
        tok = AutoTokenizer.from_pretrained(cfg["text_model"])
        mk = lambda d, tr: AggressionDataset(d, tok, cfg["max_len"], get_transforms(tr))
        model = HRFDCA(n_l2, cfg)
        lambda_r = cfg["lambda_r"]
    elif part == "text":
        cfg = TEXT_CFG
        tok = AutoTokenizer.from_pretrained(TEXT_MODELS[model_key])
        mk = lambda d, tr: AggressionDataset(d, tok, cfg["max_len"], None)
        model = TextOnlyHierarchical(TEXT_MODELS[model_key], n_l2, cfg["dropout"])
        lambda_r = 0.0
    else:
        cfg = IMAGE_CFG
        mk = lambda d, tr: AggressionDataset(d, None, 0, get_transforms(tr))
        model = ImageOnlyHierarchical(model_key, n_l2, cfg["dropout"])
        lambda_r = 0.0
    model = model.to(device)

    train_ds, val_ds, test_ds = mk(train_df, True), mk(val_df, False), mk(test_df, False)
    epochs = cfg["epochs"]
    if smoke:
        epochs = 1
        train_ds = Subset(train_ds, range(min(96, len(train_ds))))
        val_ds = Subset(val_ds, range(min(48, len(val_ds))))
        test_ds = Subset(test_ds, range(min(48, len(test_ds))))

    kw = dict(batch_size=cfg["batch_size"], num_workers=NUM_WORKERS,
              pin_memory=torch.cuda.is_available(), persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    # loss (the only thing that differs between runs of the same model)
    w1 = inverse_freq_weights(train_df["level1"].values, 2)
    w2 = inverse_freq_weights(train_df.loc[train_df["level1"] == 1, "level2"].values, n_l2)
    crit1, crit2 = build_criterion(loss_cfg, w1), build_criterion(loss_cfg, w2)
    sel1, sel2 = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()   # same checkpoint rule for every loss

    steps = len(train_loader) * epochs
    if cfg["optimizer"] == "adamw":
        opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        warm = cfg.get("warmup_steps", int(steps * cfg.get("warmup_ratio", 0.0)))
        sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=min(warm, max(steps - 1, 0)),
                                                num_training_steps=steps)
    else:
        opt = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = None

    t0 = time.time()
    tr_l, va_l = [], []
    best, best_state, best_ep = float("inf"), None, -1
    for ep in range(1, epochs + 1):
        tl = train_epoch(model, train_loader, opt, sched, crit1, crit2, lambda_r)
        vl, *_ = eval_epoch(model, val_loader, sel1, sel2, lambda_r)
        tr_l.append(tl); va_l.append(vl)
        print(f"  Epoch {ep:02d}/{epochs} | Train Loss: {tl:.4f} | Val Loss: {vl:.4f} | {time.time() - t0:.0f}s")
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"  Best epoch: {best_ep} (val loss {best:.4f})")

    _, y1t, y1p, y2t, y2p = eval_epoch(model, test_loader, sel1, sel2, lambda_r)
    tag = f"{model_key}_{loss_key}_seed{seed}"
    row = dict(part=part, model=model_key, model_name=MODEL_NAMES.get(model_key, model_key),
               loss=loss_key, loss_name=LOSS_NAMES[loss_key], seed=seed,
               best_epoch=best_ep, best_val_loss=best, minutes=(time.time() - t0) / 60)
    row.update(level_metrics(y1t, y1p, 2, ["NAg", "Ag"], ["Non-Aggressive", "Aggressive"], "CG",
                             f"{tag} | Coarse-grained"))
    if y2t:
        row.update(level_metrics(y2t, y2p, n_l2, l2_short, l2_long, "FG", f"{tag} | Fine-grained"))
        save_cm(y2t, y2p, l2_short, f"FG | {tag}", os.path.join(out_dir, f"cm_fg_{tag}.png"))
    save_cm(y1t, y1p, ["NAg", "Ag"], f"CG | {tag}", os.path.join(out_dir, f"cm_cg_{tag}.png"))
    save_curve(tr_l, va_l, tag, os.path.join(out_dir, f"loss_{tag}.png"))

    del model, best_state, opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row

# ============================================================
# 7. SUMMARIES (CSV + LaTeX)
# ============================================================
KEY = ["CG_A", "CG_P", "CG_R", "CG_F1", "CG_MacroF1", "CG_GMean", "CG_MCC",
       "FG_A", "FG_P", "FG_R", "FG_F1", "FG_MacroF1", "FG_GMean", "FG_MCC",
       "Ag_F1", "NAg_F1", "PAg_F1", "GAg_F1", "RAg_F1", "OT_F1"]

def per_run_csv(part):
    return os.path.join(OUT_ROOT, part, f"{part}_per_run.csv")

def load_runs(parts=("hrfdca", "text", "image")):
    frames = [pd.read_csv(per_run_csv(p)) for p in parts if os.path.isfile(per_run_csv(p))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def fmt(m, s):
    return f"{m:.4f}" if (pd.isna(s) or s == 0) else f"{m:.4f} $\\pm$ {s:.4f}"

def summarise():
    runs = load_runs()
    if runs.empty:
        print("No finished runs yet."); return
    os.makedirs(OUT_ROOT, exist_ok=True)
    loss_order = [l for l in LOSS_CONFIGS if l in runs["loss"].values]
    model_order = [m for m in ["hrfdca"] + list(TEXT_MODELS) + IMAGE_MODELS if m in runs["model"].values]
    metrics = [k for k in KEY if k in runs.columns]

    # (1) mean +/- std per model and loss
    g = runs.groupby(["part", "model", "loss"])[metrics]
    key = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std")).join(
        runs.groupby(["part", "model", "loss"]).size().rename("n_seeds")).reset_index()
    key["model_name"] = key["model"].map(MODEL_NAMES)
    key["loss_name"] = key["loss"].map(LOSS_NAMES)
    key["_m"] = key["model"].map({m: i for i, m in enumerate(model_order)})
    key["_l"] = key["loss"].map({l: i for i, l in enumerate(loss_order)})
    key = key.sort_values(["_m", "_l"]).drop(columns=["_m", "_l"])
    for lvl in ["CG", "FG"]:
        base = key[key["loss"] == "ce"].set_index("model")[f"{lvl}_F1_mean"]
        key[f"Delta_{lvl}_F1_vs_CE"] = key[f"{lvl}_F1_mean"] - key["model"].map(base)
    key.round(4).to_csv(os.path.join(OUT_ROOT, "all_key_results.csv"), index=False)

    # (2) paper-style F1 grid: rows = loss, columns = model (like the loss table in the Springer paper)
    for lvl in ["CG", "FG"]:
        grid = key.pivot(index="loss", columns="model", values=f"{lvl}_F1_mean")
        grid = grid.reindex(index=loss_order, columns=model_order).rename(index=LOSS_NAMES, columns=MODEL_NAMES)
        grid.round(4).to_csv(os.path.join(OUT_ROOT, f"F1_grid_{lvl}.csv"))
        print(f"\n{'Coarse' if lvl == 'CG' else 'Fine'}-grained weighted F1 (rows = loss, columns = model):")
        print(grid.round(4).to_string())

    # (3) significance vs CE (only where >= 2 seeds)
    sig = []
    for m in model_order:
        ce = runs[(runs["model"] == m) & (runs["loss"] == "ce")].set_index("seed")
        for l in loss_order:
            if l == "ce":
                continue
            o = runs[(runs["model"] == m) & (runs["loss"] == l)].set_index("seed")
            common = ce.index.intersection(o.index)
            if len(common) < 2:
                continue
            for met in ["CG_F1", "FG_F1", "CG_MacroF1", "FG_MacroF1"]:
                t, p = stats.ttest_rel(o.loc[common, met].to_numpy(float), ce.loc[common, met].to_numpy(float))
                sig.append(dict(model=MODEL_NAMES[m], loss=LOSS_NAMES[l], metric=met, n_seeds=len(common),
                                mean_diff=float((o.loc[common, met] - ce.loc[common, met]).mean()),
                                t_stat=t, p_value=p))
    if sig:
        sig = pd.DataFrame(sig).round(4)
        sig.to_csv(os.path.join(OUT_ROOT, "significance_vs_CE.csv"), index=False)
        print("\nPaired t-test vs CE:"); print(sig.to_string(index=False))

    # (4) LaTeX tables
    tex = []
    # 4a) HRFDCA detailed table
    h = key[key["model"] == "hrfdca"]
    if len(h):
        cols = [("CG_A", "A"), ("CG_F1", "F1"), ("CG_GMean", "G"), ("FG_A", "A"), ("FG_F1", "F1"), ("FG_GMean", "G")]
        best = {c: h[f"{c}_mean"].max() for c, _ in cols}
        tex += [r"\begin{table}[!t]", r"\centering",
                r"\caption{Loss-function ablation on the proposed HRFDCA (test set, mean $\pm$ std over "
                + f"{int(h['n_seeds'].max())}" + r" seeds).}", r"\label{tab:loss_ablation_hrfdca}",
                r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lcccccc}", r"\hline",
                r"\multirow{2}{*}{Loss} & \multicolumn{3}{c}{Coarse-grained} & \multicolumn{3}{c}{Fine-grained} \\",
                r"\cline{2-7}", r" & A & F1 & G & A & F1 & G \\", r"\hline"]
        for _, r in h.iterrows():
            cells = []
            for c, _ in cols:
                s = fmt(r[f"{c}_mean"], r[f"{c}_std"])
                cells.append(f"\\textbf{{{s}}}" if np.isclose(r[f"{c}_mean"], best[c]) else s)
            tex.append(f"{r['loss_name']} & " + " & ".join(cells) + r" \\")
        tex += [r"\hline", r"\end{tabular}}", r"\end{table}", ""]
    # 4b) F1 grid table across all models
    if len(model_order):
        names = [MODEL_NAMES[m] for m in model_order]
        tex += [r"\begin{table*}[!t]", r"\centering",
                r"\caption{Weighted F1 across loss functions for unimodal baselines and the proposed HRFDCA.}",
                r"\label{tab:loss_ablation_all}", r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{l" + "c" * len(names) + "}", r"\hline",
                "Loss & " + " & ".join(names) + r" \\", r"\hline"]
        for lvl, title in [("CG", "Coarse-grained"), ("FG", "Fine-grained")]:
            tex.append(f"\\multicolumn{{{len(names) + 1}}}{{c}}{{\\textit{{{title}}}}} \\\\")
            grid = key.pivot(index="loss", columns="model", values=f"{lvl}_F1_mean").reindex(
                index=loss_order, columns=model_order)
            for l in loss_order:
                cells = []
                for m in model_order:
                    v = grid.loc[l, m]
                    if pd.isna(v):
                        cells.append("-")
                    else:
                        cells.append(f"\\textbf{{{v:.4f}}}" if np.isclose(v, grid[m].max()) else f"{v:.4f}")
                tex.append(f"{LOSS_NAMES[l]} & " + " & ".join(cells) + r" \\")
            tex.append(r"\hline")
        tex += [r"\end{tabular}}", r"\end{table*}"]
    with open(os.path.join(OUT_ROOT, "ablation_tables.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(tex))

    # (5) chart: F1 per model and loss
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, lvl in zip(axes, ["CG", "FG"]):
        grid = key.pivot(index="model", columns="loss", values=f"{lvl}_F1_mean").reindex(
            index=model_order, columns=loss_order)
        x = np.arange(len(grid)); w = 0.8 / max(len(loss_order), 1)
        for i, l in enumerate(loss_order):
            ax.bar(x + (i - (len(loss_order) - 1) / 2) * w, grid[l], w, label=LOSS_NAMES[l])
        ax.set_xticks(x); ax.set_xticklabels([MODEL_NAMES[m] for m in grid.index], rotation=30, ha="right")
        ax.set_title(f"{'Coarse' if lvl == 'CG' else 'Fine'}-grained weighted F1"); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0.4, 1.0)
    axes[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_ROOT, "ablation_chart.png"), dpi=150); plt.close()

    print(f"\nSaved to {OUT_ROOT}:\n  all_key_results.csv, F1_grid_CG.csv, F1_grid_FG.csv,"
          f"\n  significance_vs_CE.csv, ablation_tables.tex, ablation_chart.png")

# ============================================================
# 8. MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["smoke", "hrfdca", "text", "image", "summary"])
    ap.add_argument("--models", default="", help="comma list, e.g. muril,xlmr")
    ap.add_argument("--losses", default="", help="comma list, e.g. ce,focal")
    ap.add_argument("--seeds", default="", help="comma list, e.g. 42,123")
    a = ap.parse_args()

    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print("CSV:", DATASET_PATH, "| exists:", os.path.isfile(DATASET_PATH))
    print("Images:", IMAGE_DIR, "| exists:", os.path.isdir(IMAGE_DIR), "| NUM_WORKERS:", NUM_WORKERS)
    if a.part == "summary":
        summarise(); return

    df, le2 = load_dataset(DATASET_PATH, IMAGE_DIR)
    losses = [x for x in a.losses.split(",") if x] or list(LOSS_CONFIGS)
    seeds_arg = [int(x) for x in a.seeds.split(",") if x]

    if a.part == "smoke":
        # tiny run of every model to check downloads, shapes and the pipeline
        plan = [("hrfdca", "hrfdca")] + [("text", m) for m in TEXT_MODELS] + [("image", m) for m in IMAGE_MODELS]
        out = os.path.join(OUT_ROOT, "smoke"); os.makedirs(out, exist_ok=True)
        ok, bad = [], []
        for part, m in plan:
            for l in ["ce", "weighted_focal"]:
                try:
                    run_one(part, m, l, 123, df, le2, out, smoke=True); ok.append(f"{m}/{l}")
                except Exception as e:
                    traceback.print_exc(); bad.append(f"{m}/{l}: {e}")
        print("\n" + "=" * 70 + f"\nSMOKE TEST: {len(ok)} OK, {len(bad)} FAILED")
        for b in bad:
            print("  FAILED ->", b)
        print("All good - start the real runs." if not bad else "Send the FAILED lines above.")
        sys.exit(1 if bad else 0)

    if a.part == "hrfdca":
        models, seeds = ["hrfdca"], seeds_arg or HRFDCA_CFG["seeds"]
    elif a.part == "text":
        models, seeds = [m for m in (a.models.split(",") if a.models else TEXT_MODELS)], seeds_arg or TEXT_CFG["seeds"]
    else:
        models, seeds = [m for m in (a.models.split(",") if a.models else IMAGE_MODELS)], seeds_arg or IMAGE_CFG["seeds"]

    out = os.path.join(OUT_ROOT, a.part); os.makedirs(out, exist_ok=True)
    csv = per_run_csv(a.part)
    done = pd.read_csv(csv) if os.path.isfile(csv) else pd.DataFrame()
    done_keys = set(zip(done["model"], done["loss"], done["seed"])) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []

    plan = [(m, l, s) for m in models for l in losses for s in seeds]
    print(f"\nPlan: {len(plan)} runs ({len([p for p in plan if p in done_keys])} already done)")
    for i, (m, l, s) in enumerate(plan, 1):
        if (m, l, s) in done_keys:
            continue
        print(f"\n>>> Run {i}/{len(plan)}")
        try:
            rows.append(run_one(a.part, m, l, s, df, le2, out))
            pd.DataFrame(rows).to_csv(csv, index=False)
        except Exception as e:
            print(f"\nERROR: {m}/{l}/seed {s} -> {e}"); traceback.print_exc()
    summarise()

if __name__ == "__main__":
    main()