# ============================================================
# LOSS-FUNCTION ABLATION - Proposed model (HRF) from hirearchical-mimosa-proposed.ipynb
#
# Pipeline kept EXACTLY as in the notebook:
#   data loading, 80-10-10 stratified split, seeds 42/123/2024, transforms, tokenizer,
#   encoders, HierarchicalResidualFusion model (fusion_dim 256, residual_hidden 256,
#   dropout 0.3), AdamW lr 2e-5 wd 1e-2, linear warmup 10%, 15 epochs, batch 8,
#   grad-clip 1.0, residual regularisation alpha 0.01, best-val-loss checkpoint.
# ONLY the classification loss of the two heads changes:
#   total = L1_loss + L2_loss + 0.01 * ||R||^2     (same structure as the notebook)
#
# Run (Windows, VS Code terminal, inside this folder):
#   powershell -ExecutionPolicy Bypass -File .\run.ps1 -DataRoot "D:\datasets"
# ============================================================

import os, gc, sys, json, warnings, traceback, random
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
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torchvision.transforms as T

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torchvision.models import swin_t, Swin_T_Weights, convnext_tiny, ConvNeXt_Tiny_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# -- 2. Configuration (UNCHANGED from the notebook) -----------
DATA_ROOT    = os.environ.get("ABLATION_DATA_ROOT", r"D:\datasets")
DATASET_PATH = os.path.join(DATA_ROOT, "hierarchical_aggression.csv")
IMAGE_DIR    = os.path.join(DATA_ROOT, "Img", "Img")
if not os.path.isdir(IMAGE_DIR) and os.path.isdir(os.path.join(DATA_ROOT, "Img")):
    IMAGE_DIR = os.path.join(DATA_ROOT, "Img")

# NOTE: the notebook header says "MuRIL + Swin-T", but these two lines (which are what
# actually ran in the notebook) load XLM-R + ConvNeXt-Tiny. Kept as-is.
# To use MuRIL + Swin-T instead, set ENCODERS = "muril_swin" below.
ENCODERS = "notebook"          # "notebook" (XLM-R + ConvNeXt-Tiny)  or  "muril_swin"
if ENCODERS == "muril_swin":
    TEXT_MODEL_NAME = "google/muril-base-cased"
    IMAGE_MODEL_FN  = swin_t
    IMAGE_WEIGHTS   = Swin_T_Weights.DEFAULT
else:
    TEXT_MODEL_NAME = "FacebookAI/xlm-roberta-base"
    IMAGE_MODEL_FN  = convnext_tiny
    IMAGE_WEIGHTS   = ConvNeXt_Tiny_Weights.DEFAULT




FUSION_DIM      = 256
RESIDUAL_HIDDEN = 256
DROPOUT         = 0.3
ALPHA_RESIDUAL  = 0.01

# -- 2b. Ablation configuration (the only new part) -----------
LOSS_CONFIGS = {
    "ce":              dict(type="ce"),                                   # notebook loss (baseline)
    "weighted_ce":     dict(type="ce", class_weight=True),
    "label_smooth_ce": dict(type="ce", label_smoothing=0.1),
    "focal":           dict(type="focal", gamma=2.0),
    "weighted_focal":  dict(type="focal", gamma=2.0, class_weight=True),
}
LOSS_NAMES = {
    "ce": "Cross-Entropy (baseline)", "weighted_ce": "Weighted CE",
    "label_smooth_ce": "Label-Smoothing CE (eps=0.1)", "focal": "Focal (gamma=2)",
    "weighted_focal": "Weighted Focal (gamma=2)",
}
# Best epoch is chosen with the notebook's validation loss (plain CE + residual term) for every
# variant, so only the training objective differs between runs.
SELECT_WITH_TRAIN_LOSS = False
SAVE_CHECKPOINTS = False       # True = save best .pt per run (~1.1 GB each)

_IN_NOTEBOOK = "ipykernel" in sys.modules
NUM_WORKERS  = int(os.environ.get("ABLATION_NUM_WORKERS", 0 if _IN_NOTEBOOK else 4))

_BASE   = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT_DIR = os.path.join(_BASE, "hrf_loss_ablation_outputs")
os.makedirs(OUT_DIR, exist_ok=True)
PER_SEED_CSV = os.path.join(OUT_DIR, "hrf_loss_ablation_per_seed.csv")
KEY_CSV      = os.path.join(OUT_DIR, "hrf_loss_ablation_key_results.csv")
SIG_CSV      = os.path.join(OUT_DIR, "hrf_loss_ablation_significance.csv")

# -- 3. Reproducibility ---------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -- 4. Image utilities ---------------------------------------
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
    candidate = os.path.join(image_dir, filename)
    if os.path.isfile(candidate):
        return candidate
    candidate = os.path.join(image_dir, os.path.basename(filename))
    if os.path.isfile(candidate):
        return candidate
    return None

# -- 5. Load dataset ------------------------------------------
def load_dataset(path, image_dir):
    df = pd.read_csv(path)
    print(f"Raw shape: {df.shape}")
    print(df[["binary", "Label"]].value_counts().head(20))
    for col in ["image_name", "Captions", "Label", "binary"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()
    df["binary"] = df["binary"].astype(str).str.strip().str.lower()
    df = df[df["binary"].isin(["yes", "no"])].copy()
    df["level1"] = (df["binary"] == "yes").astype(int)

    df["Label"] = df["Label"].astype(str).str.strip()
    agg_mask = df["level1"] == 1
    if agg_mask.sum() == 0:
        raise ValueError("No aggressive samples found.")
    le2 = LabelEncoder()
    df["level2"] = -1
    df.loc[agg_mask, "level2"] = le2.fit_transform(df.loc[agg_mask, "Label"])

    df["text"] = df["Captions"].astype(str).fillna("").str.strip()
    df["image_name"] = df["image_name"].astype(str).str.strip()
    df["image_path"] = df["image_name"].apply(lambda x: resolve_image_path(image_dir, x))

    missing_count = df["image_path"].isna().sum()
    print(f"\nMissing image paths: {missing_count}")
    if missing_count > 0:
        print(df.loc[df["image_path"].isna(), "image_name"].head(20).tolist())

    df = df[df["image_path"].notna()].copy()
    df.dropna(subset=["text", "level1"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if len(df) == 0:
        raise ValueError("Dataset became empty after resolving paths.")

    agg_mask = df["level1"] == 1
    print(f"\nFinal usable shape: {df.shape}")
    print(f"\nLevel-1 distribution:\n{df['level1'].value_counts()}")
    print(f"\nLevel-2 distribution (aggressive only):\n{df.loc[agg_mask, 'Label'].value_counts()}")
    return df, le2

# -- 6. Stratified split --------------------------------------
def stratified_split(df, train_ratio, val_ratio, test_ratio, seed):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio),
        stratify=df["level1"].astype(str), random_state=seed)
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        stratify=temp_df["level1"].astype(str), random_state=seed)
    print(f"  Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))

# -- 7. Dataset -----------------------------------------------
class MultimodalAggressionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, transform):
        self.df = df.reset_index(drop=True)
        self.texts = self.df["text"].tolist()
        self.image_paths = self.df["image_path"].tolist()
        self.label1 = self.df["level1"].tolist()
        self.label2 = self.df["level2"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len, padding="max_length",
                             truncation=True, return_tensors="pt")
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": self.transform(img),
            "label1": torch.tensor(self.label1[idx], dtype=torch.long),
            "label2": torch.tensor(self.label2[idx], dtype=torch.long),
        }

# -- 8. Backbone builders -------------------------------------
def build_image_backbone(model_fn, weights):
    model = model_fn(weights=weights)
    if hasattr(model, "head"):            # Swin
        feature_dim = model.head.in_features
        model.head = nn.Identity()
    elif hasattr(model, "classifier"):    # ConvNeXt
        feature_dim = list(model.classifier.children())[-1].in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported image model structure: {model_fn.__name__}")
    return model, feature_dim

class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

class ImageEncoder(nn.Module):
    def __init__(self, model_fn, weights):
        super().__init__()
        self.backbone, self.hidden_size = build_image_backbone(model_fn, weights)

    def forward(self, pixel_values):
        feat = self.backbone(pixel_values)
        if feat.dim() > 2:
            feat = feat.flatten(1)
        return feat

# -- 9. Hierarchical Residual Fusion Model (UNCHANGED) --------
class HierarchicalResidualFusion(nn.Module):
    def __init__(self, text_model_name, image_model_fn, image_weights, num_level2_classes,
                 fusion_dim=256, residual_hidden=256, dropout=0.3):
        super().__init__()
        self.text_encoder = TextEncoder(text_model_name)
        self.image_encoder = ImageEncoder(image_model_fn, image_weights)
        text_dim = self.text_encoder.hidden_size
        image_dim = self.image_encoder.hidden_size

        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.base_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim), nn.ReLU(), nn.Dropout(dropout))
        self.head1 = nn.Linear(fusion_dim, 2)
        self.residual_block = nn.Sequential(
            nn.Linear(fusion_dim + 2, residual_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(residual_hidden, fusion_dim))
        self.head2 = nn.Linear(fusion_dim, num_level2_classes)

    def forward(self, input_ids, attention_mask, pixel_values):
        text_feat = self.text_proj(self.text_encoder(input_ids, attention_mask))
        image_feat = self.image_proj(self.image_encoder(pixel_values))
        fused = self.base_fusion(torch.cat([text_feat, image_feat], dim=1))   # F
        logits1 = self.head1(fused)
        residual = self.residual_block(torch.cat([fused, logits1], dim=1))   # R
        refined = fused + residual                                           # F + R
        logits2 = self.head2(refined)
        return logits1, logits2, fused, residual, refined

# -- 10. LOSS FUNCTIONS (ablation) ----------------------------
class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017) with optional class weights."""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma, self.weight, self.label_smoothing = gamma, weight, label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.weight is not None:
            return loss.sum() / self.weight[targets].sum().clamp_min(1e-12)
        return loss.mean()

def inverse_freq_weights(labels, n_classes):
    """w_c = N / (K * n_c), from the TRAIN split only."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float)

def build_criterion(cfg, class_weights=None):
    w = class_weights.to(device) if (cfg.get("class_weight", False) and class_weights is not None) else None
    ls = cfg.get("label_smoothing", 0.0)
    if cfg["type"] == "ce":
        return nn.CrossEntropyLoss(weight=w, label_smoothing=ls)
    if cfg["type"] == "focal":
        return FocalLoss(gamma=cfg.get("gamma", 2.0), weight=w, label_smoothing=ls)
    raise ValueError(f"Unknown loss type: {cfg['type']}")

def hrf_loss(logits1, logits2, residual, l1, l2, crit1, crit2, alpha_residual=ALPHA_RESIDUAL):
    """Same structure as the notebook: L1 + L2 (aggressive only) + alpha * mean(R^2)."""
    loss = crit1(logits1, l1)
    agg_mask = (l1 == 1) & (l2 >= 0)
    if agg_mask.sum().item() > 0:
        loss = loss + crit2(logits2[agg_mask], l2[agg_mask])
    loss = loss + alpha_residual * residual.pow(2).mean()
    return loss, agg_mask

# -- 11. Train / eval (notebook loop, criterion passed in) ----
def train_epoch(model, loader, optimizer, scheduler, crit1, crit2):
    model.train()
    total_loss = 0.0
    for batch in loader:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        pixels = batch["pixel_values"].to(device)
        l1     = batch["label1"].to(device)
        l2     = batch["label2"].to(device)

        optimizer.zero_grad()
        logits1, logits2, fused, residual, refined = model(ids, mask, pixels)
        loss, _ = hrf_loss(logits1, logits2, residual, l1, l2, crit1, crit2)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)

def eval_epoch(model, loader, crit1, crit2):
    model.eval()
    total_loss = 0.0
    l1_true, l1_pred, l2_true, l2_pred = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            pixels = batch["pixel_values"].to(device)
            l1     = batch["label1"].to(device)
            l2     = batch["label2"].to(device)

            logits1, logits2, fused, residual, refined = model(ids, mask, pixels)
            loss, agg_mask = hrf_loss(logits1, logits2, residual, l1, l2, crit1, crit2)
            total_loss += loss.item()

            l1_true.extend(l1.cpu().numpy())
            l1_pred.extend(logits1.argmax(dim=-1).cpu().numpy())
            if agg_mask.sum().item() > 0:
                l2_true.extend(l2[agg_mask].cpu().numpy())
                l2_pred.extend(logits2[agg_mask].argmax(dim=-1).cpu().numpy())
    return total_loss / max(len(loader), 1), l1_true, l1_pred, l2_true, l2_pred

# -- 12. Metrics ----------------------------------------------
def level_metrics(y_true, y_pred, n_classes, label_names, title, prefix):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(n_classes))
    m = {
        f"{prefix}_A":        accuracy_score(y_true, y_pred),
        f"{prefix}_P":        precision_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_R":        recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_F1":       f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_MacroF1":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_MCC":      matthews_corrcoef(y_true, y_pred),
    }
    per_rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    m[f"{prefix}_GMean"] = float(np.prod(per_rec) ** (1.0 / len(per_rec)))
    per_p = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_f = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for i, short in enumerate(label_names["short"]):
        m[f"{short}_P"], m[f"{short}_R"], m[f"{short}_F1"] = per_p[i], per_rec[i], per_f[i]

    print(f"\n{'='*60}\n{title}\n{'='*60}")
    for k in ["A", "P", "R", "F1", "MacroF1", "GMean", "MCC"]:
        print(f"{k:<8}: {m[f'{prefix}_{k}']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels, target_names=label_names["long"],
                                zero_division=0))
    return m

SHORT_L2 = {"gendered aggression": "GAg", "others": "OT",
            "political aggression": "PAg", "religious aggression": "RAg"}

def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()

def plot_losses(train_losses, val_losses, title, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Val Loss", marker="s")
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=120); plt.close()

# -- 13. Experiment runner ------------------------------------
def run_experiment(df, le2, train_r, val_r, test_r, seed, loss_name, loss_cfg):
    print(f"\n{'#'*80}")
    print(f"HRF | LOSS: {loss_name} {loss_cfg} | Text: {TEXT_MODEL_NAME} | "
          f"Image: {IMAGE_MODEL_FN.__name__} | Split: {train_r}-{val_r}-{test_r} | Seed: {seed}")
    print(f"{'#'*80}")

    set_seed(seed)
    train_df, val_df, test_df = stratified_split(df, train_r, val_r, test_r, seed)

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    n_l2 = len(le2.classes_)
    l2_long = le2.classes_.tolist()
    l2_short = [SHORT_L2.get(c, c.replace(" ", "_")) for c in l2_long]

    train_ds = MultimodalAggressionDataset(train_df, tokenizer, MAX_LEN, get_transforms(train=True))
    val_ds   = MultimodalAggressionDataset(val_df, tokenizer, MAX_LEN, get_transforms(train=False))
    test_ds  = MultimodalAggressionDataset(test_df, tokenizer, MAX_LEN, get_transforms(train=False))

    dl_kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                 pin_memory=torch.cuda.is_available(), persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kw)
    val_loader   = DataLoader(val_ds, shuffle=False, **dl_kw)
    test_loader  = DataLoader(test_ds, shuffle=False, **dl_kw)

    model = HierarchicalResidualFusion(
        text_model_name=TEXT_MODEL_NAME, image_model_fn=IMAGE_MODEL_FN,
        image_weights=IMAGE_WEIGHTS, num_level2_classes=n_l2,
        fusion_dim=FUSION_DIM, residual_hidden=RESIDUAL_HIDDEN, dropout=DROPOUT).to(device)

    # ---- the only thing that differs between runs ----
    w1 = inverse_freq_weights(train_df["level1"].values, 2)
    w2 = inverse_freq_weights(train_df.loc[train_df["level1"] == 1, "level2"].values, n_l2)
    if loss_cfg.get("class_weight", False):
        print(f"  L1 class weights: {w1.numpy().round(4)}")
        print(f"  L2 class weights: {dict(zip(l2_long, w2.numpy().round(4)))}")
    crit1, crit2 = build_criterion(loss_cfg, w1), build_criterion(loss_cfg, w2)
    if SELECT_WITH_TRAIN_LOSS:
        sel1, sel2 = crit1, crit2
    else:
        sel1, sel2 = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()   # notebook's val loss

    total_steps = len(train_loader) * EPOCHS
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps)

    train_losses, val_losses = [], []
    best_val_loss, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(1, EPOCHS + 1):
        t_loss = train_epoch(model, train_loader, optimizer, scheduler, crit1, crit2)
        v_loss, *_ = eval_epoch(model, val_loader, sel1, sel2)
        train_losses.append(t_loss); val_losses.append(v_loss)
        print(f"  Epoch {epoch:02d}/{EPOCHS} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        if v_loss < best_val_loss:
            best_val_loss, best_epoch = v_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise ValueError("best_state is None. Training failed.")
    model.load_state_dict(best_state)
    print(f"  Best epoch: {best_epoch} (val loss {best_val_loss:.4f})")

    _, l1_true, l1_pred, l2_true, l2_pred = eval_epoch(model, test_loader, sel1, sel2)
    tag = f"hrf_{loss_name}_seed{seed}"

    row = dict(loss=loss_name, loss_function=LOSS_NAMES.get(loss_name, loss_name),
               text_model=TEXT_MODEL_NAME, image_model=IMAGE_MODEL_FN.__name__,
               split=f"{train_r}-{val_r}-{test_r}", seed=seed,
               best_epoch=best_epoch, best_val_loss=best_val_loss)
    row.update(level_metrics(l1_true, l1_pred, 2,
                             {"short": ["NAg", "Ag"], "long": ["Non-Aggressive", "Aggressive"]},
                             f"{tag} | Level-1 (Coarse-grained)", "CG"))
    if len(l2_true) > 0:
        row.update(level_metrics(l2_true, l2_pred, n_l2, {"short": l2_short, "long": l2_long},
                                 f"{tag} | Level-2 (Fine-grained)", "FG"))
        plot_confusion_matrix(l2_true, l2_pred, l2_short, f"FG CM | {tag}",
                              os.path.join(OUT_DIR, f"cm_fg_{tag}.png"))
    plot_confusion_matrix(l1_true, l1_pred, ["NAg", "Ag"], f"CG CM | {tag}",
                          os.path.join(OUT_DIR, f"cm_cg_{tag}.png"))
    plot_losses(train_losses, val_losses, f"Loss Curve | {tag}",
                os.path.join(OUT_DIR, f"loss_{tag}.png"))
    if SAVE_CHECKPOINTS:
        torch.save(best_state, os.path.join(OUT_DIR, f"best_{tag}.pt"))

    del model, best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row

# -- 14. Summary: key table (mean +/- std), significance, chart -
KEY_METRICS = ["CG_A", "CG_P", "CG_R", "CG_F1", "CG_MacroF1", "CG_GMean", "CG_MCC",
               "FG_A", "FG_P", "FG_R", "FG_F1", "FG_MacroF1", "FG_GMean", "FG_MCC",
               "Ag_F1", "NAg_F1", "PAg_F1", "GAg_F1", "RAg_F1", "OT_F1"]

def summarise(per_seed):
    order = [l for l in LOSS_CONFIGS if l in per_seed["loss"].values]
    metrics = [m for m in KEY_METRICS if m in per_seed.columns]

    rows = []
    for loss in order:
        sub = per_seed[per_seed["loss"] == loss]
        r = {"Loss_Function": LOSS_NAMES.get(loss, loss), "loss": loss, "n_seeds": len(sub)}
        for m in metrics:
            r[f"{m}_mean"] = sub[m].mean()
            r[f"{m}_std"] = sub[m].std(ddof=1) if len(sub) > 1 else np.nan
        rows.append(r)
    key = pd.DataFrame(rows)
    if "ce" in order:
        base = key.set_index("loss").loc["ce"]
        key["Delta_CG_F1_vs_CE"] = key["CG_F1_mean"] - base["CG_F1_mean"]
        key["Delta_FG_F1_vs_CE"] = key["FG_F1_mean"] - base["FG_F1_mean"]
    key.round(4).to_csv(KEY_CSV, index=False)

    # paired t-test vs CE across seeds (same seeds -> same splits)
    sig = []
    if "ce" in order:
        ce = per_seed[per_seed["loss"] == "ce"].set_index("seed")
        for loss in order:
            if loss == "ce":
                continue
            other = per_seed[per_seed["loss"] == loss].set_index("seed")
            common = ce.index.intersection(other.index)
            for m in ["CG_F1", "FG_F1", "CG_MacroF1", "FG_MacroF1"]:
                if len(common) >= 2 and m in other.columns:
                    t, p = stats.ttest_rel(other.loc[common, m].to_numpy(float),
                                           ce.loc[common, m].to_numpy(float))
                    sig.append(dict(loss=loss, metric=m, n_seeds=len(common),
                                    mean_diff=(other.loc[common, m] - ce.loc[common, m]).mean(),
                                    t_stat=t, p_value=p))
    if sig:
        pd.DataFrame(sig).round(4).to_csv(SIG_CSV, index=False)

    # print paper-style table
    print("\n" + "=" * 100)
    print("HRF LOSS ABLATION - test set, mean +/- std over seeds")
    print("=" * 100)
    show = pd.DataFrame({"Loss": key["Loss_Function"]})
    for m in ["CG_A", "CG_F1", "CG_GMean", "FG_A", "FG_F1", "FG_GMean", "NAg_F1"]:
        if f"{m}_mean" in key:
            show[m] = [f"{a:.4f} +/- {b:.4f}" if pd.notna(b) else f"{a:.4f}"
                       for a, b in zip(key[f"{m}_mean"], key[f"{m}_std"])]
    print(show.to_string(index=False))
    if sig:
        print("\nPaired t-test vs CE (p < 0.05 = significant):")
        print(pd.DataFrame(sig).round(4).to_string(index=False))

    # bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(key)); w = 0.2
    for i, (m, lab) in enumerate([("CG_F1", "CG weighted-F1"), ("CG_MacroF1", "CG macro-F1"),
                                  ("FG_F1", "FG weighted-F1"), ("FG_MacroF1", "FG macro-F1")]):
        if f"{m}_mean" in key:
            ax.bar(x + (i - 1.5) * w, key[f"{m}_mean"], w, yerr=key[f"{m}_std"].fillna(0),
                   capsize=3, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(key["Loss_Function"], rotation=15, ha="right")
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("Score"); ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("HRF loss-function ablation (mean +/- std over seeds)"); ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "hrf_loss_ablation_chart.png"), dpi=150)
    plt.close()

    print(f"\nSaved:\n  {PER_SEED_CSV}\n  {KEY_CSV}")
    if sig:
        print(f"  {SIG_CSV}")
    print(f"  {os.path.join(OUT_DIR, 'hrf_loss_ablation_chart.png')}")

# -- 15. Entry point ------------------------------------------
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA not available - training will be extremely slow.")
    print("Encoders:", TEXT_MODEL_NAME, "+", IMAGE_MODEL_FN.__name__)
    print("CSV exists:", os.path.isfile(DATASET_PATH), "|", DATASET_PATH)
    print("IMAGE_DIR exists:", os.path.isdir(IMAGE_DIR), "|", IMAGE_DIR)
    print("NUM_WORKERS:", NUM_WORKERS, "| outputs ->", OUT_DIR)

    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summarise(pd.read_csv(PER_SEED_CSV)); sys.exit(0)

    df, le2 = load_dataset(DATASET_PATH, IMAGE_DIR)

    # resume: skip (loss, seed) runs already saved
    done = pd.read_csv(PER_SEED_CSV) if os.path.isfile(PER_SEED_CSV) else pd.DataFrame()
    done_keys = set(zip(done["loss"], done["seed"])) if len(done) else set()
    if done_keys:
        print(f"\nResuming: {len(done_keys)} run(s) already finished will be skipped: {sorted(done_keys)}")

    rows = done.to_dict("records") if len(done) else []
    total = len(LOSS_CONFIGS) * len(SEEDS) * len(SPLITS)
    k = 0
    for (tr, vr, ter) in SPLITS:
        for loss_name, loss_cfg in LOSS_CONFIGS.items():
            for seed in SEEDS:
                k += 1
                if (loss_name, seed) in done_keys:
                    continue
                print(f"\n>>> Run {k}/{total}")
                try:
                    rows.append(run_experiment(df, le2, tr, vr, ter, seed, loss_name, loss_cfg))
                    pd.DataFrame(rows).round(6).to_csv(PER_SEED_CSV, index=False)
                except Exception as e:
                    print(f"\nERROR: loss={loss_name}, seed={seed} -> {e}")
                    traceback.print_exc()

    if not rows:
        raise SystemExit("[FATAL] No experiment completed successfully.")
    summarise(pd.DataFrame(rows))