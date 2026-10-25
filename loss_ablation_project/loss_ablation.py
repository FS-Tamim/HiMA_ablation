# ============================================================
# LOSS-FUNCTION ABLATION — Hierarchical Bengali Aggression (MULTIMODAL)
# Same pipeline as hierarchical-mimosa-multimodal.ipynb:
#   same data loading, split (80-10-10, seed 123), transforms, tokenizers,
#   backbones, fusion heads, optimizer, LR, scheduler, epochs, batch size,
#   grad clipping, best-checkpoint logic, metrics.
# ONLY the training loss changes.
#
# Target: Windows PC + RTX 4090, inside VS Code.
#   Setup once :  powershell -ExecutionPolicy Bypass -File .\setup.ps1
#   Run        :  powershell -ExecutionPolicy Bypass -File .\run.ps1 -DataRoot "D:\datasets\hierarchical-mimosa"
# (or VS Code menu: Terminal -> Run Task -> "1. Setup environment" / "2. Run loss ablation")
# No pip install inside this script — on Windows that can silently replace CUDA torch with a CPU build.
# ============================================================

# ── 1. Imports ───────────────────────────────────────────────
import os, gc, warnings, traceback, random, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no GUI windows on Windows; figures are saved to disk
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torchvision.transforms as T

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from torchvision.models import (
    vit_b_16, ViT_B_16_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    swin_t, Swin_T_Weights,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 2. Configuration (UNCHANGED from original) ───────────────
# >>> EDIT THIS to where you extracted the Google Drive folder <<<
# Expected layout (same as on Kaggle):
#   <DATA_ROOT>\hierarchical_aggression.csv
#   <DATA_ROOT>\Img\Img\<image files>
# Set by run.ps1 via the ABLATION_DATA_ROOT environment variable; the fallback below is used otherwise.
DATA_ROOT    = os.environ.get("ABLATION_DATA_ROOT", r"D:\datasets\hierarchical-mimosa")
DATASET_PATH = os.path.join(DATA_ROOT, "hierarchical_aggression.csv")
IMAGE_DIR    = os.path.join(DATA_ROOT, "Img", "Img")
if not os.path.isdir(IMAGE_DIR) and os.path.isdir(os.path.join(DATA_ROOT, "Img")):
    IMAGE_DIR = os.path.join(DATA_ROOT, "Img")   # in case the Drive zip has only one Img level

# DataLoader workers. Windows uses "spawn": workers are fine when run as `python loss_ablation.py`
# (the __main__ guard below handles it), but hang/crash inside Jupyter on Windows -> use 0 there.
_IN_NOTEBOOK = "ipykernel" in sys.modules
NUM_WORKERS  = int(os.environ.get("ABLATION_NUM_WORKERS", 0 if _IN_NOTEBOOK else 4))

MAX_LEN      = 128
IMAGE_SIZE   = 224
BATCH_SIZE   = 8
EPOCHS       = 15
LR           = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 1e-2
SEEDS        = [123]
SPLITS       = [(0.8, 0.1, 0.1)]

TEXT_MODELS = {
    "muril": "google/muril-base-cased",
    "xlmr":  "FacebookAI/xlm-roberta-base",
}

IMAGE_MODELS = {
    "vit_b_16":      (vit_b_16, ViT_B_16_Weights.DEFAULT),
    "swin_t":        (swin_t, Swin_T_Weights.DEFAULT),
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),
}

# ── 2b. ABLATION configuration (the only new part) ───────────
# Which (fusion, text, image) combos to ablate.
# Default = your best Level-2 model from the main run (late_xlmr_swin_t, L2 F1 0.8054).
# Each run takes ~40 min on Kaggle GPU, so 5 losses x 1 model ≈ 3.5 h.
# Add more tuples if you want, e.g. ("early", "muril", "convnext_tiny") (best L1).
ABLATION_MODELS = [
    ("muril", "muril", "swin_t"),
]

# Loss variants. Applied to BOTH hierarchy levels (L1 binary head, L2 fine-grained head).
#   type          : "ce" or "focal"
#   class_weight  : inverse-frequency class weights computed from the TRAIN split
#   label_smoothing: CE label smoothing
#   gamma         : focal focusing parameter
#   lambda2       : weight on the Level-2 loss (total = L1 + lambda2 * L2); original = 1.0
LOSS_CONFIGS = {
    "ce":              dict(type="ce"),                                   # baseline (= original pipeline)
    "weighted_ce":     dict(type="ce", class_weight=True),
    "label_smooth_ce": dict(type="ce", label_smoothing=0.1),
    "focal":           dict(type="focal", gamma=2.0),
    "weighted_focal":  dict(type="focal", gamma=2.0, class_weight=True),
    # Optional hierarchy-weighting variants (uncomment to include):
    # "ce_lambda0.5":  dict(type="ce", lambda2=0.5),
    # "ce_lambda2.0":  dict(type="ce", lambda2=2.0),
}

# Checkpoint selection. False = select best epoch by plain (unweighted) CE val loss for
# every variant, so only the TRAINING objective differs between runs (fairer comparison).
# True = select by the variant's own loss on val (exactly mirrors original train==val loss).
# For the "ce" baseline both options are identical to the original pipeline.
SELECT_WITH_TRAIN_LOSS = False

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd(),
                       "loss_ablation_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 3. Reproducibility ───────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ── 4. Image utilities ───────────────────────────────────────
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
    else:
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

# ── 5. Load & preprocess dataset ─────────────────────────────
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
        raise ValueError("Dataset became empty after resolving text/image paths.")

    agg_mask = df["level1"] == 1
    print(f"\nFinal usable shape: {df.shape}")
    print(f"\nLevel-1 distribution:\n{df['level1'].value_counts()}")
    print(f"\nLevel-2 distribution (aggressive only):\n{df.loc[agg_mask, 'Label'].value_counts()}")
    return df, le2

# ── 6. Stratified split ──────────────────────────────────────
def stratified_split(df, train_ratio, val_ratio, test_ratio, seed):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    if df["level1"].nunique() < 2:
        raise ValueError("Need both level-1 classes for stratified split.")

    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio),
        stratify=df["level1"].astype(str), random_state=seed
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        stratify=temp_df["level1"].astype(str), random_state=seed
    )
    print(f"  Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))

# ── 7. Dataset class ─────────────────────────────────────────
class MultimodalAggressionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, transform):
        self.texts       = df["text"].tolist()
        self.image_paths = df["image_path"].tolist()
        self.label1      = df["level1"].tolist()
        self.label2      = df["level2"].tolist()
        self.tokenizer   = tokenizer
        self.max_len     = max_len
        self.transform   = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values":   self.transform(img),
            "label1":         torch.tensor(self.label1[idx], dtype=torch.long),
            "label2":         torch.tensor(self.label2[idx], dtype=torch.long),
        }

# ── 8. Backbone builders ─────────────────────────────────────
def build_image_backbone(model_fn, weights):
    model = model_fn(weights=weights)
    if hasattr(model, "classifier"):      # ConvNeXt
        feature_dim = list(model.classifier.children())[-1].in_features
        model.classifier = nn.Identity()
    elif hasattr(model, "heads"):         # ViT
        feature_dim = model.heads.head.in_features
        model.heads = nn.Identity()
    elif hasattr(model, "head"):          # Swin
        feature_dim = model.head.in_features
        model.head = nn.Identity()
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

# ── 9. Early Fusion Model ────────────────────────────────────
class EarlyFusionHierarchicalModel(nn.Module):
    def __init__(self, text_model_name, image_model_fn, image_weights, num_level2_classes, dropout=0.3):
        super().__init__()
        self.text_encoder = TextEncoder(text_model_name)
        self.image_encoder = ImageEncoder(image_model_fn, image_weights)
        fused_dim = self.text_encoder.hidden_size + self.image_encoder.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head1 = nn.Linear(fused_dim, 2)
        self.head2 = nn.Linear(fused_dim, num_level2_classes)

    def forward(self, input_ids, attention_mask, pixel_values):
        fused = torch.cat([self.text_encoder(input_ids, attention_mask),
                           self.image_encoder(pixel_values)], dim=1)
        fused = self.dropout(fused)
        return self.head1(fused), self.head2(fused)

# ── 10. Late Fusion Model ────────────────────────────────────
class LateFusionHierarchicalModel(nn.Module):
    def __init__(self, text_model_name, image_model_fn, image_weights, num_level2_classes, dropout=0.3):
        super().__init__()
        self.text_encoder = TextEncoder(text_model_name)
        self.image_encoder = ImageEncoder(image_model_fn, image_weights)
        text_dim = self.text_encoder.hidden_size
        image_dim = self.image_encoder.hidden_size
        self.text_dropout = nn.Dropout(dropout)
        self.image_dropout = nn.Dropout(dropout)
        self.text_head1 = nn.Linear(text_dim, 2)
        self.text_head2 = nn.Linear(text_dim, num_level2_classes)
        self.image_head1 = nn.Linear(image_dim, 2)
        self.image_head2 = nn.Linear(image_dim, num_level2_classes)

    def forward(self, input_ids, attention_mask, pixel_values):
        text_feat = self.text_dropout(self.text_encoder(input_ids, attention_mask))
        image_feat = self.image_dropout(self.image_encoder(pixel_values))
        logits1 = (self.text_head1(text_feat) + self.image_head1(image_feat)) / 2.0
        logits2 = (self.text_head2(text_feat) + self.image_head2(image_feat)) / 2.0
        return logits1, logits2

def build_fusion_model(fusion_type, text_model_path, image_model_fn, image_weights, n_l2):
    if fusion_type == "early":
        return EarlyFusionHierarchicalModel(text_model_path, image_model_fn, image_weights, n_l2)
    elif fusion_type == "late":
        return LateFusionHierarchicalModel(text_model_path, image_model_fn, image_weights, n_l2)
    raise ValueError(f"Unknown fusion type: {fusion_type}")

# ── 11. LOSS FUNCTIONS (ablation) ────────────────────────────
class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017) with optional class weights (alpha)."""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        # p_t from the unweighted, unsmoothed CE
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.weight is not None:
            return loss.sum() / self.weight[targets].sum().clamp_min(1e-12)
        return loss.mean()

def inverse_freq_weights(labels, n_classes):
    """w_c = N / (K * n_c), computed from the TRAIN split only."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float)

def build_criterion(cfg, class_weights=None):
    w = class_weights.to(device) if (cfg.get("class_weight", False) and class_weights is not None) else None
    ls = cfg.get("label_smoothing", 0.0)
    if cfg["type"] == "ce":
        return nn.CrossEntropyLoss(weight=w, label_smoothing=ls)
    if cfg["type"] == "focal":
        return FocalLoss(gamma=cfg.get("gamma", 2.0), weight=w, label_smoothing=ls)
    raise ValueError(f"Unknown loss type: {cfg['type']}")

def hierarchical_loss(logits1, logits2, l1, l2, crit1, crit2, lambda2):
    """Identical structure to original: L1 on all samples, L2 only on aggressive samples."""
    loss1 = crit1(logits1, l1)
    agg_mask = (l1 == 1) & (l2 >= 0)
    if agg_mask.sum().item() > 0:
        return loss1 + lambda2 * crit2(logits2[agg_mask], l2[agg_mask]), agg_mask
    return loss1, agg_mask

# ── 12. Training & evaluation (same loop, criterion passed in) ──
def train_epoch(model, loader, optimizer, scheduler, crit1, crit2, lambda2):
    model.train()
    total_loss = 0.0
    for batch in loader:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        pixels = batch["pixel_values"].to(device)
        l1     = batch["label1"].to(device)
        l2     = batch["label2"].to(device)

        optimizer.zero_grad()
        logits1, logits2 = model(ids, mask, pixels)
        loss, _ = hierarchical_loss(logits1, logits2, l1, l2, crit1, crit2, lambda2)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)

def eval_epoch(model, loader, crit1, crit2, lambda2):
    model.eval()
    total_loss = 0.0
    all_l1_true, all_l1_pred, all_l2_true, all_l2_pred = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            pixels = batch["pixel_values"].to(device)
            l1     = batch["label1"].to(device)
            l2     = batch["label2"].to(device)

            logits1, logits2 = model(ids, mask, pixels)
            loss, agg_mask = hierarchical_loss(logits1, logits2, l1, l2, crit1, crit2, lambda2)
            total_loss += loss.item()

            all_l1_true.extend(l1.cpu().numpy())
            all_l1_pred.extend(logits1.argmax(dim=-1).cpu().numpy())
            if agg_mask.sum().item() > 0:
                all_l2_true.extend(l2[agg_mask].cpu().numpy())
                all_l2_pred.extend(logits2[agg_mask].argmax(dim=-1).cpu().numpy())
    return total_loss / max(len(loader), 1), all_l1_true, all_l1_pred, all_l2_true, all_l2_pred

# ── 13. Metrics helpers (same; macro-F1 added to the record) ──
def compute_metrics(y_true, y_pred, label_names=None, title=""):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*60}\n{title}\n{'='*60}")
    print(f"Accuracy : {acc:.4f}\nPrecision: {pre:.4f}\nRecall   : {rec:.4f}\nF1       : {f1:.4f}\nMacro-F1 : {mf1:.4f}")
    if label_names is not None:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))
    return {"acc": acc, "pre": pre, "rec": rec, "f1": f1, "macro_f1": mf1}

def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
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

# ── 14. Experiment runner ────────────────────────────────────
def run_experiment(df, le2, fusion_type, text_model_name, text_model_path,
                   image_model_name, image_model_fn, image_weights,
                   train_r, val_r, test_r, seed, loss_name, loss_cfg):

    print(f"\n{'#'*80}")
    print(f"LOSS: {loss_name} {loss_cfg} | Fusion: {fusion_type} | Text: {text_model_name} | "
          f"Image: {image_model_name} | Split: {train_r}-{val_r}-{test_r} | Seed: {seed}")
    print(f"{'#'*80}")

    set_seed(seed)
    train_df, val_df, test_df = stratified_split(df, train_r, val_r, test_r, seed)

    tokenizer = AutoTokenizer.from_pretrained(text_model_path)
    n_l2 = len(le2.classes_)
    l2_names = le2.classes_.tolist()

    train_ds = MultimodalAggressionDataset(train_df, tokenizer, MAX_LEN, get_transforms(train=True))
    val_ds   = MultimodalAggressionDataset(val_df, tokenizer, MAX_LEN, get_transforms(train=False))
    test_ds  = MultimodalAggressionDataset(test_df, tokenizer, MAX_LEN, get_transforms(train=False))

    pin = torch.cuda.is_available()
    dl_kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=pin,
                 persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kw)

    model = build_fusion_model(fusion_type, text_model_path, image_model_fn, image_weights, n_l2).to(device)

    # ---- loss setup (the only thing that differs between runs) ----
    w1 = inverse_freq_weights(train_df["level1"].values, 2)
    w2 = inverse_freq_weights(train_df.loc[train_df["level1"] == 1, "level2"].values, n_l2)
    if loss_cfg.get("class_weight", False):
        print(f"  L1 class weights: {w1.numpy().round(4)}")
        print(f"  L2 class weights: {dict(zip(l2_names, w2.numpy().round(4)))}")
    crit1 = build_criterion(loss_cfg, w1)
    crit2 = build_criterion(loss_cfg, w2)
    lambda2 = loss_cfg.get("lambda2", 1.0)

    if SELECT_WITH_TRAIN_LOSS:
        sel_crit1, sel_crit2, sel_lambda2 = crit1, crit2, lambda2
    else:
        sel_crit1, sel_crit2, sel_lambda2 = nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), 1.0

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    train_losses, val_losses = [], []
    best_val_loss, best_state, best_epoch = float("inf"), None, -1

    for epoch in range(1, EPOCHS + 1):
        t_loss = train_epoch(model, train_loader, optimizer, scheduler, crit1, crit2, lambda2)
        v_loss, *_ = eval_epoch(model, val_loader, sel_crit1, sel_crit2, sel_lambda2)
        train_losses.append(t_loss); val_losses.append(v_loss)
        print(f"  Epoch {epoch:02d}/{EPOCHS} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")
        if v_loss < best_val_loss:
            best_val_loss, best_epoch = v_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise ValueError("best_state is None. Training failed before validation tracking.")
    model.load_state_dict(best_state)
    print(f"  Best epoch: {best_epoch} (val loss {best_val_loss:.4f})")

    _, l1_true, l1_pred, l2_true, l2_pred = eval_epoch(model, test_loader, crit1, crit2, lambda2)

    tag = (f"{loss_name}__{fusion_type}_{text_model_name}_{image_model_name}"
           f"_split{int(train_r*100)}-{int(val_r*100)}-{int(test_r*100)}_seed{seed}")

    m1 = compute_metrics(l1_true, l1_pred, label_names=["Non-Aggressive", "Aggressive"],
                         title=f"{tag} | Level-1 (Binary)")
    l1_nonagg_recall = recall_score(l1_true, l1_pred, labels=[0], average=None, zero_division=0)[0]

    m2 = {"acc": np.nan, "pre": np.nan, "rec": np.nan, "f1": np.nan, "macro_f1": np.nan}
    if len(l2_true) > 0:
        m2 = compute_metrics(l2_true, l2_pred, label_names=l2_names, title=f"{tag} | Level-2 (Fine-Grained)")
        plot_confusion_matrix(l2_true, l2_pred, l2_names, f"Level-2 CM | {tag}", f"{OUT_DIR}/cm_l2_{tag}.png")

    plot_confusion_matrix(l1_true, l1_pred, ["Non-Agg", "Agg"], f"Level-1 CM | {tag}", f"{OUT_DIR}/cm_l1_{tag}.png")
    plot_losses(train_losses, val_losses, f"Loss Curve | {tag}", f"{OUT_DIR}/loss_{tag}.png")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dict(
        loss=loss_name, fusion=fusion_type, text_model=text_model_name, image_model=image_model_name,
        model=f"{fusion_type}_{text_model_name}_{image_model_name}",
        split=f"{train_r}-{val_r}-{test_r}", seed=seed, best_epoch=best_epoch,
        l1_acc=m1["acc"], l1_pre=m1["pre"], l1_rec=m1["rec"], l1_f1=m1["f1"],
        l1_macro_f1=m1["macro_f1"], l1_nonagg_recall=l1_nonagg_recall,
        l2_acc=m2["acc"], l2_pre=m2["pre"], l2_rec=m2["rec"], l2_f1=m2["f1"], l2_macro_f1=m2["macro_f1"],
    )

# ── 15. Summary ──────────────────────────────────────────────
def summarise_ablation(results_df):
    results_df.to_csv(f"{OUT_DIR}/loss_ablation_results.csv", index=False)

    cols = ["loss", "model", "best_epoch", "l1_acc", "l1_f1", "l1_macro_f1", "l1_nonagg_recall",
            "l2_acc", "l2_f1", "l2_macro_f1"]
    print("\n" + "=" * 100)
    print("LOSS ABLATION RESULTS (test set)")
    print("=" * 100)
    print(results_df[cols].round(4).to_string(index=False))

    # Delta vs CE baseline per model
    if "ce" in results_df["loss"].values:
        print("\n── Δ vs CE baseline (positive = better) ──")
        metrics = ["l1_f1", "l1_macro_f1", "l2_f1", "l2_macro_f1"]
        for m in results_df["model"].unique():
            sub = results_df[results_df["model"] == m].set_index("loss")
            if "ce" not in sub.index:
                continue
            delta = (sub[metrics] - sub.loc["ce", metrics]).round(4)
            print(f"\n{m}\n{delta.to_string()}")

    # Grouped bar plot per model
    for m in results_df["model"].unique():
        sub = results_df[results_df["model"] == m].set_index("loss")
        metrics = ["l1_f1", "l1_macro_f1", "l2_f1", "l2_macro_f1"]
        ax = sub[metrics].plot(kind="bar", figsize=(12, 5), rot=0)
        ax.set_ylim(0, 1.0); ax.set_ylabel("Score"); ax.set_title(f"Loss ablation | {m}")
        ax.legend(["L1 weighted-F1", "L1 macro-F1", "L2 weighted-F1", "L2 macro-F1"], loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{OUT_DIR}/ablation_{m}.png", dpi=130); plt.close()

    print(f"\nResults saved to {OUT_DIR}/loss_ablation_results.csv")

# ── 16. Entry Point ──────────────────────────────────────────
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — you have a CPU-only torch. Reinstall with the cu118 command at the top.")
    else:
        print("GPU:", torch.cuda.get_device_name(0))
    print("CSV exists:", os.path.isfile(DATASET_PATH), "|", DATASET_PATH)
    print("IMAGE_DIR exists:", os.path.isdir(IMAGE_DIR), "|", IMAGE_DIR)
    print("NUM_WORKERS:", NUM_WORKERS, "| outputs ->", OUT_DIR)
    df, le2 = load_dataset(DATASET_PATH, IMAGE_DIR)

    all_results = []
    for (fusion_type, text_model_name, image_model_name) in ABLATION_MODELS:
        text_model_path = TEXT_MODELS[text_model_name]
        image_model_fn, image_weights = IMAGE_MODELS[image_model_name]
        for loss_name, loss_cfg in LOSS_CONFIGS.items():
            for (tr, vr, ter) in SPLITS:
                for seed in SEEDS:
                    try:
                        res = run_experiment(
                            df=df, le2=le2, fusion_type=fusion_type,
                            text_model_name=text_model_name, text_model_path=text_model_path,
                            image_model_name=image_model_name, image_model_fn=image_model_fn,
                            image_weights=image_weights, train_r=tr, val_r=vr, test_r=ter,
                            seed=seed, loss_name=loss_name, loss_cfg=loss_cfg,
                        )
                        all_results.append(res)
                        # save after every run so a Kaggle timeout doesn't lose everything
                        pd.DataFrame(all_results).to_csv(f"{OUT_DIR}/loss_ablation_partial.csv", index=False)
                    except Exception as e:
                        print(f"\nERROR: loss={loss_name}, {fusion_type}_{text_model_name}_{image_model_name}, seed={seed} → {e}")
                        traceback.print_exc()

    if len(all_results) == 0:
        raise SystemExit("[FATAL] No experiment completed successfully.")

    summarise_ablation(pd.DataFrame(all_results))
