import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CAPELLA_TS_RE = re.compile(r"(\d{14})_(\d{14})")


@dataclass
class TrainConfig:
    train_triplets_csv: str = "dataset_csv/train_triplets.csv"
    val_triplets_csv: str = "dataset_csv/val_triplets.csv"
    val_images_csv: str = "dataset_csv/val_images.csv"
    out_dir: str = "convnext_change_pipeline"
    img_size: int = 256
    use_first_band_only: bool = True
    train_fraction: float = 0.1
    val_fraction: float = 0.1
    val_image_fraction: float = 1.0
    seed: int = 42
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    batch_size: int = 32
    num_workers: int = 4
    num_epochs: int = 20
    grad_accum_steps: int = 2
    lr: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 5
    grad_clip_norm: float = 1.0
    backbone_name: str = "convnext_base"
    pretrained: bool = True
    freeze_backbone: bool = False
    embed_dim: int = 256
    dropout: float = 0.20
    triplet_margin: float = 0.35
    w_batch_hard: float = 1.0
    w_standard_triplet: float = 0.7
    w_pair_bce: float = 0.25
    filter_invalid_at_init: bool = True
    min_valid_ratio: float = 0.70
    min_center_valid_ratio: float = 0.85
    valid_mask_threshold: float = 0.5
    use_spatial_hard_negative: bool = True
    spatial_hard_neg_topk: int = 32
    spatial_hard_neg_replace_prob: float = 0.70
    top_k_suspicious_pairs: int = 20
    top_k_real_events: int = 20
    top_k_artifact_events: int = 20
    top_k_timeline_figures: int = 30
    save_dpi: int = 300
    real_min_run: int = 3
    real_min_total_gain_factor: float = 1.8
    real_max_drop_frac: float = 0.25
    artifact_spike_factor: float = 2.4
    artifact_return_factor: float = 1.25
    artifact_next_drop_frac: float = 0.55


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_capella_timestamp_from_name(path_or_name: str):
    name = Path(path_or_name).name
    m = CAPELLA_TS_RE.search(name)
    if m is None:
        return None
    ts = m.group(1)
    try:
        return pd.to_datetime(ts, format="%Y%m%d%H%M%S", utc=True)
    except Exception:
        return None


def normalize_img(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros_like(img, dtype=np.float32)

    vals = img[finite]
    p1 = np.percentile(vals, 1)
    p99 = np.percentile(vals, 99)

    if p99 <= p1:
        mn, mx = vals.min(), vals.max()
        if mx <= mn:
            return np.zeros_like(img, dtype=np.float32)
        img = (img - mn) / (mx - mn + 1e-8)
    else:
        img = (img - p1) / (p99 - p1 + 1e-8)

    img = np.clip(img, 0.0, 1.0)
    return img.astype(np.float32)


def parse_bbox_coords(bbox_name: str):
    s = str(bbox_name)
    m = re.match(
        r"bbox_([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+)_([-+]?\d*\.?\d+)",
        s
    )
    if m is None:
        return None

    xmin = float(m.group(1))
    ymin = float(m.group(2))
    xmax = float(m.group(3))
    ymax = float(m.group(4))
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "cx": cx,
        "cy": cy,
    }


def read_tif_with_mask(path: str, cfg: TrainConfig):
    with rasterio.open(path) as src:
        valid_mask = src.dataset_mask().astype(np.float32) / 255.0

        if cfg.use_first_band_only:
            arr = src.read(1, out_shape=(cfg.img_size, cfg.img_size))
            img = normalize_img(arr)[None, :, :]
            mask_rs = cv2.resize(valid_mask, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_NEAREST)
            mask_rs = (mask_rs >= cfg.valid_mask_threshold).astype(np.float32)[None, :, :]
            return img.astype(np.float32), mask_rs.astype(np.float32)

        arr = src.read(out_shape=(src.count, cfg.img_size, cfg.img_size))
        img = np.stack([normalize_img(b) for b in arr], axis=0)
        mask_rs = cv2.resize(valid_mask, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_NEAREST)
        mask_rs = (mask_rs >= cfg.valid_mask_threshold).astype(np.float32)[None, :, :]
        return img.astype(np.float32), mask_rs.astype(np.float32)


def compute_valid_stats(mask: np.ndarray):
    m = mask[0]
    valid_ratio = float(m.mean())
    h, w = m.shape
    cy0, cy1 = int(h * 0.25), int(h * 0.75)
    cx0, cx1 = int(w * 0.25), int(w * 0.75)
    center_ratio = float(m[cy0:cy1, cx0:cx1].mean())
    return valid_ratio, center_ratio


def is_valid_sample(mask: np.ndarray, cfg: TrainConfig):
    valid_ratio, center_ratio = compute_valid_stats(mask)
    return (valid_ratio >= cfg.min_valid_ratio) and (center_ratio >= cfg.min_center_valid_ratio)


def random_rotate_90_pair(x: np.ndarray, m: np.ndarray):
    k = random.randint(0, 3)
    if k == 0:
        return x, m
    x = np.ascontiguousarray(np.rot90(x, k=k, axes=(1, 2)))
    m = np.ascontiguousarray(np.rot90(m, k=k, axes=(1, 2)))
    return x, m


def random_flip_pair(x: np.ndarray, m: np.ndarray):
    if random.random() < 0.5:
        x = x[:, :, ::-1]
        m = m[:, :, ::-1]
    if random.random() < 0.5:
        x = x[:, ::-1, :]
        m = m[:, ::-1, :]
    return np.ascontiguousarray(x), np.ascontiguousarray(m)


def random_crop_resize_pair(x: np.ndarray, m: np.ndarray, valid_mask_threshold: float, min_scale: float = 0.90):
    c, h, w = x.shape
    scale = random.uniform(min_scale, 1.0)
    new_h = max(64, int(h * scale))
    new_w = max(64, int(w * scale))

    y0 = random.randint(0, h - new_h) if h > new_h else 0
    x0 = random.randint(0, w - new_w) if w > new_w else 0

    crop_x = x[:, y0:y0 + new_h, x0:x0 + new_w]
    crop_m = m[:, y0:y0 + new_h, x0:x0 + new_w]

    x_rs = np.stack(
        [cv2.resize(crop_x[i], (w, h), interpolation=cv2.INTER_LINEAR) for i in range(c)],
        axis=0,
    )
    m_rs = np.stack(
        [cv2.resize(crop_m[i], (w, h), interpolation=cv2.INTER_NEAREST) for i in range(m.shape[0])],
        axis=0,
    )
    m_rs = (m_rs >= valid_mask_threshold).astype(np.float32)
    return x_rs.astype(np.float32), m_rs.astype(np.float32)


def random_translate_pair(x: np.ndarray, m: np.ndarray, valid_mask_threshold: float, max_shift: int = 16):
    c, h, w = x.shape
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    out_x = np.zeros_like(x)
    out_m = np.zeros_like(m)

    for i in range(c):
        out_x[i] = cv2.warpAffine(x[i], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    for i in range(m.shape[0]):
        out_m[i] = cv2.warpAffine(m[i], M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    out_m = (out_m >= valid_mask_threshold).astype(np.float32)
    return out_x.astype(np.float32), out_m.astype(np.float32)


def random_brightness_contrast(x: np.ndarray, contrast_range=(0.85, 1.20), brightness_range=(-0.08, 0.08)):
    alpha = random.uniform(*contrast_range)
    beta = random.uniform(*brightness_range)
    x = alpha * x + beta
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def random_gamma(x: np.ndarray, gamma_range=(0.85, 1.20)):
    gamma = random.uniform(*gamma_range)
    x = np.clip(x, 1e-6, 1.0)
    x = x ** gamma
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def add_gaussian_noise(x: np.ndarray, std_range=(0.0, 0.03)):
    std = random.uniform(*std_range)
    noise = np.random.normal(0.0, std, size=x.shape).astype(np.float32)
    x = x + noise
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def add_speckle_noise(x: np.ndarray, std_range=(0.0, 0.08)):
    noise = np.random.normal(0.0, random.uniform(*std_range), size=x.shape).astype(np.float32)
    x = x * (1.0 + noise)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def random_blur(x: np.ndarray, p: float = 0.20):
    if random.random() > p:
        return x
    k = random.choice([3, 5])
    out = np.stack([cv2.GaussianBlur(x[i], (k, k), 0) for i in range(x.shape[0])], axis=0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def strong_aug_pair(x: np.ndarray, m: np.ndarray, cfg: TrainConfig):
    x = x.copy()
    m = m.copy()

    if random.random() < 0.90:
        x, m = random_rotate_90_pair(x, m)
    if random.random() < 0.90:
        x, m = random_flip_pair(x, m)
    if random.random() < 0.60:
        x, m = random_crop_resize_pair(x, m, cfg.valid_mask_threshold, min_scale=0.90)
    if random.random() < 0.50:
        x, m = random_translate_pair(x, m, cfg.valid_mask_threshold, max_shift=16)

    if random.random() < 0.80:
        x = random_brightness_contrast(x)
    if random.random() < 0.50:
        x = random_gamma(x)
    if random.random() < 0.60:
        x = add_gaussian_noise(x)
    if random.random() < 0.60:
        x = add_speckle_noise(x)
    if random.random() < 0.20:
        x = random_blur(x, p=1.0)

    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    m = (m >= cfg.valid_mask_threshold).astype(np.float32)
    return x, m


class PairMiningDataset(Dataset):
    def __init__(self, csv_path: str, cfg: TrainConfig, fraction: float = 1.0, training: bool = False):
        df = pd.read_csv(csv_path)
        if fraction < 1.0:
            df = df.sample(frac=fraction, random_state=cfg.seed).reset_index(drop=True)

        self.cfg = cfg
        self.training = training

        if cfg.filter_invalid_at_init:
            df = self._filter_invalid_rows(df)

        self.df = df.reset_index(drop=True)
        self.bbox_to_indices = self._build_bbox_index()
        self.bbox_neighbors = self._build_bbox_neighbors() if cfg.use_spatial_hard_negative else {}

    def _filter_invalid_rows(self, df: pd.DataFrame):
        keep_rows = []
        for _, row in df.iterrows():
            try:
                _, a_mask = read_tif_with_mask(row["anchor_path"], self.cfg)
                _, p_mask = read_tif_with_mask(row["positive_path"], self.cfg)
                _, n_mask = read_tif_with_mask(row["negative_path"], self.cfg)
                ok = is_valid_sample(a_mask, self.cfg) and is_valid_sample(p_mask, self.cfg) and is_valid_sample(n_mask, self.cfg)
                if ok:
                    keep_rows.append(row)
            except Exception:
                continue
        return pd.DataFrame(keep_rows).reset_index(drop=True)

    def _build_bbox_index(self):
        bbox_to_indices = {}
        for i, row in self.df.iterrows():
            bbox = str(row["bbox_name"])
            bbox_to_indices.setdefault(bbox, []).append(i)
        return bbox_to_indices

    def _build_bbox_neighbors(self):
        unique_bboxes = list(self.bbox_to_indices.keys())
        centers = {}
        for b in unique_bboxes:
            c = parse_bbox_coords(b)
            if c is not None:
                centers[b] = (c["cx"], c["cy"])

        neighbors = {}
        bbox_list = list(centers.keys())
        arr = np.array([centers[b] for b in bbox_list], dtype=np.float64)

        for i, b in enumerate(bbox_list):
            d = np.sqrt(((arr - arr[i]) ** 2).sum(axis=1))
            order = np.argsort(d)
            close = []
            for j in order:
                bb = bbox_list[j]
                if bb == b:
                    continue
                close.append(bb)
                if len(close) >= self.cfg.spatial_hard_neg_topk:
                    break
            neighbors[b] = close
        return neighbors

    def __len__(self):
        return len(self.df)

    def _maybe_replace_negative(self, row):
        if not self.training or not self.cfg.use_spatial_hard_negative:
            return row["negative_path"]

        if random.random() > self.cfg.spatial_hard_neg_replace_prob:
            return row["negative_path"]

        bbox = str(row["bbox_name"])
        neigh = self.bbox_neighbors.get(bbox, [])
        if len(neigh) == 0:
            return row["negative_path"]

        random.shuffle(neigh)
        for nb in neigh[: min(8, len(neigh))]:
            idxs = self.bbox_to_indices.get(nb, [])
            if len(idxs) == 0:
                continue
            ridx = random.choice(idxs)
            neg_path = self.df.iloc[ridx]["anchor_path"]
            if neg_path != row["anchor_path"] and neg_path != row["positive_path"]:
                return neg_path

        return row["negative_path"]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        neg_path = self._maybe_replace_negative(row)

        a, am = read_tif_with_mask(row["anchor_path"], self.cfg)
        p, pm = read_tif_with_mask(row["positive_path"], self.cfg)
        n, nm = read_tif_with_mask(neg_path, self.cfg)

        if self.training:
            a, am = strong_aug_pair(a, am, self.cfg)
            p, pm = strong_aug_pair(p, pm, self.cfg)
            n, nm = strong_aug_pair(n, nm, self.cfg)

        return {
            "anchor": torch.from_numpy(a),
            "positive": torch.from_numpy(p),
            "negative": torch.from_numpy(n),
            "anchor_mask": torch.from_numpy(am),
            "positive_mask": torch.from_numpy(pm),
            "negative_mask": torch.from_numpy(nm),
            "anchor_path": row["anchor_path"],
            "positive_path": row["positive_path"],
            "negative_path": neg_path,
            "bbox_name": str(row.get("bbox_name", "")),
            "area_name": str(row.get("area_name", "")),
            "sample_id": str(row.get("bbox_name", f"id_{idx}")),
        }


class ValImageDataset(Dataset):
    def __init__(self, csv_path: str, cfg: TrainConfig, fraction: float = 1.0):
        df = pd.read_csv(csv_path)
        if fraction < 1.0:
            df = df.sample(frac=fraction, random_state=cfg.seed).reset_index(drop=True)

        self.cfg = cfg

        if cfg.filter_invalid_at_init:
            keep_rows = []
            for _, row in df.iterrows():
                try:
                    _, mask = read_tif_with_mask(row["file_path"], cfg)
                    if is_valid_sample(mask, cfg):
                        keep_rows.append(row)
                except Exception:
                    continue
            df = pd.DataFrame(keep_rows)

        self.df = df.reset_index(drop=True)
        print(f"Filtered validation images: kept {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img, mask = read_tif_with_mask(row["file_path"], self.cfg)
        ts = parse_capella_timestamp_from_name(row["file_path"])
        ts_str = ts.isoformat() if ts is not None else ""

        return {
            "img": torch.from_numpy(img),
            "mask": torch.from_numpy(mask),
            "image_id": str(row.get("image_id", idx)),
            "file_path": row["file_path"],
            "split": str(row.get("split", "val")),
            "area_name": str(row.get("area_name", "")),
            "bbox_name": str(row.get("bbox_name", "")),
            "filename": str(row.get("filename", Path(row["file_path"]).name)),
            "timestamp": ts_str,
        }


class ConvNeXtMetricNet(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()

        self.encoder = timm.create_model(
            cfg.backbone_name,
            pretrained=cfg.pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.channels = self.encoder.feature_info.channels()

        if cfg.freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.embed_head = nn.Sequential(
            nn.Linear(self.channels[-1], 512),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, cfg.embed_dim),
        )

        self.pair_head = nn.Sequential(
            nn.Linear(self.channels[-1] * 3, 256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, 1),
        )

    def _prepare_input(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def extract_feats(self, x):
        x = self._prepare_input(x)
        return self.encoder(x)

    def embed(self, x):
        feats = self.extract_feats(x)
        last = feats[-1]
        pooled = F.adaptive_avg_pool2d(last, 1).flatten(1)
        emb = self.embed_head(pooled)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    def forward_pair(self, x1, x2):
        feats1 = self.extract_feats(x1)
        feats2 = self.extract_feats(x2)

        last1 = feats1[-1]
        last2 = feats2[-1]

        p1 = F.adaptive_avg_pool2d(last1, 1).flatten(1)
        p2 = F.adaptive_avg_pool2d(last2, 1).flatten(1)

        emb1 = F.normalize(self.embed_head(p1), p=2, dim=1)
        emb2 = F.normalize(self.embed_head(p2), p=2, dim=1)

        diff = torch.abs(p1 - p2)
        pair_feat = torch.cat([p1, p2, diff], dim=1)
        pair_logit = self.pair_head(pair_feat)

        return {
            "emb1": emb1,
            "emb2": emb2,
            "pair_logit": pair_logit,
        }


def standard_triplet_loss(za, zp, zn, margin):
    d_ap = F.pairwise_distance(za, zp)
    d_an = F.pairwise_distance(za, zn)
    loss = torch.clamp(d_ap - d_an + margin, min=0.0).mean()
    acc = (d_ap < d_an).float().mean()
    return loss, d_ap.detach(), d_an.detach(), acc.detach()


def batch_hard_triplet_loss(za, zp, sample_ids: List[str], margin):
    device = za.device
    B = za.shape[0]

    d_ap = F.pairwise_distance(za, zp)
    d_anchor_to_pos = torch.cdist(za, zp, p=2)
    d_anchor_to_anchor = torch.cdist(za, za, p=2)

    same = torch.zeros((B, B), dtype=torch.bool, device=device)
    for i in range(B):
        for j in range(B):
            if sample_ids[i] == sample_ids[j]:
                same[i, j] = True

    d_ap_neg = d_anchor_to_pos.masked_fill(same, float("inf"))
    hard_neg_pos = d_ap_neg.min(dim=1).values

    eye = torch.eye(B, dtype=torch.bool, device=device)
    mask_anc = same | eye
    d_aa_neg = d_anchor_to_anchor.masked_fill(mask_anc, float("inf"))
    hard_neg_anchor = d_aa_neg.min(dim=1).values

    d_an = torch.minimum(hard_neg_pos, hard_neg_anchor)
    finite = torch.isfinite(d_an)

    if finite.sum() == 0:
        zero = torch.tensor(0.0, device=device)
        return zero, d_ap.detach(), torch.zeros_like(d_ap), zero

    loss = torch.clamp(d_ap[finite] - d_an[finite] + margin, min=0.0).mean()
    acc = (d_ap[finite] < d_an[finite]).float().mean()
    return loss, d_ap.detach(), d_an.detach(), acc.detach()


def pair_bce_loss(pair_logit, label):
    return F.binary_cross_entropy_with_logits(pair_logit.view(-1), label.view(-1).float())


def build_datasets(cfg: TrainConfig):
    print("Building datasets...")
    train_ds = PairMiningDataset(cfg.train_triplets_csv, cfg, fraction=cfg.train_fraction, training=True)
    val_ds = PairMiningDataset(cfg.val_triplets_csv, cfg, fraction=cfg.val_fraction, training=False)
    val_img_ds = ValImageDataset(cfg.val_images_csv, cfg, fraction=cfg.val_image_fraction)
    print(f"Train subset size: {len(train_ds)}")
    print(f"Validation subset size: {len(val_ds)}")
    print(f"Validation image count: {len(val_img_ds)}")
    return train_ds, val_ds, val_img_ds


def build_dataloaders(train_ds, val_ds, val_img_ds, cfg: TrainConfig):
    print("Building dataloaders...")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_img_loader = DataLoader(
        val_img_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, val_img_loader


def build_model(cfg: TrainConfig):
    print("Building model...")
    model = ConvNeXtMetricNet(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    return model, optimizer, scheduler


def run_epoch(model, loader, cfg: TrainConfig, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and cfg.device.startswith("cuda")))

    total_loss = 0.0
    total_count = 0

    logs = {
        "loss_total": [],
        "loss_batch_hard": [],
        "loss_triplet_std": [],
        "loss_pair": [],
        "acc_std_triplet": [],
        "acc_batch_hard": [],
        "d_ap_std": [],
        "d_an_std": [],
    }

    all_val_meta = []

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        a = batch["anchor"].to(cfg.device, non_blocking=True)
        p = batch["positive"].to(cfg.device, non_blocking=True)
        n = batch["negative"].to(cfg.device, non_blocking=True)
        sample_ids = list(batch["sample_id"])

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=(cfg.use_amp and cfg.device.startswith("cuda"))):
                za = model.embed(a)
                zp = model.embed(p)
                zn = model.embed(n)

                loss_std, d_ap_std, d_an_std, acc_std = standard_triplet_loss(
                    za, zp, zn, margin=cfg.triplet_margin
                )

                loss_bh, _, _, acc_bh = batch_hard_triplet_loss(
                    za, zp, sample_ids=sample_ids, margin=cfg.triplet_margin
                )

                out_pos = model.forward_pair(a, p)
                pair_logit_pos = out_pos["pair_logit"]
                pair_label_pos = torch.ones_like(pair_logit_pos.view(-1))

                out_neg = model.forward_pair(a, n)
                pair_logit_neg = out_neg["pair_logit"]
                pair_label_neg = torch.zeros_like(pair_logit_neg.view(-1))

                loss_pair = 0.5 * (
                    pair_bce_loss(pair_logit_pos, pair_label_pos) +
                    pair_bce_loss(pair_logit_neg, pair_label_neg)
                )

                loss = (
                    cfg.w_standard_triplet * loss_std +
                    cfg.w_batch_hard * loss_bh +
                    cfg.w_pair_bce * loss_pair
                ) / cfg.grad_accum_steps

            if is_train:
                scaler.scale(loss).backward()

                if ((step + 1) % cfg.grad_accum_steps == 0) or ((step + 1) == len(loader)):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        bs = a.size(0)
        total_loss += loss.item() * bs * cfg.grad_accum_steps
        total_count += bs

        logs["loss_total"].append(float(loss.item() * cfg.grad_accum_steps))
        logs["loss_batch_hard"].append(float(loss_bh.item()))
        logs["loss_triplet_std"].append(float(loss_std.item()))
        logs["loss_pair"].append(float(loss_pair.item()))
        logs["acc_std_triplet"].append(float(acc_std.item()))
        logs["acc_batch_hard"].append(float(acc_bh.item()))
        logs["d_ap_std"].extend(d_ap_std.detach().cpu().numpy().tolist())
        logs["d_an_std"].extend(d_an_std.detach().cpu().numpy().tolist())

        for i in range(bs):
            all_val_meta.append({
                "anchor_path": batch["anchor_path"][i],
                "positive_path": batch["positive_path"][i],
                "negative_path": batch["negative_path"][i],
                "bbox_name": batch["bbox_name"][i],
                "area_name": batch["area_name"][i],
                "sample_id": batch["sample_id"][i],
                "d_ap": float(d_ap_std[i].detach().cpu()),
                "d_an": float(d_an_std[i].detach().cpu()),
            })

    avg_loss = total_loss / max(total_count, 1)
    summary = {
        "loss": avg_loss,
        "loss_batch_hard": float(np.mean(logs["loss_batch_hard"])) if logs["loss_batch_hard"] else np.nan,
        "loss_triplet_std": float(np.mean(logs["loss_triplet_std"])) if logs["loss_triplet_std"] else np.nan,
        "loss_pair": float(np.mean(logs["loss_pair"])) if logs["loss_pair"] else np.nan,
        "acc_std_triplet": float(np.mean(logs["acc_std_triplet"])) if logs["acc_std_triplet"] else np.nan,
        "acc_batch_hard": float(np.mean(logs["acc_batch_hard"])) if logs["acc_batch_hard"] else np.nan,
        "d_ap": np.array(logs["d_ap_std"]),
        "d_an": np.array(logs["d_an_std"]),
        "meta": all_val_meta,
    }
    return summary


def train_model(model, train_loader, val_loader, optimizer, scheduler, cfg: TrainConfig):
    out_dir = Path(cfg.out_dir)
    safe_mkdir(out_dir)

    history = []
    best_val = float("inf")
    best_ckpt = out_dir / "best_model.pt"
    bad_epochs = 0

    print(f"Using device {cfg.device}")
    print("Starting training...")

    for epoch in range(cfg.num_epochs):
        train_sum = run_epoch(model, train_loader, cfg, optimizer=optimizer)
        val_sum = run_epoch(model, val_loader, cfg, optimizer=None)

        scheduler.step()

        row = {
            "epoch": epoch + 1,
            "train_loss": train_sum["loss"],
            "val_loss": val_sum["loss"],
            "train_loss_batch_hard": train_sum["loss_batch_hard"],
            "val_loss_batch_hard": val_sum["loss_batch_hard"],
            "train_loss_triplet_std": train_sum["loss_triplet_std"],
            "val_loss_triplet_std": val_sum["loss_triplet_std"],
            "train_loss_pair": train_sum["loss_pair"],
            "val_loss_pair": val_sum["loss_pair"],
            "train_acc_std_triplet": train_sum["acc_std_triplet"],
            "val_acc_std_triplet": val_sum["acc_std_triplet"],
            "train_acc_batch_hard": train_sum["acc_batch_hard"],
            "val_acc_batch_hard": val_sum["acc_batch_hard"],
            "train_mean_d_ap": float(np.mean(train_sum["d_ap"])) if len(train_sum["d_ap"]) > 0 else np.nan,
            "train_mean_d_an": float(np.mean(train_sum["d_an"])) if len(train_sum["d_an"]) > 0 else np.nan,
            "val_mean_d_ap": float(np.mean(val_sum["d_ap"])) if len(val_sum["d_ap"]) > 0 else np.nan,
            "val_mean_d_an": float(np.mean(val_sum["d_an"])) if len(val_sum["d_an"]) > 0 else np.nan,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)

        print(
            f"Epoch {epoch + 1}/{cfg.num_epochs} "
            f"train_loss={row['train_loss']:.5f} "
            f"val_loss={row['val_loss']:.5f} "
            f"train_bh_acc={row['train_acc_batch_hard']:.4f} "
            f"val_bh_acc={row['val_acc_batch_hard']:.4f} "
            f"val_d_ap={row['val_mean_d_ap']:.4f} "
            f"val_d_an={row['val_mean_d_an']:.4f} "
            f"lr={row['lr']:.2e}"
        )

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            bad_epochs = 0
            torch.save(model.state_dict(), best_ckpt)
            print("Saving new best checkpoint...")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Stopping early at epoch {epoch + 1}")
                break

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    return history, best_ckpt, best_val


@torch.no_grad()
def embed_val_images(model, loader, cfg: TrainConfig):
    model.eval()
    rows = []
    for batch in loader:
        img = batch["img"].to(cfg.device, non_blocking=True)
        emb = model.embed(img).detach().cpu().numpy()

        for i in range(len(emb)):
            rows.append({
                "image_id": batch["image_id"][i],
                "file_path": batch["file_path"][i],
                "split": batch["split"][i],
                "area_name": batch["area_name"][i],
                "bbox_name": batch["bbox_name"][i],
                "filename": batch["filename"][i],
                "timestamp": batch["timestamp"][i],
                "embedding": emb[i],
            })
    return pd.DataFrame(rows)


def plot_training_curves(history: List[Dict], cfg: TrainConfig):
    out_dir = Path(cfg.out_dir) / "curves"
    safe_mkdir(out_dir)
    df = pd.DataFrame(history)

    curves = [
        ("train_loss", "val_loss", "Total loss", "loss_total.png"),
        ("train_acc_std_triplet", "val_acc_std_triplet", "Standard triplet accuracy", "triplet_acc_std.png"),
        ("train_acc_batch_hard", "val_acc_batch_hard", "Batch-hard triplet accuracy", "triplet_acc_batch_hard.png"),
        ("train_loss_pair", "val_loss_pair", "Pair BCE loss", "pair_loss.png"),
    ]

    for c1, c2, title, fname in curves:
        if c1 not in df.columns or c2 not in df.columns:
            continue
        plt.figure(figsize=(7, 4))
        plt.plot(df["epoch"], df[c1], marker="o", linewidth=2, label="train")
        plt.plot(df["epoch"], df[c2], marker="o", linewidth=2, label="validation")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.grid(True, alpha=0.25)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=cfg.save_dpi)
        plt.close()


def plot_val_distance_hist(d_ap, d_an, cfg: TrainConfig):
    out_path = Path(cfg.out_dir) / "val_distance_hist.png"
    plt.figure(figsize=(8, 5))
    plt.hist(d_ap, bins=50, alpha=0.70, label="anchor-positive")
    plt.hist(d_an, bins=50, alpha=0.70, label="anchor-negative")
    plt.xlabel("Embedding distance")
    plt.ylabel("Count")
    plt.title("Validation embedding distances")
    plt.grid(True, alpha=0.20)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.save_dpi)
    plt.close()


def plot_val_gap_hist(d_ap, d_an, cfg: TrainConfig):
    out_path = Path(cfg.out_dir) / "val_gap_hist.png"
    gap = d_an - d_ap
    plt.figure(figsize=(8, 5))
    plt.hist(gap, bins=50, alpha=0.85)
    plt.xlabel("Distance gap = d(an) - d(ap)")
    plt.ylabel("Count")
    plt.title("Validation distance gap")
    plt.grid(True, alpha=0.20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.save_dpi)
    plt.close()


def save_summary_json(summary: Dict, cfg: TrainConfig):
    with open(Path(cfg.out_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run_full_training_pipeline(cfg: TrainConfig):
    set_seed(cfg.seed)
    safe_mkdir(Path(cfg.out_dir))

    train_ds, val_ds, val_img_ds = build_datasets(cfg)
    train_loader, val_loader, val_img_loader = build_dataloaders(train_ds, val_ds, val_img_ds, cfg)
    model, optimizer, scheduler = build_model(cfg)

    history, best_ckpt, best_val = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
    )

    plot_training_curves(history, cfg)

    print("Loading best checkpoint...")
    model.load_state_dict(torch.load(best_ckpt, map_location=cfg.device))

    print("Running final validation...")
    val_sum = run_epoch(model, val_loader, cfg, optimizer=None)
    plot_val_distance_hist(val_sum["d_ap"], val_sum["d_an"], cfg)
    plot_val_gap_hist(val_sum["d_ap"], val_sum["d_an"], cfg)

    print("Embedding validation images...")
    embed_df = embed_val_images(model, val_img_loader, cfg)
    embed_df.to_pickle(Path(cfg.out_dir) / "val_image_embeddings.pkl")
    embed_df.drop(columns=["embedding"]).to_csv(Path(cfg.out_dir) / "val_image_embedding_index.csv", index=False)

    summary = {
        "best_val_loss": float(best_val),
        "final_val_mean_d_ap": float(np.mean(val_sum["d_ap"])) if len(val_sum["d_ap"]) > 0 else None,
        "final_val_mean_d_an": float(np.mean(val_sum["d_an"])) if len(val_sum["d_an"]) > 0 else None,
        "final_val_triplet_acc": float(val_sum["acc_std_triplet"]),
        "final_val_batch_hard_acc": float(val_sum["acc_batch_hard"]),
        "device": cfg.device,
        "img_size": cfg.img_size,
        "backbone": cfg.backbone_name,
        "spatial_hard_negative": cfg.use_spatial_hard_negative,
        "config": asdict(cfg),
    }

    save_summary_json(summary, cfg)

    print(f"Finished training pipeline. Outputs saved to {cfg.out_dir}")

    return {
        "model": model,
        "history": history,
        "summary": summary,
        "embed_df": embed_df,
        "best_checkpoint": str(best_ckpt),
    }