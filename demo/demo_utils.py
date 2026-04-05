import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform as rio_transform
from rasterio.enums import Resampling

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


@dataclass
class DemoConfig:
    default_locations: Dict[str, Dict[str, float]]
    img_size: int = 256
    use_first_band_only: bool = True
    valid_mask_threshold: float = 0.5
    min_valid_ratio: float = 0.40
    min_center_valid_ratio: float = 0.55
    backbone_name: str = "convnext_base"
    pretrained: bool = False
    freeze_backbone: bool = False
    embed_dim: int = 256
    dropout: float = 0.20
    crop_size_m: int = 3000
    canvas_size: int = 3000
    patch_size: int = 256
    stride: int = 256
    default_top_k_patches: int = 12
    default_top_k_timelines: int = 12
    default_checkpoint_path: str = "convnext_change_pipeline/best_model.pt"
    default_download_dir: str = "downloaded_geo_tifs"
    page_title: str = "Capella Patch Timeline Demo"


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def build_tif_url(stac_id: str, start_datetime: str) -> str:
    dt = pd.to_datetime(start_datetime, utc=True)
    return (
        f"https://capella-open-data.s3.amazonaws.com/data/"
        f"{dt.year}/{dt.month}/{dt.day}/{stac_id}/{stac_id}.tif"
    )


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


def download_file(url: str, out_path: Path, timeout: int = 120, chunk_size: int = 1024 * 1024):
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def compute_valid_stats(mask: np.ndarray):
    m = mask[0]
    valid_ratio = float(m.mean())

    h, w = m.shape
    cy0, cy1 = int(h * 0.25), int(h * 0.75)
    cx0, cx1 = int(w * 0.25), int(w * 0.75)
    center_ratio = float(m[cy0:cy1, cx0:cx1].mean())

    return valid_ratio, center_ratio


def is_valid_sample(mask: np.ndarray, cfg: DemoConfig):
    valid_ratio, center_ratio = compute_valid_stats(mask)
    return (valid_ratio >= cfg.min_valid_ratio) and (center_ratio >= cfg.min_center_valid_ratio)


def robust_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return (x - med) / (mad + eps)


def safe_window_tuple(src, max_size: int = 256):
    return ((0, min(max_size, src.height)), (0, min(max_size, src.width)))


def validate_tif_readable(path: str) -> Tuple[bool, str]:
    try:
        with rasterio.open(path) as src:
            _ = src.read(1, window=safe_window_tuple(src, 256))
        return True, ""
    except Exception as e:
        return False, str(e)


def tif_center_latlon(path: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        with rasterio.open(path) as src:
            bounds = src.bounds
            cx = (bounds.left + bounds.right) / 2.0
            cy = (bounds.top + bounds.bottom) / 2.0
            xs, ys = rio_transform(src.crs, "EPSG:4326", [cx], [cy])
            lon = float(xs[0])
            lat = float(ys[0])
            return lat, lon
    except Exception:
        return None, None


def tif_start_datetime_from_tags(path: str) -> Optional[pd.Timestamp]:
    try:
        with rasterio.open(path) as src:
            tags = src.tags()
            desc = tags.get("TIFFTAG_IMAGEDESCRIPTION", "")
            if desc:
                try:
                    import json
                    meta = json.loads(desc)
                    ts = meta.get("collect", {}).get("start_timestamp", None)
                    if ts is not None:
                        return pd.to_datetime(ts, utc=True, errors="coerce")
                except Exception:
                    pass

            dt = tags.get("TIFFTAG_DATETIME", None)
            if dt is not None:
                return pd.to_datetime(dt, utc=True, errors="coerce")
    except Exception:
        return None

    return None


def build_downloaded_df_from_local_folder(folder_path: str) -> pd.DataFrame:
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    if not folder.is_dir():
        raise ValueError(f"Path is not a folder: {folder_path}")

    tif_files = sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff")))

    rows = []
    for p in tif_files:
        stac_id = p.stem
        is_readable, read_error = validate_tif_readable(str(p))
        lat, lon = tif_center_latlon(str(p))
        start_datetime = tif_start_datetime_from_tags(str(p))

        rows.append({
            "stac_id": stac_id,
            "start_datetime": start_datetime,
            "center_lat": lat,
            "center_lon": lon,
            "distance_km": np.nan,
            "local_tif_path": str(p),
            "download_status": "local",
            "download_error": "",
            "is_readable": is_readable,
            "read_error": read_error,
            "tif_url": "",
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return pd.DataFrame(columns=[
            "stac_id",
            "start_datetime",
            "center_lat",
            "center_lon",
            "distance_km",
            "local_tif_path",
            "download_status",
            "download_error",
            "is_readable",
            "read_error",
            "tif_url",
        ])

    df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True, errors="coerce")
    df = df.sort_values(["start_datetime", "stac_id"], na_position="last").reset_index(drop=True)
    return df


def add_distance_to_target(downloaded_df: pd.DataFrame, target_lat: float, target_lon: float) -> pd.DataFrame:
    df = downloaded_df.copy()

    if "center_lat" not in df.columns or "center_lon" not in df.columns:
        df["distance_km"] = np.nan
        return df

    def _dist(r):
        if pd.isna(r["center_lat"]) or pd.isna(r["center_lon"]):
            return np.nan
        return haversine_km(target_lat, target_lon, float(r["center_lat"]), float(r["center_lon"]))

    df["distance_km"] = df.apply(_dist, axis=1)
    return df


def parse_bbox_from_text(text: str) -> Optional[Tuple[float, float, float, float]]:
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None

    m = re.search(r"bbox_([\-0-9\.]+)_([\-0-9\.]+)_([\-0-9\.]+)_([\-0-9\.]+)", text)
    if m:
        return (
            float(m.group(1)),
            float(m.group(2)),
            float(m.group(3)),
            float(m.group(4)),
        )

    parts = [p.strip() for p in re.split(r"[,\s]+", text) if p.strip()]
    if len(parts) == 4:
        try:
            return tuple(float(p) for p in parts)
        except Exception:
            return None

    return None


def filter_geo_scenes(
    df: pd.DataFrame,
    target_lat: float,
    target_lon: float,
    delta_lat: float,
    delta_lon: float,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    required_cols = ["product_type", "center_lat", "center_lon", "stac_id", "start_datetime"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in catalog: {missing}")

    geo = df[df["product_type"].astype(str).str.upper() == "GEO"].copy()

    min_lat = target_lat - delta_lat
    max_lat = target_lat + delta_lat
    min_lon = target_lon - delta_lon
    max_lon = target_lon + delta_lon

    filtered = geo[
        (geo["center_lat"] >= min_lat)
        & (geo["center_lat"] <= max_lat)
        & (geo["center_lon"] >= min_lon)
        & (geo["center_lon"] <= max_lon)
    ].copy()

    if len(filtered) == 0:
        return filtered

    filtered["distance_km"] = filtered.apply(
        lambda r: haversine_km(
            target_lat,
            target_lon,
            float(r["center_lat"]),
            float(r["center_lon"]),
        ),
        axis=1,
    )

    filtered = filtered.sort_values(["distance_km", "start_datetime"]).reset_index(drop=True)

    if max_rows is not None:
        filtered = filtered.head(max_rows).copy()

    filtered["tif_url"] = filtered.apply(
        lambda r: build_tif_url(str(r["stac_id"]), str(r["start_datetime"])),
        axis=1,
    )

    preferred_cols = [
        "stac_id",
        "collect_id",
        "start_datetime",
        "end_datetime",
        "center_lon",
        "center_lat",
        "distance_km",
        "platform",
        "constellation",
        "instrument_mode",
        "frequency_band",
        "center_frequency",
        "polarizations",
        "orbit_state",
        "product_type",
        "observation_direction",
        "incidence_angle",
        "azimuth",
        "squint_angle",
        "layover_angle",
        "look_angle",
        "resolution_range",
        "resolution_azimuth",
        "resolution_ground_range",
        "pixel_spacing_range",
        "pixel_spacing_azimuth",
        "image_length",
        "image_width",
        "looks_range",
        "looks_azimuth",
        "orbital_plane",
        "collection_type",
        "tif_url",
    ]
    final_cols = [c for c in preferred_cols if c in filtered.columns] + [
        c for c in filtered.columns if c not in preferred_cols
    ]
    return filtered[final_cols]


class ConvNeXtMetricNet(nn.Module):
    def __init__(self, cfg: DemoConfig):
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


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, device: str, cfg: DemoConfig):
    model = ConvNeXtMetricNet(cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model, {"missing": missing, "unexpected": unexpected}


@torch.no_grad()
def extract_multi_scale_feature_maps(model, x: torch.Tensor) -> List[torch.Tensor]:
    model.eval()
    feats = model.extract_feats(x)
    return [feats[1], feats[2], feats[3]]


def normalize_map(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    mn, mx = float(np.min(m)), float(np.max(m))
    if mx <= mn:
        return np.zeros_like(m, dtype=np.float32)
    return (m - mn) / (mx - mn + 1e-8)


def upscale_map(m: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    return cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)


def smooth_map(m: np.ndarray, ksize: int = 7) -> np.ndarray:
    if ksize <= 1:
        return m.astype(np.float32)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(m, (ksize, ksize), 0).astype(np.float32)


@torch.no_grad()
def compute_feature_difference_map_from_patches(
    model,
    patch1: np.ndarray,
    patch2: np.ndarray,
    device: str,
    cfg: DemoConfig,
    normalize_per_channel: bool = False,
    feature_weights: Optional[List[float]] = None,
) -> np.ndarray:
    x1 = torch.from_numpy(patch1).unsqueeze(0).to(device)
    x2 = torch.from_numpy(patch2).unsqueeze(0).to(device)

    feats1 = extract_multi_scale_feature_maps(model, x1)
    feats2 = extract_multi_scale_feature_maps(model, x2)

    if feature_weights is None:
        feature_weights = [0.25, 0.35, 0.40]

    fused = np.zeros((cfg.patch_size, cfg.patch_size), dtype=np.float32)

    for w, f1, f2 in zip(feature_weights, feats1, feats2):
        if normalize_per_channel:
            f1 = F.normalize(f1, p=2, dim=1)
            f2 = F.normalize(f2, p=2, dim=1)

        diff = torch.norm(f1 - f2, p=2, dim=1)
        diff_np = diff[0].detach().cpu().numpy().astype(np.float32)

        diff_np = normalize_map(diff_np)
        diff_np = upscale_map(diff_np, cfg.patch_size, cfg.patch_size)
        diff_np = normalize_map(diff_np)

        fused += float(w) * diff_np

    return normalize_map(fused)


def threshold_feature_map(
    m: np.ndarray,
    method: str = "percentile",
    percentile: float = 90.0,
    min_abs: float = 0.18,
) -> np.ndarray:
    m = np.clip(m, 0.0, 1.0).astype(np.float32)

    if method == "otsu":
        img_u8 = (m * 255.0).astype(np.uint8)
        _, mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    thr = max(np.percentile(m, percentile), min_abs)
    return ((m >= thr).astype(np.uint8) * 255)


def clean_binary_mask(
    mask: np.ndarray,
    open_ksize: int = 3,
    close_ksize: int = 7,
    min_area: int = 80,
) -> np.ndarray:
    out = mask.copy()

    if open_ksize > 1:
        if open_ksize % 2 == 0:
            open_ksize += 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)

    if close_ksize > 1:
        if close_ksize % 2 == 0:
            close_ksize += 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    clean = np.zeros_like(out)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 255

    return clean


def mask_to_topk_bboxes(
    mask: np.ndarray,
    top_k: int = 3,
    min_area: int = 80,
) -> List[Tuple[int, int, int, int, int]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        boxes.append((x, y, x + w - 1, y + h - 1, area))

    boxes = sorted(boxes, key=lambda z: z[4], reverse=True)
    return boxes[:top_k]


def draw_topk_bboxes_on_gray_image(
    img: np.ndarray,
    bboxes: List[Tuple[int, int, int, int, int]],
) -> np.ndarray:
    base = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

    for k, (x0, y0, x1, y1, area) in enumerate(bboxes):
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{k+1}",
            (x0, max(12, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return vis


def _target_xy_in_scene_crs(src, target_lat: float, target_lon: float) -> Tuple[float, float]:
    xs, ys = rio_transform("EPSG:4326", src.crs, [target_lon], [target_lat])
    return float(xs[0]), float(ys[0])


def _crop_bounds_in_scene_units(src, target_lat: float, target_lon: float, crop_size_m: float):
    x, y = _target_xy_in_scene_crs(src, target_lat, target_lon)

    if src.crs.is_geographic:
        delta_lat = crop_size_m / 2.0 / 111320.0
        delta_lon = crop_size_m / 2.0 / max(111320.0 * math.cos(math.radians(target_lat)), 1e-6)
        xs, ys = rio_transform(
            "EPSG:4326",
            src.crs,
            [target_lon - delta_lon, target_lon + delta_lon],
            [target_lat - delta_lat, target_lat + delta_lat],
        )
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
    else:
        half = crop_size_m / 2.0
        xmin, xmax = x - half, x + half
        ymin, ymax = y - half, y + half

    return xmin, ymin, xmax, ymax


def _read_window_with_fallback(
    src,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    cfg: DemoConfig,
):
    window = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)

    try:
        if cfg.use_first_band_only:
            arr = src.read(
                1,
                window=window,
                out_shape=(cfg.canvas_size, cfg.canvas_size),
                boundless=True,
                fill_value=0,
            )
            img = normalize_img(arr)[None, :, :]
        else:
            arr = src.read(
                window=window,
                out_shape=(src.count, cfg.canvas_size, cfg.canvas_size),
                boundless=True,
                fill_value=0,
            )
            img = np.stack([normalize_img(b) for b in arr], axis=0)

        mask = src.read_masks(
            1,
            window=window,
            out_shape=(cfg.canvas_size, cfg.canvas_size),
            boundless=True,
        ).astype(np.float32) / 255.0

    except Exception:
        try:
            if cfg.use_first_band_only:
                arr = src.read(
                    1,
                    window=window,
                    out_shape=(cfg.canvas_size, cfg.canvas_size),
                    boundless=True,
                    fill_value=0,
                    resampling=Resampling.bilinear,
                )
                img = normalize_img(arr)[None, :, :]
            else:
                arr = src.read(
                    window=window,
                    out_shape=(src.count, cfg.canvas_size, cfg.canvas_size),
                    boundless=True,
                    fill_value=0,
                    resampling=Resampling.bilinear,
                )
                img = np.stack([normalize_img(b) for b in arr], axis=0)

            try:
                mask = src.read_masks(
                    1,
                    window=window,
                    out_shape=(cfg.canvas_size, cfg.canvas_size),
                    boundless=True,
                ).astype(np.float32) / 255.0
            except Exception:
                mask = np.ones((cfg.canvas_size, cfg.canvas_size), dtype=np.float32)

        except Exception:
            try:
                full = src.read(1)
                full_mask = src.read_masks(1).astype(np.float32) / 255.0

                row_min, col_min = src.index(xmin, ymax)
                row_max, col_max = src.index(xmax, ymin)

                r0 = max(0, min(row_min, row_max))
                r1 = min(src.height, max(row_min, row_max))
                c0 = max(0, min(col_min, col_max))
                c1 = min(src.width, max(col_min, col_max))

                if r1 <= r0 or c1 <= c0:
                    arr = np.zeros((cfg.canvas_size, cfg.canvas_size), dtype=np.float32)
                    mask = np.zeros((cfg.canvas_size, cfg.canvas_size), dtype=np.float32)
                else:
                    crop = full[r0:r1, c0:c1]
                    crop_mask = full_mask[r0:r1, c0:c1]

                    arr = cv2.resize(crop, (cfg.canvas_size, cfg.canvas_size), interpolation=cv2.INTER_LINEAR)
                    mask = cv2.resize(crop_mask, (cfg.canvas_size, cfg.canvas_size), interpolation=cv2.INTER_NEAREST)

                img = normalize_img(arr)[None, :, :]
            except Exception as e2:
                raise RuntimeError(f"Could not read crop from TIFF: {e2}")

    mask = (mask >= cfg.valid_mask_threshold).astype(np.float32)[None, :, :]
    return img.astype(np.float32), mask.astype(np.float32)


def extract_center_crop_with_mask(
    path: str,
    target_lat: float,
    target_lon: float,
    cfg: DemoConfig,
):
    with rasterio.open(path) as src:
        xmin, ymin, xmax, ymax = _crop_bounds_in_scene_units(
            src=src,
            target_lat=target_lat,
            target_lon=target_lon,
            crop_size_m=cfg.crop_size_m,
        )
        return _read_window_with_fallback(
            src=src,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            cfg=cfg,
        )


def extract_bbox_crop_with_mask(
    path: str,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    cfg: DemoConfig,
):
    with rasterio.open(path) as src:
        return _read_window_with_fallback(
            src=src,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            cfg=cfg,
        )


def split_canvas_into_patches(
    img: np.ndarray,
    mask: np.ndarray,
    cfg: DemoConfig,
) -> List[Dict]:
    patches = []
    _, h, w = img.shape

    for y0 in range(0, h - cfg.patch_size + 1, cfg.stride):
        for x0 in range(0, w - cfg.patch_size + 1, cfg.stride):
            px = img[:, y0:y0 + cfg.patch_size, x0:x0 + cfg.patch_size]
            pm = mask[:, y0:y0 + cfg.patch_size, x0:x0 + cfg.patch_size]

            if px.shape[-2:] != (cfg.patch_size, cfg.patch_size):
                continue
            if pm.shape[-2:] != (cfg.patch_size, cfg.patch_size):
                continue

            valid_ratio, center_ratio = compute_valid_stats(pm)
            ok = is_valid_sample(pm, cfg)

            patches.append({
                "patch_id": f"r{y0}_c{x0}",
                "patch_row": y0,
                "patch_col": x0,
                "img": px.astype(np.float32),
                "mask": pm.astype(np.float32),
                "valid_ratio": valid_ratio,
                "center_valid_ratio": center_ratio,
                "is_valid": ok,
            })

    return patches


@torch.no_grad()
def embed_scene_patches(
    model,
    tif_path: str,
    target_lat: float,
    target_lon: float,
    device: str,
    cfg: DemoConfig,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> List[Dict]:
    try:
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            img, mask = extract_bbox_crop_with_mask(
                path=tif_path,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                cfg=cfg,
            )
        else:
            img, mask = extract_center_crop_with_mask(
                path=tif_path,
                target_lat=target_lat,
                target_lon=target_lon,
                cfg=cfg,
            )
    except Exception:
        return []

    patches = split_canvas_into_patches(img=img, mask=mask, cfg=cfg)

    rows = []
    for p in patches:
        if not p["is_valid"]:
            continue

        try:
            x = torch.from_numpy(p["img"]).unsqueeze(0).to(device)
            emb = model.embed(x).squeeze(0).detach().cpu().numpy()
        except Exception:
            continue

        rows.append({
            "patch_id": p["patch_id"],
            "patch_row": p["patch_row"],
            "patch_col": p["patch_col"],
            "valid_ratio": p["valid_ratio"],
            "center_valid_ratio": p["center_valid_ratio"],
            "embedding": emb,
        })

    return rows


def run_patch_embedding_analysis(
    model,
    downloaded_df: pd.DataFrame,
    target_lat: float,
    target_lon: float,
    device: str,
    cfg: DemoConfig,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> pd.DataFrame:
    rows = []

    ordered = downloaded_df.copy()
    ordered["start_datetime"] = pd.to_datetime(ordered["start_datetime"], utc=True, errors="coerce")
    ordered = ordered.sort_values("start_datetime").reset_index(drop=True)

    progress = st.progress(0.0)
    info = st.empty()

    total = len(ordered)
    for i, (_, row) in enumerate(ordered.iterrows(), start=1):
        tif_path = str(row["local_tif_path"])
        if not Path(tif_path).exists():
            continue

        info.info(f"Embedding patches: {i}/{total} | {row['stac_id']}")

        try:
            scene_patch_rows = embed_scene_patches(
                model=model,
                tif_path=tif_path,
                target_lat=target_lat,
                target_lon=target_lon,
                device=device,
                cfg=cfg,
                bbox=bbox,
            )

            if len(scene_patch_rows) == 0:
                rows.append({
                    "stac_id": row["stac_id"],
                    "start_datetime": row["start_datetime"],
                    "local_tif_path": tif_path,
                    "distance_km": row.get("distance_km", np.nan),
                    "patch_id": None,
                    "patch_row": None,
                    "patch_col": None,
                    "valid_ratio": np.nan,
                    "center_valid_ratio": np.nan,
                    "embedding": None,
                    "error": "No valid patches extracted from scene.",
                })
            else:
                for pr in scene_patch_rows:
                    rows.append({
                        "stac_id": row["stac_id"],
                        "start_datetime": row["start_datetime"],
                        "local_tif_path": tif_path,
                        "distance_km": row.get("distance_km", np.nan),
                        **pr,
                        "error": "",
                    })

        except Exception as e:
            rows.append({
                "stac_id": row["stac_id"],
                "start_datetime": row["start_datetime"],
                "local_tif_path": tif_path,
                "distance_km": row.get("distance_km", np.nan),
                "patch_id": None,
                "patch_row": None,
                "patch_col": None,
                "valid_ratio": np.nan,
                "center_valid_ratio": np.nan,
                "embedding": None,
                "error": str(e),
            })

        progress.progress(i / max(total, 1))

    progress.empty()
    info.empty()

    expected_cols = [
        "stac_id",
        "start_datetime",
        "local_tif_path",
        "distance_km",
        "patch_id",
        "patch_row",
        "patch_col",
        "valid_ratio",
        "center_valid_ratio",
        "embedding",
        "error",
    ]

    out = pd.DataFrame(rows)

    if len(out) == 0:
        return pd.DataFrame(columns=expected_cols)

    for c in expected_cols:
        if c not in out.columns:
            out[c] = None

    out = out[out["embedding"].notnull()].copy().reset_index(drop=True)

    if len(out) == 0:
        return pd.DataFrame(columns=expected_cols)

    out["start_datetime"] = pd.to_datetime(out["start_datetime"], utc=True, errors="coerce")
    out = out.sort_values(["patch_id", "start_datetime"]).reset_index(drop=True)
    return out


def build_patch_timeline(patch_embed_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    g = patch_embed_df.copy()
    g["start_datetime"] = pd.to_datetime(g["start_datetime"], utc=True, errors="coerce")
    g = g.dropna(subset=["start_datetime"]).sort_values("start_datetime").reset_index(drop=True)

    if len(g) < 3:
        return None

    embs = np.stack(g["embedding"].to_list(), axis=0).astype(np.float32)
    ref0 = embs[0]

    d_ref0 = np.linalg.norm(embs - ref0[None, :], axis=1).astype(np.float32)

    step = np.zeros(len(g), dtype=np.float32)
    step[1:] = np.linalg.norm(embs[1:] - embs[:-1], axis=1).astype(np.float32)

    cumulative = np.cumsum(step).astype(np.float32)

    step_z = np.zeros_like(step)
    if len(step) > 2:
        step_z[1:] = robust_zscore(step[1:])

    ref_gain = np.zeros(len(g), dtype=np.float32)
    ref_gain[1:] = d_ref0[1:] - d_ref0[:-1]

    trend = np.zeros(len(g), dtype=np.float32)
    for i in range(1, len(g)):
        a = max(1, i - 2)
        trend[i] = float(np.mean(step[a:i + 1]))

    change_score = np.maximum(step, np.maximum(trend, d_ref0)).astype(np.float32)

    g["dist_to_ref0"] = d_ref0
    g["step_change"] = step
    g["cumulative_step_change"] = cumulative
    g["step_z"] = step_z
    g["ref_gain"] = ref_gain
    g["trend_step"] = trend
    g["change_score"] = change_score
    g["date_str"] = g["start_datetime"].dt.strftime("%Y-%m-%d")
    return g


def classify_timeline_event_type(
    tl: pd.DataFrame,
    temp_min_z: float = 2.5,
    temp_after_mean_max_frac: float = 0.65,
    continuous_min_run: int = 3,
    continuous_min_mean_z: float = 0.8,
    continuous_min_ref_gain: float = 0.10,
):
    if tl is None or len(tl) < 4:
        return {
            "event_type": "other",
            "score": 0.0,
            "event_idx": None,
            "start_idx": None,
            "end_idx": None,
        }

    step = tl["step_change"].values.astype(np.float32)
    step_z = tl["step_z"].values.astype(np.float32)
    d_ref0 = tl["dist_to_ref0"].values.astype(np.float32)

    best_temp_score = 0.0
    best_temp_idx = None

    for i in range(2, len(tl) - 1):
        cur_z = float(step_z[i])
        cur_step = float(step[i])

        if cur_z < temp_min_z:
            continue

        after = step[i + 1:min(len(tl), i + 3)]
        after_mean = float(np.mean(after)) if len(after) > 0 else 0.0
        cond_relax = after_mean < cur_step * temp_after_mean_max_frac

        future_ref = d_ref0[i:min(len(tl), i + 3)]
        future_ref_gain = future_ref[-1] - future_ref[0] if len(future_ref) >= 2 else 0.0
        cond_no_strong_build = future_ref_gain < max(0.15, 0.6 * cur_step)

        prev_step = float(step[i - 1])
        next_step = float(step[i + 1]) if i + 1 < len(step) else 0.0
        cond_local_peak = cur_step >= prev_step and cur_step >= next_step

        if cond_relax and cond_no_strong_build and cond_local_peak:
            score = cur_z + max(0.0, cur_step - after_mean)
            if score > best_temp_score:
                best_temp_score = score
                best_temp_idx = i

    best_cont_score = 0.0
    best_cont_range = None

    for start in range(2, len(tl) - continuous_min_run + 1):
        for end in range(start + continuous_min_run - 1, len(tl)):
            seg_z = step_z[start:end + 1]
            seg_ref = d_ref0[start:end + 1]

            mean_z = float(np.mean(seg_z))
            pos_ratio = float((seg_z > 0.0).mean())
            total_ref_gain = float(seg_ref[-1] - seg_ref[0])

            if (
                mean_z >= continuous_min_mean_z
                and pos_ratio >= 0.66
                and total_ref_gain >= continuous_min_ref_gain
            ):
                score = mean_z + 0.25 * (end - start + 1) + total_ref_gain
                if score > best_cont_score:
                    best_cont_score = score
                    best_cont_range = (start, end)

    if best_temp_score > 0 and best_temp_score >= best_cont_score:
        return {
            "event_type": "temporary",
            "score": best_temp_score,
            "event_idx": best_temp_idx,
            "start_idx": None,
            "end_idx": None,
        }

    if best_cont_score > 0:
        return {
            "event_type": "continuous",
            "score": best_cont_score,
            "event_idx": None,
            "start_idx": best_cont_range[0],
            "end_idx": best_cont_range[1],
        }

    return {
        "event_type": "other",
        "score": 0.0,
        "event_idx": None,
        "start_idx": None,
        "end_idx": None,
    }


def build_all_patch_timeline_summaries(embed_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    if embed_df is None or len(embed_df) == 0 or "patch_id" not in embed_df.columns:
        return pd.DataFrame(), {}

    summaries = []
    timelines = {}

    for patch_id, g in embed_df.groupby("patch_id"):
        tl = build_patch_timeline(g)
        if tl is None:
            continue

        timelines[patch_id] = tl

        best_step_idx = int(np.argmax(tl["step_change"].values))
        best_ref_idx = int(np.argmax(tl["dist_to_ref0"].values))

        evt = classify_timeline_event_type(tl)

        interesting_score = (
            0.70 * float(tl["step_change"].max()) +
            0.30 * float(tl["dist_to_ref0"].max()) +
            0.02 * len(tl)
        )

        summaries.append({
            "patch_id": patch_id,
            "patch_row": int(tl["patch_row"].iloc[0]),
            "patch_col": int(tl["patch_col"].iloc[0]),
            "num_dates": len(tl),
            "max_dist_to_prev": float(tl["step_change"].max()),
            "mean_dist_to_prev": float(tl["step_change"].mean()),
            "max_dist_to_ref0": float(tl["dist_to_ref0"].max()),
            "interesting_score": interesting_score,
            "event_type": evt["event_type"],
            "event_score": float(evt["score"]),
            "event_idx": evt["event_idx"],
            "event_start_idx": evt["start_idx"],
            "event_end_idx": evt["end_idx"],
            "best_change_before_scene": tl.iloc[best_step_idx - 1]["stac_id"] if best_step_idx > 0 else None,
            "best_change_before_date": tl.iloc[best_step_idx - 1]["date_str"] if best_step_idx > 0 else None,
            "best_change_after_scene": tl.iloc[best_step_idx]["stac_id"],
            "best_change_after_date": tl.iloc[best_step_idx]["date_str"],
            "best_ref_scene": tl.iloc[best_ref_idx]["stac_id"],
            "best_ref_date": tl.iloc[best_ref_idx]["date_str"],
        })

    summary_df = pd.DataFrame(summaries)
    if len(summary_df) > 0:
        summary_df = summary_df.sort_values(
            ["interesting_score", "max_dist_to_prev", "max_dist_to_ref0"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

    return summary_df, timelines


def extract_single_patch_preview(
    tif_path: str,
    target_lat: float,
    target_lon: float,
    patch_row: int,
    patch_col: int,
    cfg: DemoConfig,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[np.ndarray]:
    try:
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            img, _ = extract_bbox_crop_with_mask(
                path=tif_path,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                cfg=cfg,
            )
        else:
            img, _ = extract_center_crop_with_mask(
                path=tif_path,
                target_lat=target_lat,
                target_lon=target_lon,
                cfg=cfg,
            )

        patch = img[:, patch_row:patch_row + cfg.patch_size, patch_col:patch_col + cfg.patch_size]
        if patch.shape[-2:] != (cfg.patch_size, cfg.patch_size):
            return None
        return patch[0]
    except Exception:
        return None


def plot_patch_pair(
    before_path: str,
    after_path: str,
    target_lat: float,
    target_lon: float,
    patch_row: int,
    patch_col: int,
    title: str,
    cfg: DemoConfig,
    bbox: Optional[Tuple[float, float, float, float]] = None,
):
    before = extract_single_patch_preview(
        tif_path=before_path,
        target_lat=target_lat,
        target_lon=target_lon,
        patch_row=patch_row,
        patch_col=patch_col,
        cfg=cfg,
        bbox=bbox,
    )
    after = extract_single_patch_preview(
        tif_path=after_path,
        target_lat=target_lat,
        target_lon=target_lon,
        patch_row=patch_row,
        patch_col=patch_col,
        cfg=cfg,
        bbox=bbox,
    )

    if before is None or after is None:
        st.warning("One of the patches could not be read, skipped preview.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(before, cmap="gray")
    axes[0].set_title("Before")
    axes[0].axis("off")

    axes[1].imshow(after, cmap="gray")
    axes[1].set_title("After")
    axes[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_single_patch_timeline(tl: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(tl))

    ax.plot(x, tl["dist_to_ref0"].values, marker="o", linewidth=2.2, label="Distance to first date")
    ax.plot(x, tl["step_change"].values, marker="s", linewidth=1.8, label="Step change")

    ax.set_xticks(x)
    ax.set_xticklabels(tl["date_str"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Embedding distance")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_cumulative_event_timeline_figure(
    tl: pd.DataFrame,
    target_lat: float,
    target_lon: float,
    patch_row: int,
    patch_col: int,
    title: str,
    event_type: str,
    cfg: DemoConfig,
    event_idx: Optional[int] = None,
    event_start_idx: Optional[int] = None,
    event_end_idx: Optional[int] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
):
    n = len(tl)
    fig = plt.figure(figsize=(max(12, 1.2 * n), 8.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.8, 1.2])

    x = np.arange(n)

    ax1 = fig.add_subplot(gs[0, 0])
    l1 = ax1.plot(
        x,
        tl["cumulative_step_change"].values,
        marker="o",
        linewidth=2.2,
        label="Cumulative step change"
    )
    l2 = ax1.plot(
        x,
        tl["dist_to_ref0"].values,
        marker="D",
        linewidth=1.8,
        label="Distance to first date"
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(tl["date_str"].tolist(), rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Embedding distance")
    ax1.grid(True, alpha=0.25)
    ax1.set_title(title)

    if event_type == "temporary" and event_idx is not None:
        ax1.axvline(event_idx, linestyle="--", linewidth=2.0, alpha=0.9)
    elif event_type == "continuous" and event_start_idx is not None and event_end_idx is not None:
        ax1.axvspan(event_start_idx - 0.2, event_end_idx + 0.2, alpha=0.16)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    bars = ax2.bar(
        x,
        tl["step_change"].values,
        alpha=0.28,
        width=0.55,
        label="Step change"
    )
    ax2.plot(
        x,
        tl["trend_step"].values,
        linewidth=1.7,
        marker="s",
        label="Local trend"
    )
    ax2.set_ylabel("Step / trend")
    ax2.grid(True, alpha=0.20)

    if event_type == "temporary" and event_idx is not None:
        ax2.axvline(event_idx, linestyle="--", linewidth=2.0, alpha=0.9)
    elif event_type == "continuous" and event_start_idx is not None and event_end_idx is not None:
        ax2.axvspan(event_start_idx - 0.2, event_end_idx + 0.2, alpha=0.16)

    handles = l1 + l2 + [bars, ax2.lines[-1]]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper left", frameon=True)

    sub = gs[2, 0].subgridspec(1, n)

    for i, (_, row) in enumerate(tl.iterrows()):
        ax = fig.add_subplot(sub[0, i])

        preview = extract_single_patch_preview(
            tif_path=row["local_tif_path"],
            target_lat=target_lat,
            target_lon=target_lon,
            patch_row=patch_row,
            patch_col=patch_col,
            cfg=cfg,
            bbox=bbox,
        )

        if preview is None:
            ax.text(0.5, 0.5, "Unreadable", ha="center", va="center", fontsize=9)
            ax.set_title(f"{row['date_str']}", fontsize=8)
            ax.axis("off")
            continue

        ax.imshow(preview, cmap="gray")
        ax.set_title(
            f"{row['date_str']}\n"
            f"step={row['step_change']:.2f}\n"
            f"ref0={row['dist_to_ref0']:.2f}",
            fontsize=8
        )

        if event_type == "temporary" and event_idx is not None and i == event_idx:
            for s in ax.spines.values():
                s.set_linewidth(2.5)
                s.set_alpha(0.95)
        elif event_type == "continuous" and event_start_idx is not None and event_end_idx is not None and event_start_idx <= i <= event_end_idx:
            for s in ax.spines.values():
                s.set_linewidth(2.5)
                s.set_alpha(0.95)

        ax.axis("off")

    fig.tight_layout()
    return fig


def extract_single_patch_with_mask(
    tif_path: str,
    target_lat: float,
    target_lon: float,
    patch_row: int,
    patch_col: int,
    cfg: DemoConfig,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            img, mask = extract_bbox_crop_with_mask(
                path=tif_path,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                cfg=cfg,
            )
        else:
            img, mask = extract_center_crop_with_mask(
                path=tif_path,
                target_lat=target_lat,
                target_lon=target_lon,
                cfg=cfg,
            )

        patch = img[:, patch_row:patch_row + cfg.patch_size, patch_col:patch_col + cfg.patch_size]
        patch_mask = mask[:, patch_row:patch_row + cfg.patch_size, patch_col:patch_col + cfg.patch_size]

        if patch.shape[-2:] != (cfg.patch_size, cfg.patch_size):
            return None, None
        if patch_mask.shape[-2:] != (cfg.patch_size, cfg.patch_size):
            return None, None

        return patch.astype(np.float32), patch_mask.astype(np.float32)
    except Exception:
        return None, None


def localize_pair_from_feature_maps(
    model,
    before_path: str,
    after_path: str,
    target_lat: float,
    target_lon: float,
    patch_row: int,
    patch_col: int,
    device: str,
    cfg: DemoConfig,
    normalize_per_channel: bool = False,
    smooth_ksize: int = 7,
    threshold_method: str = "percentile",
    threshold_percentile: float = 90.0,
    threshold_min_abs: float = 0.18,
    open_ksize: int = 3,
    close_ksize: int = 7,
    min_area: int = 80,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Dict:
    before_patch, _ = extract_single_patch_with_mask(
        tif_path=before_path,
        target_lat=target_lat,
        target_lon=target_lon,
        patch_row=patch_row,
        patch_col=patch_col,
        cfg=cfg,
        bbox=bbox,
    )
    after_patch, _ = extract_single_patch_with_mask(
        tif_path=after_path,
        target_lat=target_lat,
        target_lon=target_lon,
        patch_row=patch_row,
        patch_col=patch_col,
        cfg=cfg,
        bbox=bbox,
    )

    if before_patch is None or after_patch is None:
        raise ValueError("Could not read one of the patches from the TIFF.")

    diff_up = compute_feature_difference_map_from_patches(
        model=model,
        patch1=before_patch,
        patch2=after_patch,
        device=device,
        cfg=cfg,
        normalize_per_channel=normalize_per_channel,
        feature_weights=[0.25, 0.35, 0.40],
    )

    diff_sm = smooth_map(diff_up, ksize=smooth_ksize)
    diff_sm = normalize_map(diff_sm)

    mask = threshold_feature_map(
        diff_sm,
        method=threshold_method,
        percentile=threshold_percentile,
        min_abs=threshold_min_abs,
    )
    mask = clean_binary_mask(
        mask,
        open_ksize=open_ksize,
        close_ksize=close_ksize,
        min_area=min_area,
    )

    top_bboxes = mask_to_topk_bboxes(mask, top_k=3, min_area=min_area)
    bbox_out = top_bboxes[0][:4] if len(top_bboxes) > 0 else None

    before_vis = draw_topk_bboxes_on_gray_image(before_patch[0], top_bboxes)
    after_vis = draw_topk_bboxes_on_gray_image(after_patch[0], top_bboxes)

    return {
        "before_patch": before_patch[0],
        "after_patch": after_patch[0],
        "diff_up": diff_up,
        "diff_sm": diff_sm,
        "mask": mask,
        "bbox": bbox_out,
        "top_bboxes": top_bboxes,
        "before_vis": before_vis,
        "after_vis": after_vis,
    }


def plot_feature_localization_result(result: Dict, title: str):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(result["before_patch"], cmap="gray")
    axes[0, 0].set_title("Before patch")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(result["after_patch"], cmap="gray")
    axes[0, 1].set_title("After patch")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(result["diff_up"], cmap="inferno")
    axes[0, 2].set_title("Multi-scale feature diff")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(result["diff_sm"], cmap="inferno")
    axes[1, 0].set_title("Smoothed diff")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(result["mask"], cmap="gray")
    axes[1, 1].set_title("Thresholded mask")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(result["after_vis"])
    axes[1, 2].set_title("Top localization boxes")
    axes[1, 2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def init_state():
    defaults = {
        "catalog_df": pd.DataFrame(),
        "filtered_df": pd.DataFrame(),
        "downloaded_df": pd.DataFrame(),
        "patch_embed_df": pd.DataFrame(),
        "patch_summary_df": pd.DataFrame(),
        "patch_timelines": {},
        "active_bbox": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_sidebar(cfg_base: DemoConfig):
    st.sidebar.title("Controls")

    input_mode = st.sidebar.radio(
        "Input mode",
        ["Catalog CSV + download", "Local TIFF folder"],
        index=0,
    )

    preset_name = st.sidebar.selectbox("Location", list(cfg_base.default_locations.keys()))
    preset = cfg_base.default_locations[preset_name]

    st.sidebar.markdown("Fixed analysis settings")
    st.sidebar.caption(f"Geo center crop: {cfg_base.crop_size_m} m")
    st.sidebar.caption(f"Resized canvas: {cfg_base.canvas_size}")
    st.sidebar.caption(f"Patch size: {cfg_base.patch_size}")
    st.sidebar.caption(f"Stride: {cfg_base.stride}")
    st.sidebar.caption(f"Backbone: {cfg_base.backbone_name}")
    st.sidebar.caption(f"Embedding dim: {cfg_base.embed_dim}")
    st.sidebar.caption(f"Use first band only: {cfg_base.use_first_band_only}")

    st.sidebar.markdown("Scene filtering")
    delta_lat = st.sidebar.slider("DELTA_LAT", 0.01, 0.30, 0.10, 0.01)
    delta_lon = st.sidebar.slider("DELTA_LON", 0.01, 0.30, 0.10, 0.01)
    max_rows = st.sidebar.slider("Max scenes", 2, 100, 10, 1)

    st.sidebar.markdown("Download")
    download_dir = st.sidebar.text_input("Download directory", value=cfg_base.default_download_dir)
    timeout = st.sidebar.slider("Timeout (seconds)", 30, 300, 120, 10)
    do_download = st.sidebar.toggle("Actually download TIFFs", value=True)

    st.sidebar.markdown("Local TIFF folder")
    local_tif_folder = st.sidebar.text_input("Local TIFF folder path", value=cfg_base.default_download_dir)

    st.sidebar.markdown("Optional bbox constraint")
    use_bbox = st.sidebar.toggle("Use bbox for analysis", value=False)
    bbox_text = st.sidebar.text_input(
        "BBox (xmin,ymin,xmax,ymax or folder-name text)",
        value="",
    )
    auto_bbox_from_folder = st.sidebar.toggle("Auto-parse bbox from local folder name", value=True)

    st.sidebar.markdown("Model")
    checkpoint_path = st.sidebar.text_input("Checkpoint path (.pt)", value=cfg_base.default_checkpoint_path)

    st.sidebar.markdown("Results")
    top_k_patches = st.sidebar.slider("Top interesting pairs", 1, 30, cfg_base.default_top_k_patches, 1)
    top_k_timelines = st.sidebar.slider("Top interesting timelines", 1, 30, cfg_base.default_top_k_timelines, 1)

    return {
        "input_mode": input_mode,
        "location_name": preset_name,
        "target_lat": float(preset["lat"]),
        "target_lon": float(preset["lon"]),
        "delta_lat": delta_lat,
        "delta_lon": delta_lon,
        "max_rows": max_rows,
        "download_dir": download_dir,
        "timeout": timeout,
        "do_download": do_download,
        "local_tif_folder": local_tif_folder,
        "use_bbox": use_bbox,
        "bbox_text": bbox_text,
        "auto_bbox_from_folder": auto_bbox_from_folder,
        "checkpoint_path": checkpoint_path,
        "top_k_patches": top_k_patches,
        "top_k_timelines": top_k_timelines,
    }


def resolve_active_bbox(ui_cfg) -> Optional[Tuple[float, float, float, float]]:
    if not ui_cfg["use_bbox"]:
        return None

    bbox = parse_bbox_from_text(ui_cfg["bbox_text"])
    if bbox is not None:
        return bbox

    if ui_cfg["auto_bbox_from_folder"]:
        folder_name = Path(ui_cfg["local_tif_folder"]).name
        bbox = parse_bbox_from_text(folder_name)
        if bbox is not None:
            return bbox

    return None


def run_filtering(ui_cfg):
    st.session_state.filtered_df = filter_geo_scenes(
        df=st.session_state.catalog_df,
        target_lat=ui_cfg["target_lat"],
        target_lon=ui_cfg["target_lon"],
        delta_lat=ui_cfg["delta_lat"],
        delta_lon=ui_cfg["delta_lon"],
        max_rows=ui_cfg["max_rows"],
    )


def run_download(ui_cfg):
    df = st.session_state.filtered_df.copy()
    if len(df) == 0:
        return

    out_dir = Path(ui_cfg["download_dir"])
    safe_mkdir(out_dir)

    progress = st.progress(0.0)
    msg = st.empty()

    rows = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        stac_id = str(row["stac_id"])
        tif_url = str(row["tif_url"])
        out_path = out_dir / f"{stac_id}.tif"

        status = "missing"
        error_text = ""
        is_readable = False
        read_error = ""

        try:
            if out_path.exists():
                status = "exists"
            else:
                if ui_cfg["do_download"]:
                    download_file(tif_url, out_path, timeout=ui_cfg["timeout"])
                    status = "downloaded"
                else:
                    status = "ready"

            if out_path.exists():
                is_readable, read_error = validate_tif_readable(str(out_path))

        except Exception as e:
            status = "error"
            error_text = str(e)

        rr = row.to_dict()
        rr["local_tif_path"] = str(out_path)
        rr["download_status"] = status
        rr["download_error"] = error_text
        rr["is_readable"] = is_readable
        rr["read_error"] = read_error
        rows.append(rr)

        progress.progress(i / max(total, 1))
        msg.info(f"{i}/{total} | {stac_id} -> {status}")

    progress.empty()
    msg.empty()

    st.session_state.downloaded_df = pd.DataFrame(rows)


def run_load_local_folder(ui_cfg):
    df = build_downloaded_df_from_local_folder(ui_cfg["local_tif_folder"])
    df = add_distance_to_target(df, ui_cfg["target_lat"], ui_cfg["target_lon"])
    st.session_state.downloaded_df = df
    st.session_state.active_bbox = resolve_active_bbox(ui_cfg)


def run_analysis(ui_cfg, core_cfg: DemoConfig):
    downloaded = st.session_state.downloaded_df.copy()
    if len(downloaded) == 0:
        return

    ok = downloaded[downloaded["download_status"].isin(["exists", "downloaded", "local"])].copy()
    if len(ok) == 0:
        st.warning("No TIFFs are available for analysis.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, load_info = load_model(ui_cfg["checkpoint_path"], device=device, cfg=core_cfg)

    active_bbox = resolve_active_bbox(ui_cfg)
    st.session_state.active_bbox = active_bbox

    patch_embed_df = run_patch_embedding_analysis(
        model=model,
        downloaded_df=ok,
        target_lat=ui_cfg["target_lat"],
        target_lon=ui_cfg["target_lon"],
        device=device,
        cfg=core_cfg,
        bbox=active_bbox,
    )

    if patch_embed_df is None or len(patch_embed_df) == 0:
        st.warning(
            "No valid patches were extracted from the selected scenes. "
            "Some TIFFs may be partially unreadable, the selected lat/lon may not fall into useful image content, "
            "or the bbox is too restrictive or misaligned."
        )
        st.session_state.patch_embed_df = pd.DataFrame()
        st.session_state.patch_summary_df = pd.DataFrame()
        st.session_state.patch_timelines = {}
        return load_info

    patch_summary_df, patch_timelines = build_all_patch_timeline_summaries(patch_embed_df)

    st.session_state.patch_embed_df = patch_embed_df
    st.session_state.patch_summary_df = patch_summary_df
    st.session_state.patch_timelines = patch_timelines

    return load_info


def render_main_page(core_cfg: DemoConfig):
    init_state()

    st.title(core_cfg.page_title)
    st.write(
        "This version supports catalog download mode and direct local TIFF folder mode, "
        "and can optionally constrain the analysis to a user-provided bbox."
    )

    ui_cfg = render_sidebar(core_cfg)

    active_bbox_preview = resolve_active_bbox(ui_cfg)
    if ui_cfg["use_bbox"]:
        if active_bbox_preview is not None:
            st.info(
                f"BBox mode is active: xmin={active_bbox_preview[0]:.3f}, "
                f"ymin={active_bbox_preview[1]:.3f}, "
                f"xmax={active_bbox_preview[2]:.3f}, "
                f"ymax={active_bbox_preview[3]:.3f}"
            )
        else:
            st.warning("BBox mode is on, but no valid bbox could be parsed.")

    if ui_cfg["input_mode"] == "Local TIFF folder":
        st.info("Local TIFF folder mode is active. You can load TIFFs directly from the folder path without CSV filtering or download.")

    st.subheader("1. Load catalog CSV")
    uploaded = st.file_uploader("Upload Capella catalog CSV", type=["csv"])

    if uploaded is not None:
        try:
            st.session_state.catalog_df = pd.read_csv(uploaded)
            st.success(f"Loaded catalog with {len(st.session_state.catalog_df)} rows.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if len(st.session_state.catalog_df) > 0:
        with st.expander("Catalog preview", expanded=False):
            st.dataframe(st.session_state.catalog_df.head(20), use_container_width=True)

    st.subheader("2. Selected location")
    c1, c2, c3 = st.columns(3)
    c1.metric("Preset", ui_cfg["location_name"])
    c2.metric("Latitude", f"{ui_cfg['target_lat']:.6f}")
    c3.metric("Longitude", f"{ui_cfg['target_lon']:.6f}")

    st.subheader("3. Prepare scenes")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("Run scene filtering", use_container_width=True):
            try:
                if len(st.session_state.catalog_df) == 0:
                    st.warning("Please upload a catalog CSV first.")
                else:
                    run_filtering(ui_cfg)
                    st.success(f"Found {len(st.session_state.filtered_df)} GEO scenes.")
            except Exception as e:
                st.error(str(e))

    with col_b:
        if st.button("Download TIFFs", use_container_width=True):
            if len(st.session_state.filtered_df) == 0:
                st.warning("Please filter scenes first.")
            else:
                run_download(ui_cfg)
                st.success("Download step finished.")

    with col_c:
        if st.button("Load local TIFF folder", use_container_width=True):
            try:
                run_load_local_folder(ui_cfg)
                st.success(f"Loaded {len(st.session_state.downloaded_df)} local TIFF files.")
            except Exception as e:
                st.error(str(e))

    if len(st.session_state.filtered_df) > 0:
        st.markdown("Filtered scenes")
        show_cols = [c for c in ["stac_id", "start_datetime", "center_lat", "center_lon", "distance_km", "tif_url"] if c in st.session_state.filtered_df.columns]
        st.dataframe(st.session_state.filtered_df[show_cols], use_container_width=True)

    if len(st.session_state.downloaded_df) > 0:
        st.markdown("Available scenes")
        show_cols = [c for c in [
            "stac_id",
            "start_datetime",
            "center_lat",
            "center_lon",
            "distance_km",
            "local_tif_path",
            "download_status",
            "download_error",
            "is_readable",
            "read_error",
        ] if c in st.session_state.downloaded_df.columns]
        st.dataframe(st.session_state.downloaded_df[show_cols], use_container_width=True)

    st.subheader("4. Run analysis")
    if st.button("Run patch analysis", use_container_width=True):
        if len(st.session_state.downloaded_df) == 0:
            st.warning("Please either download TIFFs or load a local TIFF folder first.")
        else:
            try:
                load_info = run_analysis(ui_cfg, core_cfg)
                st.success("Patch analysis finished.")
                with st.expander("Checkpoint load info", expanded=False):
                    st.write(load_info)
            except Exception as e:
                st.error(f"Patch analysis failed: {e}")

    if len(st.session_state.patch_summary_df) > 0:
        tabs = st.tabs(["Interesting", "Temporary", "Continuous", "Localization"])
        active_bbox = st.session_state.get("active_bbox", None)

        with tabs[0]:
            st.markdown("Top interesting patch pairs")
            top_pairs_df = st.session_state.patch_summary_df.head(ui_cfg["top_k_patches"]).copy()
            st.dataframe(top_pairs_df, use_container_width=True)

            st.markdown("Top interesting timelines")
            top_tl_df = st.session_state.patch_summary_df.head(ui_cfg["top_k_timelines"]).copy()
            st.dataframe(top_tl_df, use_container_width=True)

            timeline_patch_ids = top_tl_df["patch_id"].tolist()
            selected_patch_id = st.selectbox("Select an interesting patch timeline", timeline_patch_ids)

            if selected_patch_id is not None:
                tl = st.session_state.patch_timelines[selected_patch_id]
                row = st.session_state.patch_summary_df[
                    st.session_state.patch_summary_df["patch_id"] == selected_patch_id
                ].iloc[0]

                plot_single_patch_timeline(
                    tl=tl,
                    title=f"Timeline for {selected_patch_id} | interesting_score={row['interesting_score']:.4f}",
                )
                cols_show = [c for c in [
                    "stac_id",
                    "date_str",
                    "patch_id",
                    "valid_ratio",
                    "center_valid_ratio",
                    "dist_to_ref0",
                    "step_change",
                    "change_score",
                ] if c in tl.columns]
                st.dataframe(tl[cols_show], use_container_width=True)

            st.markdown("Top interesting patch pairs previews")
            for _, row in top_pairs_df.iterrows():
                if pd.isna(row["best_change_before_scene"]) or pd.isna(row["best_change_after_scene"]):
                    continue

                with st.expander(
                    f"{row['patch_id']} | score={row['interesting_score']:.4f} | {row['best_change_before_date']} -> {row['best_change_after_date']}",
                    expanded=False,
                ):
                    title = (
                        f"Patch {row['patch_id']} | "
                        f"{row['best_change_before_date']} -> {row['best_change_after_date']} | "
                        f"interesting_score={row['interesting_score']:.4f}"
                    )

                    before_rows = st.session_state.downloaded_df[
                        st.session_state.downloaded_df["stac_id"] == row["best_change_before_scene"]
                    ]
                    after_rows = st.session_state.downloaded_df[
                        st.session_state.downloaded_df["stac_id"] == row["best_change_after_scene"]
                    ]

                    if len(before_rows) > 0 and len(after_rows) > 0:
                        plot_patch_pair(
                            before_path=before_rows.iloc[0]["local_tif_path"],
                            after_path=after_rows.iloc[0]["local_tif_path"],
                            target_lat=ui_cfg["target_lat"],
                            target_lon=ui_cfg["target_lon"],
                            patch_row=int(row["patch_row"]),
                            patch_col=int(row["patch_col"]),
                            title=title,
                            cfg=core_cfg,
                            bbox=active_bbox,
                        )

        with tabs[1]:
            st.markdown("Top temporary changes")
            temp_df = (
                st.session_state.patch_summary_df[
                    st.session_state.patch_summary_df["event_type"] == "temporary"
                ]
                .sort_values(["event_score", "interesting_score"], ascending=[False, False])
                .head(10)
                .copy()
            )

            cols_show = [c for c in [
                "patch_id",
                "event_type",
                "event_score",
                "interesting_score",
                "num_dates",
                "max_dist_to_prev",
                "max_dist_to_ref0",
            ] if c in temp_df.columns]
            st.dataframe(temp_df[cols_show], use_container_width=True)

            for _, row in temp_df.iterrows():
                patch_id = row["patch_id"]
                tl = st.session_state.patch_timelines.get(patch_id, None)
                if tl is None:
                    continue

                with st.expander(
                    f"{patch_id} | temporary | score={row['event_score']:.4f}",
                    expanded=False,
                ):
                    fig = plot_cumulative_event_timeline_figure(
                        tl=tl,
                        target_lat=ui_cfg["target_lat"],
                        target_lon=ui_cfg["target_lon"],
                        patch_row=int(row["patch_row"]),
                        patch_col=int(row["patch_col"]),
                        title=f"Temporary change | {patch_id} | score={row['event_score']:.4f}",
                        event_type="temporary",
                        cfg=core_cfg,
                        event_idx=int(row["event_idx"]) if not pd.isna(row["event_idx"]) else None,
                        event_start_idx=None,
                        event_end_idx=None,
                        bbox=active_bbox,
                    )
                    st.pyplot(fig)
                    plt.close(fig)

                    cols_show = [c for c in [
                        "stac_id",
                        "date_str",
                        "patch_id",
                        "step_change",
                        "cumulative_step_change",
                        "dist_to_ref0",
                        "step_z",
                        "ref_gain",
                        "trend_step",
                        "change_score",
                    ] if c in tl.columns]
                    st.dataframe(tl[cols_show], use_container_width=True)

        with tabs[2]:
            st.markdown("Top continuous changes")
            cont_df = (
                st.session_state.patch_summary_df[
                    st.session_state.patch_summary_df["event_type"] == "continuous"
                ]
                .sort_values(["event_score", "interesting_score"], ascending=[False, False])
                .head(10)
                .copy()
            )

            cols_show = [c for c in [
                "patch_id",
                "event_type",
                "event_score",
                "interesting_score",
                "num_dates",
                "max_dist_to_prev",
                "max_dist_to_ref0",
            ] if c in cont_df.columns]
            st.dataframe(cont_df[cols_show], use_container_width=True)

            for _, row in cont_df.iterrows():
                patch_id = row["patch_id"]
                tl = st.session_state.patch_timelines.get(patch_id, None)
                if tl is None:
                    continue

                with st.expander(
                    f"{patch_id} | continuous | score={row['event_score']:.4f}",
                    expanded=False,
                ):
                    fig = plot_cumulative_event_timeline_figure(
                        tl=tl,
                        target_lat=ui_cfg["target_lat"],
                        target_lon=ui_cfg["target_lon"],
                        patch_row=int(row["patch_row"]),
                        patch_col=int(row["patch_col"]),
                        title=f"Continuous change | {patch_id} | score={row['event_score']:.4f}",
                        event_type="continuous",
                        cfg=core_cfg,
                        event_idx=None,
                        event_start_idx=int(row["event_start_idx"]) if not pd.isna(row["event_start_idx"]) else None,
                        event_end_idx=int(row["event_end_idx"]) if not pd.isna(row["event_end_idx"]) else None,
                        bbox=active_bbox,
                    )
                    st.pyplot(fig)
                    plt.close(fig)

                    cols_show = [c for c in [
                        "stac_id",
                        "date_str",
                        "patch_id",
                        "step_change",
                        "cumulative_step_change",
                        "dist_to_ref0",
                        "step_z",
                        "ref_gain",
                        "trend_step",
                        "change_score",
                    ] if c in tl.columns]
                    st.dataframe(tl[cols_show], use_container_width=True)

        with tabs[3]:
            st.markdown("Multi-scale feature localization on top interesting pairs")
            st.caption(
                "Localization is computed from fused mid/high-level ConvNeXt feature differences. "
                "No retraining is required."
            )

            top_loc_df = st.session_state.patch_summary_df.head(20).copy()
            cols_show = [c for c in [
                "patch_id",
                "interesting_score",
                "best_change_before_date",
                "best_change_after_date",
                "max_dist_to_prev",
                "max_dist_to_ref0",
            ] if c in top_loc_df.columns]
            st.dataframe(top_loc_df[cols_show], use_container_width=True)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _ = load_model(ui_cfg["checkpoint_path"], device=device, cfg=core_cfg)

            for _, row in top_loc_df.iterrows():
                if pd.isna(row["best_change_before_scene"]) or pd.isna(row["best_change_after_scene"]):
                    continue

                before_rows = st.session_state.downloaded_df[
                    st.session_state.downloaded_df["stac_id"] == row["best_change_before_scene"]
                ]
                after_rows = st.session_state.downloaded_df[
                    st.session_state.downloaded_df["stac_id"] == row["best_change_after_scene"]
                ]

                if len(before_rows) == 0 or len(after_rows) == 0:
                    continue

                before_path = before_rows.iloc[0]["local_tif_path"]
                after_path = after_rows.iloc[0]["local_tif_path"]

                exp_title = (
                    f"{row['patch_id']} | {row['best_change_before_date']} -> {row['best_change_after_date']} "
                    f"| score={row['interesting_score']:.4f}"
                )

                with st.expander(exp_title, expanded=False):
                    try:
                        result = localize_pair_from_feature_maps(
                            model=model,
                            before_path=before_path,
                            after_path=after_path,
                            target_lat=ui_cfg["target_lat"],
                            target_lon=ui_cfg["target_lon"],
                            patch_row=int(row["patch_row"]),
                            patch_col=int(row["patch_col"]),
                            device=device,
                            cfg=core_cfg,
                            normalize_per_channel=False,
                            smooth_ksize=7,
                            threshold_method="percentile",
                            threshold_percentile=90.0,
                            threshold_min_abs=0.18,
                            open_ksize=3,
                            close_ksize=7,
                            min_area=80,
                            bbox=active_bbox,
                        )

                        if len(result.get("top_bboxes", [])) > 0:
                            st.write("Top localization boxes:")
                            for i, (x0, y0, x1, y1, area) in enumerate(result["top_bboxes"], start=1):
                                st.write(f"{i}. bbox=({x0}, {y0}, {x1}, {y1}), area={area}")
                        else:
                            st.write("Top localization boxes: No bbox detected")

                        plot_feature_localization_result(
                            result=result,
                            title=(
                                f"Multi-scale feature localization | {row['patch_id']} | "
                                f"{row['best_change_before_date']} -> {row['best_change_after_date']}"
                            ),
                        )
                    except Exception as e:
                        st.error(f"Localization failed: {e}")

    st.info(
        f"Fixed settings used for every run: crop={core_cfg.crop_size_m} m, "
        f"canvas={core_cfg.canvas_size}, patch={core_cfg.patch_size}, stride={core_cfg.stride}. "
        "The app supports catalog+download mode, local TIFF folder mode, and optional bbox-constrained analysis."
    )