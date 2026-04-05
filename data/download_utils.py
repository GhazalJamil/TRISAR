# IEEE DATA FUSION CONTEST
# BY JAMIL GHAZAL
# FEW OF OUR FUNCTIONS ARE  DERIVED FROM CAPLELLA SPACE EXAMPLE NOTEBOOK:
# https://www.grss-ieee.org/wp-content/uploads/2026/02/STAC_to_CSV.ipynb_.zip 

import csv
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from rasterio.warp import transform_bounds


PathLike = Union[str, Path]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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


def find_geo_scenes_near_point(
    csv_path: PathLike,
    output_csv: PathLike,
    target_lat: float,
    target_lon: float,
    delta_lat: float = 0.1,
    delta_lon: float = 0.1,
    max_rows: Optional[int] = None,
    limit_rows: Optional[int] = None,
) -> pd.DataFrame:
    print("Loading metadata...")
    df = pd.read_csv(csv_path)

    print("Filtering GEO scenes...")
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

    print("Computing distances...")
    filtered["distance_km"] = filtered.apply(
        lambda r: haversine_km(
            target_lat,
            target_lon,
            float(r["center_lat"]),
            float(r["center_lon"]),
        ),
        axis=1,
    )

    sort_cols = ["distance_km"]
    if "start_datetime" in filtered.columns:
        sort_cols.append("start_datetime")

    filtered = filtered.sort_values(sort_cols).reset_index(drop=True)

    if max_rows is not None:
        filtered = filtered.head(max_rows).copy()

    print("Building tif urls...")
    filtered["tif_url"] = filtered.apply(
        lambda r: build_tif_url(r["stac_id"], r["start_datetime"]),
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

    filtered = filtered[final_cols]

    if limit_rows is not None:
        filtered = filtered.head(limit_rows).copy()

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)

    print(f"Saved {len(filtered)} rows to {output_csv}")
    return filtered


def download_file(url: str, out_path: Path, timeout: int = 120, chunk_size: int = 1024 * 1024) -> None:
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def download_tifs_from_csv(
    input_csv: PathLike,
    out_dir: PathLike,
    timeout: int = 120,
) -> None:
    print("Loading scene table...")
    df = pd.read_csv(input_csv)

    if "tif_url" not in df.columns:
        raise ValueError("Column 'tif_url' is missing.")
    if "stac_id" not in df.columns:
        raise ValueError("Column 'stac_id' is missing.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(df)} rows to download.")

    for i, row in df.iterrows():
        stac_id = str(row["stac_id"])
        tif_url = str(row["tif_url"]).strip()

        if not tif_url or tif_url.lower() == "nan":
            print(f"[{i + 1}/{len(df)}] Skipping {stac_id}, missing url.")
            continue

        out_path = out_dir / f"{stac_id}.tif"

        if out_path.exists():
            print(f"[{i + 1}/{len(df)}] File already exists: {out_path.name}")
            continue

        print(f"[{i + 1}/{len(df)}] Downloading {stac_id}...")
        try:
            download_file(tif_url, out_path, timeout=timeout)
        except Exception as e:
            print(f"[{i + 1}/{len(df)}] Download failed for {stac_id}: {e}")

    print("Download finished.")


def list_tifs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.tif") if p.is_file()])


def sanitize_bbox_name(bounds: Tuple[float, float, float, float], decimals: int = 3) -> str:
    minx, miny, maxx, maxy = bounds
    return (
        f"bbox_{minx:.{decimals}f}_{miny:.{decimals}f}_{maxx:.{decimals}f}_{maxy:.{decimals}f}"
    ).replace("-", "m")


def patch_is_empty(arr: np.ndarray, nodata: Optional[float] = None) -> bool:
    if arr is None or arr.size == 0:
        return True
    if np.all(np.isnan(arr)):
        return True
    if nodata is not None:
        valid = arr != nodata
        if not np.any(valid):
            return True
    if np.all(arr == 0):
        return True
    return False


def save_patch(out_path: Path, arr: np.ndarray, src_profile: dict, out_transform) -> None:
    profile = src_profile.copy()
    profile.update(
        {
            "driver": "GTiff",
            "height": arr.shape[1],
            "width": arr.shape[2],
            "transform": out_transform,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)


def read_patch_by_bounds(
    tif_path: Path,
    target_bounds: Tuple[float, float, float, float],
    target_crs,
    out_size: int,
    skip_empty_patches: bool,
    nodata_override: Optional[float],
):
    with rasterio.open(tif_path) as src:
        src_nodata = nodata_override if nodata_override is not None else src.nodata

        if src.crs != target_crs:
            local_bounds = transform_bounds(target_crs, src.crs, *target_bounds, densify_pts=21)
        else:
            local_bounds = target_bounds

        win = from_bounds(*local_bounds, transform=src.transform)
        full_win = Window(0, 0, src.width, src.height)

        try:
            win = win.intersection(full_win)
        except Exception:
            return None, None, None

        if win.width <= 1 or win.height <= 1:
            return None, None, None

        arr = src.read(
            window=win,
            out_shape=(src.count, out_size, out_size),
            resampling=Resampling.bilinear,
        )

        if skip_empty_patches and patch_is_empty(arr, src_nodata):
            return None, None, None

        out_transform = rasterio.transform.from_bounds(*target_bounds, out_size, out_size)
        profile = src.profile.copy()
        return arr, profile, out_transform


def process_area(
    area_dir: Path,
    output_root: Path,
    patch_size: int,
    require_full_patch: bool,
    skip_empty_patches: bool,
    min_images_per_patch: int,
    nodata_override: Optional[float],
) -> None:
    area_name = area_dir.name
    print(f"Processing area {area_name}...")

    tif_paths = list_tifs(area_dir)
    if not tif_paths:
        print(f"No tif files found in {area_dir}")
        return

    ref_tif = tif_paths[0]
    print(f"Using reference tif {ref_tif.name}")
    print(f"Found {len(tif_paths)} tif files")

    area_out_dir = output_root / area_name
    area_out_dir.mkdir(parents=True, exist_ok=True)

    total_ref_patches = 0
    saved_groups = 0
    removed_groups = 0

    with rasterio.open(ref_tif) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_profile = ref.profile.copy()
        ref_nodata = nodata_override if nodata_override is not None else ref.nodata

        n_cols = ref.width // patch_size
        n_rows = ref.height // patch_size

        print(f"Reference grid: {n_rows} rows x {n_cols} cols")

        for row in range(n_rows):
            for col in range(n_cols):
                x_off = col * patch_size
                y_off = row * patch_size
                win = Window(x_off, y_off, patch_size, patch_size)

                if require_full_patch:
                    if x_off + patch_size > ref.width or y_off + patch_size > ref.height:
                        continue

                ref_arr = ref.read(window=win)

                if skip_empty_patches and patch_is_empty(ref_arr, ref_nodata):
                    continue

                patch_bounds = rasterio.windows.bounds(win, ref_transform)
                bbox_name = sanitize_bbox_name(patch_bounds, decimals=3)
                bbox_dir = area_out_dir / bbox_name
                bbox_dir.mkdir(parents=True, exist_ok=True)

                ref_patch_transform = window_transform(win, ref_transform)
                ref_patch_path = bbox_dir / ref_tif.name
                if not ref_patch_path.exists():
                    save_patch(ref_patch_path, ref_arr, ref_profile, ref_patch_transform)

                saved_count = 1

                for tif_path in tif_paths[1:]:
                    arr, profile, out_transform = read_patch_by_bounds(
                        tif_path=tif_path,
                        target_bounds=patch_bounds,
                        target_crs=ref_crs,
                        out_size=patch_size,
                        skip_empty_patches=skip_empty_patches,
                        nodata_override=nodata_override,
                    )

                    if arr is None:
                        continue

                    out_path = bbox_dir / tif_path.name
                    if not out_path.exists():
                        save_patch(out_path, arr, profile, out_transform)

                    saved_count += 1

                total_ref_patches += 1

                if saved_count < min_images_per_patch:
                    shutil.rmtree(bbox_dir, ignore_errors=True)
                    removed_groups += 1
                else:
                    saved_groups += 1

                if total_ref_patches % 100 == 0:
                    print(
                        f"Processed {total_ref_patches} reference patches, "
                        f"kept {saved_groups} groups, removed {removed_groups} groups"
                    )

    print(f"Finished area {area_name}")
    print(f"Processed reference patches: {total_ref_patches}")
    print(f"Kept bbox groups: {saved_groups}")
    print(f"Removed bbox groups: {removed_groups}")


def build_patch_dataset(
    input_root: PathLike,
    output_root: PathLike,
    patch_size: int = 512,
    require_full_patch: bool = True,
    skip_empty_patches: bool = True,
    min_images_per_patch: int = 2,
    nodata_override: Optional[float] = None,
) -> None:
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    area_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if not area_dirs:
        raise RuntimeError(f"No area folders found in {input_root}")

    print(f"Found {len(area_dirs)} area folders")

    for area_dir in area_dirs:
        process_area(
            area_dir=area_dir,
            output_root=output_root,
            patch_size=patch_size,
            require_full_patch=require_full_patch,
            skip_empty_patches=skip_empty_patches,
            min_images_per_patch=min_images_per_patch,
            nodata_override=nodata_override,
        )

    print(f"Patch dataset saved to {output_root}")


def list_area_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_bbox_dirs(area_dir: Path) -> List[Path]:
    return sorted([p for p in area_dir.iterdir() if p.is_dir()])


def parse_split(area_name: str, validation_area_names: Set[str]) -> str:
    return "val" if area_name in validation_area_names else "train"


def build_image_manifest(root: Path, validation_area_names: Set[str]):
    images = []
    bbox_to_images: Dict[Tuple[str, str, str], List[dict]] = {}
    image_counter = 0

    for area_dir in list_area_dirs(root):
        area_name = area_dir.name
        split = parse_split(area_name, validation_area_names)

        for bbox_dir in list_bbox_dirs(area_dir):
            bbox_name = bbox_dir.name
            tif_paths = list_tifs(bbox_dir)

            if len(tif_paths) == 0:
                continue

            key = (split, area_name, bbox_name)
            bbox_to_images[key] = []

            for tif_path in tif_paths:
                image_id = f"{split}_{image_counter:08d}"
                image_counter += 1

                rec = {
                    "image_id": image_id,
                    "file_path": str(tif_path.as_posix()),
                    "split": split,
                    "area_name": area_name,
                    "bbox_name": bbox_name,
                    "filename": tif_path.name,
                }

                images.append(rec)
                bbox_to_images[key].append(rec)

    return images, bbox_to_images


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        print(f"No rows to save for {path}")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sample_negative_image(
    bbox_to_images: Dict[Tuple[str, str, str], List[dict]],
    split: str,
    exclude_area: str,
    exclude_bbox: str,
) -> dict:
    same_area_candidates = []
    other_candidates = []

    for (sp, area_name, bbox_name), imgs in bbox_to_images.items():
        if sp != split:
            continue
        if area_name == exclude_area and bbox_name == exclude_bbox:
            continue

        if area_name == exclude_area:
            same_area_candidates.extend(imgs)
        else:
            other_candidates.extend(imgs)

    pool = same_area_candidates if same_area_candidates else other_candidates
    if not pool:
        raise RuntimeError(
            f"No negative candidates found for split={split}, area={exclude_area}, bbox={exclude_bbox}"
        )

    return random.choice(pool)


def build_pairs(
    bbox_to_images: Dict[Tuple[str, str, str], List[dict]],
    negatives_per_positive: int,
):
    train_pairs = []
    val_pairs = []

    for (split, area_name, bbox_name), imgs in bbox_to_images.items():
        if len(imgs) < 2:
            continue

        positives = []
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                positives.append((imgs[i], imgs[j]))

        pair_rows = []

        for a, b in positives:
            pair_rows.append(
                {
                    "anchor_id": a["image_id"],
                    "anchor_path": a["file_path"],
                    "other_id": b["image_id"],
                    "other_path": b["file_path"],
                    "label": 1,
                    "split": split,
                    "area_name": area_name,
                    "bbox_name": bbox_name,
                }
            )

        for a, _ in positives:
            for _ in range(negatives_per_positive):
                neg = sample_negative_image(
                    bbox_to_images=bbox_to_images,
                    split=split,
                    exclude_area=area_name,
                    exclude_bbox=bbox_name,
                )

                pair_rows.append(
                    {
                        "anchor_id": a["image_id"],
                        "anchor_path": a["file_path"],
                        "other_id": neg["image_id"],
                        "other_path": neg["file_path"],
                        "label": 0,
                        "split": split,
                        "area_name": area_name,
                        "bbox_name": bbox_name,
                    }
                )

        if split == "train":
            train_pairs.extend(pair_rows)
        else:
            val_pairs.extend(pair_rows)

    return train_pairs, val_pairs


def build_triplets(
    bbox_to_images: Dict[Tuple[str, str, str], List[dict]],
    negatives_per_positive: int,
):
    train_triplets = []
    val_triplets = []

    for (split, area_name, bbox_name), imgs in bbox_to_images.items():
        if len(imgs) < 2:
            continue

        for i in range(len(imgs)):
            for j in range(len(imgs)):
                if i == j:
                    continue

                anchor = imgs[i]
                positive = imgs[j]

                for _ in range(negatives_per_positive):
                    negative = sample_negative_image(
                        bbox_to_images=bbox_to_images,
                        split=split,
                        exclude_area=area_name,
                        exclude_bbox=bbox_name,
                    )

                    row = {
                        "anchor_id": anchor["image_id"],
                        "anchor_path": anchor["file_path"],
                        "positive_id": positive["image_id"],
                        "positive_path": positive["file_path"],
                        "negative_id": negative["image_id"],
                        "negative_path": negative["file_path"],
                        "split": split,
                        "area_name": area_name,
                        "bbox_name": bbox_name,
                    }

                    if split == "train":
                        train_triplets.append(row)
                    else:
                        val_triplets.append(row)

    return train_triplets, val_triplets


def build_dataset_csvs(
    patch_root: PathLike,
    out_dir: PathLike,
    validation_area_names: Optional[Set[str]] = None,
    negatives_per_positive: int = 2,
    seed: int = 42,
) -> None:
    if validation_area_names is None:
        validation_area_names = {"downloaded_geo_tifs_validation"}

    random.seed(seed)

    patch_root = Path(patch_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building image manifest...")
    images, bbox_to_images = build_image_manifest(patch_root, validation_area_names)

    print(f"Total images: {len(images)}")
    print(f"Total bbox groups: {len(bbox_to_images)}")

    train_images = [x for x in images if x["split"] == "train"]
    val_images = [x for x in images if x["split"] == "val"]

    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")

    print("Building pairs...")
    train_pairs, val_pairs = build_pairs(
        bbox_to_images=bbox_to_images,
        negatives_per_positive=negatives_per_positive,
    )
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")

    print("Building triplets...")
    train_triplets, val_triplets = build_triplets(
        bbox_to_images=bbox_to_images,
        negatives_per_positive=negatives_per_positive,
    )
    print(f"Train triplets: {len(train_triplets)}")
    print(f"Validation triplets: {len(val_triplets)}")

    write_csv(out_dir / "train_images.csv", train_images)
    write_csv(out_dir / "val_images.csv", val_images)
    write_csv(out_dir / "train_pairs.csv", train_pairs)
    write_csv(out_dir / "val_pairs.csv", val_pairs)
    write_csv(out_dir / "train_triplets.csv", train_triplets)
    write_csv(out_dir / "val_triplets.csv", val_triplets)

    print(f"Dataset csv files saved to {out_dir}")