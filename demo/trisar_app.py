import argparse
import ast
import streamlit as st

from TRISAR.demo.demo_utils import DemoConfig, render_main_page


def parse_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")


def parse_locations_arg(text: str):
    try:
        obj = ast.literal_eval(text)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Could not parse locations dict: {e}")

    if not isinstance(obj, dict):
        raise argparse.ArgumentTypeError("Locations must be a dict.")

    for key, val in obj.items():
        if not isinstance(key, str):
            raise argparse.ArgumentTypeError("Location names must be strings.")
        if not isinstance(val, dict):
            raise argparse.ArgumentTypeError("Each location value must be a dict.")
        if "lat" not in val or "lon" not in val:
            raise argparse.ArgumentTypeError("Each location dict must contain 'lat' and 'lon'.")
        val["lat"] = float(val["lat"])
        val["lon"] = float(val["lon"])

    return obj


def build_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--default-locations",
        type=parse_locations_arg,
        default={
            "San Jose": {"lat": 37.313149020, "lon": -121.894385880},
            # "Alicante": {"lat": 38.384322, "lon": -0.415070},
        },
    )

    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--use-first-band-only", type=parse_bool, default=True)

    parser.add_argument("--valid-mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-valid-ratio", type=float, default=0.40)
    parser.add_argument("--min-center-valid-ratio", type=float, default=0.55)

    parser.add_argument("--backbone-name", type=str, default="convnext_base")
    parser.add_argument("--pretrained", type=parse_bool, default=False)
    parser.add_argument("--freeze-backbone", type=parse_bool, default=False)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.20)

    parser.add_argument("--crop-size-m", type=int, default=3000)
    parser.add_argument("--canvas-size", type=int, default=3000)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)

    parser.add_argument("--default-top-k-patches", type=int, default=12)
    parser.add_argument("--default-top-k-timelines", type=int, default=12)

    parser.add_argument("--default-checkpoint-path", type=str, default="convnext_change_pipeline/best_model.pt")
    parser.add_argument("--default-download-dir", type=str, default="downloaded_geo_tifs")
    parser.add_argument("--page-title", type=str, default="Capella Patch Timeline Demo")

    return parser


def parse_demo_config():
    parser = build_arg_parser()
    args, _ = parser.parse_known_args()

    return DemoConfig(
        default_locations=args.default_locations,
        img_size=args.img_size,
        use_first_band_only=args.use_first_band_only,
        valid_mask_threshold=args.valid_mask_threshold,
        min_valid_ratio=args.min_valid_ratio,
        min_center_valid_ratio=args.min_center_valid_ratio,
        backbone_name=args.backbone_name,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        crop_size_m=args.crop_size_m,
        canvas_size=args.canvas_size,
        patch_size=args.patch_size,
        stride=args.stride,
        default_top_k_patches=args.default_top_k_patches,
        default_top_k_timelines=args.default_top_k_timelines,
        default_checkpoint_path=args.default_checkpoint_path,
        default_download_dir=args.default_download_dir,
        page_title=args.page_title,
    )


cfg = parse_demo_config()

st.set_page_config(
    page_title=cfg.page_title,
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


if __name__ == "__main__":
    render_main_page(cfg)


# streamlit run trisar_app.py -- \
#   --default-locations "{'San Jose': {'lat': 37.313149020, 'lon': -121.894385880}, 'Alicante': {'lat': 38.384322, 'lon': -0.415070}, 'Oroville': {'lat': 39.526979021, 'lon': -121.513563007}}" \
#   --img-size 256 \
#   --use-first-band-only true \
#   --valid-mask-threshold 0.5 \
#   --min-valid-ratio 0.40 \
#   --min-center-valid-ratio 0.55 \
#   --backbone-name convnext_base \
#   --pretrained false \
#   --freeze-backbone false \
#   --embed-dim 256 \
#   --dropout 0.20 \
#   --crop-size-m 3000 \
#   --canvas-size 3000 \
#   --patch-size 256 \
#   --stride 256 \
#   --default-top-k-patches 12 \
#   --default-top-k-timelines 12 \
#   --default-checkpoint-path convnext_change_pipeline/best_model.pt \
#   --default-download-dir downloaded_geo_tifs \
#   --page-title "Capella Patch Timeline Demo"