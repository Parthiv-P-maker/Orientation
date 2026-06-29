"""CLI entry point for the Reference-Based Object Orientation Measurement System."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import AppConfig
from .detector import create_detector


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reference-Based Object Orientation Measurement System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=("image", "video", "live"), default="live", help="Operating mode.")
    p.add_argument("--reference", help="Reference image path (image mode).")
    p.add_argument("--source", help="Input image/video path.")
    p.add_argument("--webcam", type=int, default=0, help="Camera index for live mode.")
    p.add_argument("--output", default="outputs", help="Output directory.")
    p.add_argument("--output-video", help="Path for annotated output video.")

    p.add_argument("--mask-mode", choices=("color", "hsv", "dark", "bright", "adaptive", "edge", "background"), default="color")
    p.add_argument("--method", choices=("moments", "pca", "box", "ellipse", "hull"), default="moments")
    p.add_argument("--tracker", choices=("centroid",), default="centroid")

    p.add_argument("--hue-min", type=int, default=0)
    p.add_argument("--hue-max", type=int, default=179)
    p.add_argument("--sat-min", type=int, default=55)
    p.add_argument("--value-min", type=int, default=45)
    p.add_argument("--min-area", type=float, default=1000.0)
    p.add_argument("--max-objects", type=int)
    p.add_argument("--largest-only", action="store_true")

    p.add_argument("--decimals", type=int, default=2)
    p.add_argument("--camera-width", type=int, default=1280)
    p.add_argument("--camera-height", type=int, default=720)

    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--save-json", action="store_true")
    p.add_argument("--show-debug", action="store_true")
    p.add_argument("--no-display", action="store_true")

    p.add_argument("--inspection", action="store_true", help="Enable tolerance inspection.")
    p.add_argument("--target-angle", type=float, default=0.0)
    p.add_argument("--tolerance", type=float, default=2.0)

    p.add_argument("--config", help="Load settings from a JSON config file.")
    p.add_argument("--save-config", help="Save current settings to a JSON config file.")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        cfg = AppConfig.load(Path(args.config))
    else:
        cfg = AppConfig()

    if args.save_config:
        cfg.save(Path(args.save_config))
        print(f"Config saved to {args.save_config}")
        return

    max_objects = 1 if args.largest_only else args.max_objects

    detector = create_detector(
        mask_mode=args.mask_mode,
        method=args.method,
        min_area=args.min_area,
        max_objects=max_objects,
        hue_min=args.hue_min,
        hue_max=args.hue_max,
        sat_min=args.sat_min,
        value_min=args.value_min,
    )

    output_dir = Path(args.output)

    if args.mode == "image":
        from .image_mode import run_image_mode
        if not args.reference:
            parser.error("--reference is required for image mode.")
        if not args.source:
            parser.error("--source is required for image mode.")
        run_image_mode(args.reference, args.source, detector, output_dir, args.decimals, args.no_display)

    elif args.mode == "video":
        from .video_mode import run_video_mode
        if not args.source:
            parser.error("--source is required for video mode.")
        run_video_mode(
            args.source, detector, output_dir,
            output_video=args.output_video,
            no_display=args.no_display,
            save_csv=args.save_csv,
            save_json=args.save_json,
            decimals=args.decimals,
            show_debug=args.show_debug,
        )

    elif args.mode == "live":
        from .live_mode import run_live_mode
        run_live_mode(
            args.webcam, detector, output_dir,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            output_video=args.output_video,
            save_csv=args.save_csv,
            save_json=args.save_json,
            decimals=args.decimals,
            show_debug=args.show_debug,
            inspection_enabled=args.inspection,
            target_angle=args.target_angle,
            tolerance=args.tolerance,
        )


if __name__ == "__main__":
    main()
