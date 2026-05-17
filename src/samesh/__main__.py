"""Command-line entry point for SAMesh."""
import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "mesh_segmentation.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="samesh",
        description="Zero-shot 3D mesh segmentation using SAM2.",
    )
    parser.add_argument("input", type=str, help="Path to input mesh file (e.g. .glb).")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Path to YAML config. Defaults to {DEFAULT_CONFIG} if it exists.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory override.")
    parser.add_argument("--cache", type=str, default=None, help="Cache directory override.")
    parser.add_argument("--checkpoint", type=str, default=None, help="SAM2 checkpoint path override.")
    parser.add_argument("--visualize", action="store_true", help="Save per-view visualizations.")
    parser.add_argument("--extension", type=str, default="glb", help="Output mesh extension (default: glb).")
    parser.add_argument(
        "--target-labels",
        type=str,
        default=None,
        help="Optional JSON file or comma-separated list of target labels.",
    )
    parser.add_argument("--texture", action="store_true", help="Preserve mesh texture instead of stripping it.")
    return parser


def _resolve_config_path(arg_config: str | None) -> Path:
    if arg_config is not None:
        path = Path(arg_config)
        if not path.is_file():
            raise FileNotFoundError(f"Config not found: {path}")
        return path
    if DEFAULT_CONFIG.is_file():
        return DEFAULT_CONFIG
    raise FileNotFoundError(
        f"No --config provided and default config not found at {DEFAULT_CONFIG}."
    )


def _resolve_target_labels(arg: str | None):
    if arg is None:
        return None
    path = Path(arg)
    if path.is_file():
        import json
        with path.open() as f:
            return json.load(f)
    return [s.strip() for s in arg.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config_path = _resolve_config_path(args.config)
    config = OmegaConf.load(config_path)

    if args.output is not None:
        config.output = args.output
    if args.cache is not None:
        config.cache = args.cache
    if args.checkpoint is not None:
        config.sam.checkpoint = args.checkpoint

    target_labels = _resolve_target_labels(args.target_labels)

    from samesh.models.sam_mesh import segment_mesh

    segment_mesh(
        args.input,
        config,
        visualize=args.visualize,
        extension=args.extension,
        target_labels=target_labels,
        texture=args.texture,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
