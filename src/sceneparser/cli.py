"""
Command-Line Interface for SpriteStack Scene Parser
===================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import SceneConstructionPipeline
from .shared.render import level_to_ascii, grid_to_image


def cmd_parse(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        print("Error: No prompt provided.", file=sys.stderr)
        return 1

    pipeline = SceneConstructionPipeline()
    result = pipeline.run(prompt)

    print("=== Extracted Scene Entities ===")
    print(json.dumps(result["scene_json"], indent=2))

    print("\n=== 2D Canvas Placements ===")
    for p in result["placements"]:
        print(f"  • {p['name']} ({p['type']}) at (x={p['x']}, y={p['y']})")

    print("\n=== Generated 16x16 Level Grid ===")
    print(level_to_ascii(result["level_grid"]))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = grid_to_image(result["level_grid"], theme=result["theme"])
        img.save(out_path)
        print(f"\nSaved level visual rendering to: {out_path}")

    return 0


def cmd_wide(args: argparse.Namespace) -> int:
    prompts = args.prompts
    if not prompts:
        print("Error: Provide at least one prompt for wide level generation.", file=sys.stderr)
        return 1

    pipeline = SceneConstructionPipeline()
    result = pipeline.run_wide(prompts)

    print(f"Generated {len(prompts)}-screen continuous level (Width: {result['width']} tiles, Height: 16)")
    print(level_to_ascii(result["level_grid"]))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = grid_to_image(result["level_grid"])
        img.save(out_path)
        print(f"\nSaved wide level image to: {out_path}")

    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is not installed. Run: pip install uvicorn", file=sys.stderr)
        return 1

    from .server import app
    print(f"Starting SpriteStack Scene Parser API on {args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    import torch
    pipe = SceneConstructionPipeline()
    print("SpriteStack Scene Parser Information")
    print("====================================")
    print(f"PyTorch version:             {torch.__version__}")
    print(f"CUDA available:              {torch.cuda.is_available()}")
    print(f"Stage 1 DistilBERT loaded:   {pipe.parser.is_transformer_loaded}")
    print(f"Stage 2 Architect loaded:    {pipe.generator.is_loaded}")
    print(f"Execution Device:            {pipe.generator.device}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sceneparser",
        description="Two-Stage NLP Scene Construction & Level Generator for SpriteStack Studio",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Subcommand: parse
    p_parse = sub.add_parser("parse", help="Parse prompt into structured entities & 16x16 level")
    p_parse.add_argument("prompt", nargs="+", help="Natural language scene description")
    p_parse.add_argument("--output", "-o", help="Optional path to save visual PNG render")

    # Subcommand: wide
    p_wide = sub.add_parser("wide", help="Generate multi-screen stitched continuous level")
    p_wide.add_argument("prompts", nargs="+", help="One prompt per screen chunk")
    p_wide.add_argument("--output", "-o", help="Optional path to save wide PNG render")

    # Subcommand: serve
    p_serve = sub.add_parser("serve", help="Launch FastAPI server for SpriteStack Studio")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host interface")
    p_serve.add_argument("--port", type=int, default=8000, help="Port")

    # Subcommand: info
    sub.add_parser("info", help="Show system environment and model detection info")

    args = parser.parse_args(argv)

    if args.command == "parse":
        return cmd_parse(args)
    elif args.command == "wide":
        return cmd_wide(args)
    elif args.command == "serve":
        return cmd_serve(args)
    elif args.command == "info":
        return cmd_info(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
