"""
CLI entry point for CDXML to SVG/PNG/JPG converter.

Usage:
    python convert.py input.cdxml                  # Converts to all formats
    python convert.py input.cdxml -f svg png       # Specific formats
    python convert.py input.cdxml -o output_dir    # Custom output directory
    python convert.py *.cdxml                      # Batch convert (shell glob)
    python convert.py input.cdxml --scale 3        # Higher resolution
    python convert.py input.cdxml --line-width 1.2 # Thicker structure lines
    python convert.py input.cdxml -f svg/png --font "font name"  # Use a custom font for text labels in images

"""

import argparse
import glob
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cdxml_converter.converter import convert_cdxml


def main():
    parser = argparse.ArgumentParser(
        description="Convert ChemDraw CDXML files to SVG, PNG, and JPG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert.py singlestep_1.cdxml
  python convert.py *.cdxml -f svg png -o output/
  python convert.py complex.cdxml --scale 3 --png-scale 3
        """,
    )

    parser.add_argument(
        "input",
        nargs="+",
        help="Input CDXML file(s). Supports glob patterns.",
    )
    parser.add_argument(
        "-f", "--formats",
        nargs="+",
        choices=["svg", "png", "jpg"],
        default=["svg", "png", "jpg"],
        help="Output format(s). Default: svg png jpg",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory. Default: same as input file.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="SVG rendering scale factor. Default: 2.0",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=15.0,
        help="Padding around the diagram (points). Default: 15.0",
    )
    parser.add_argument(
        "--bg-color",
        default="white",
        help="Background color. Default: white",
    )
    parser.add_argument(
        "--png-scale",
        type=int,
        default=2,
        help="Scale factor for PNG/JPG rasterization. Default: 2",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=None,
        help="Override the structure line width from the CDXML file.",
    )

    parser.add_argument(
    "--font",
    default="Arial",
    help="Font family used in SVG text"
)
    
    args = parser.parse_args()

    # Expand glob patterns
    input_files = []
    for pattern in args.input:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend(expanded)
        else:
            print(f"Warning: No files matching '{pattern}'", file=sys.stderr)

    if not input_files:
        print("Error: No input files found.", file=sys.stderr)
        sys.exit(1)

    # Filter to .cdxml files
    cdxml_files = [f for f in input_files if f.lower().endswith(".cdxml")]
    if not cdxml_files:
        print("Error: No .cdxml files found in input.", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {len(cdxml_files)} file(s)...")

    # For batch operations, share a single browser instance for PNG/JPG
    need_raster = "png" in args.formats or "jpg" in args.formats
    rasterizer = None
    rasterizer_ctx = None

    if need_raster:
        try:
            from cdxml_converter.svg_rasterizer import SVGRasterizer
            rasterizer_ctx = SVGRasterizer()
            rasterizer = rasterizer_ctx.__enter__()
        except ImportError:
            pass  # Will fall back to PIL inside convert_cdxml

    try:
        success = 0
        failed = 0
        for filepath in cdxml_files:
            try:
                results = convert_cdxml(
                    input_path=filepath,
                    output_dir=args.output_dir,
                    formats=args.formats,
                    scale=args.scale,
                    padding=args.padding,
                    bg_color=args.bg_color,
                    png_scale=args.png_scale,
                    font_family=args.font,
                    line_width=args.line_width,
                    _rasterizer=rasterizer,
                )
                print(f"  {filepath}:")
                for fmt, path in results.items():
                    print(f"    -> {path}")
                success += 1
            except Exception as e:
                print(f"  {filepath}: ERROR - {e}", file=sys.stderr)
                failed += 1
    finally:
        if rasterizer_ctx is not None:
            rasterizer_ctx.__exit__(None, None, None)

    print(f"\nDone: {success} succeeded, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
