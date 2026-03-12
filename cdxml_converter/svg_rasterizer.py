"""
SVG-to-PNG/JPG rasterizer using headless Chromium via Playwright.

Produces pixel-perfect rasterization of SVG content by rendering it
in a real browser engine. Much higher quality than PIL-based rendering.
"""

import io
import re
from PIL import Image


class SVGRasterizer:
    """Rasterizes SVG strings to PNG/JPG using headless Chromium.

    Use as a context manager for efficient batch processing (the browser
    is launched once and reused across multiple rasterize() calls).

    Usage:
        with SVGRasterizer() as rasterizer:
            png_bytes = rasterizer.rasterize(svg_string, scale=2)
            jpg_bytes = rasterizer.rasterize(svg_string, scale=2, fmt="jpg")
    """

    def __init__(self):
        self._pw_ctx = None
        self._playwright = None
        self._browser = None

    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self._pw_ctx = sync_playwright()
        self._playwright = self._pw_ctx.__enter__()
        self._browser = self._playwright.chromium.launch()
        return self

    def __exit__(self, *args):
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._pw_ctx:
            self._pw_ctx.__exit__(*args)
            self._pw_ctx = None
        self._playwright = None

    def rasterize(self, svg_string: str, scale: int = 2,
                  fmt: str = "png", jpg_quality: int = 95) -> bytes:
        """Convert an SVG string to PNG or JPG bytes.

        Args:
            svg_string: Complete SVG markup string.
            scale: Device scale factor (2 = 2x resolution, 4 = 4x, etc.).
            fmt: Output format - "png" or "jpg".
            jpg_quality: JPEG quality (1-100). Only used when fmt="jpg".

        Returns:
            Image bytes in the requested format.
        """
        if self._browser is None:
            raise RuntimeError(
                "SVGRasterizer must be used as a context manager: "
                "with SVGRasterizer() as r: ..."
            )

        # Extract SVG dimensions from the markup
        w, h = _parse_svg_dimensions(svg_string)

        page = self._browser.new_page(device_scale_factor=scale)
        try:
            # Wrap SVG in minimal HTML
            html = (
                '<!DOCTYPE html>'
                '<html><head><style>'
                'html,body{margin:0;padding:0;overflow:hidden;background:white;}'
                'svg{display:block;}'
                '</style></head><body>'
                f'{svg_string}'
                '</body></html>'
            )

            page.set_content(html)
            page.set_viewport_size({
                "width": max(1, int(w + 0.5)),
                "height": max(1, int(h + 0.5)),
            })

            # Screenshot the SVG element
            svg_el = page.query_selector("svg")
            if svg_el is None:
                # Fallback: screenshot the whole page
                png_bytes = page.screenshot(type="png")
            else:
                png_bytes = svg_el.screenshot(type="png")

        finally:
            page.close()

        # For PNG, return directly
        if fmt == "png":
            return png_bytes

        # For JPG, convert via Pillow
        img = Image.open(io.BytesIO(png_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=jpg_quality)
        return buf.getvalue()


def svg_to_png(svg_string: str, scale: int = 2) -> bytes:
    """Convenience function: convert SVG string to PNG bytes.

    Launches and closes a browser for each call. For batch operations,
    use SVGRasterizer as a context manager instead.
    """
    with SVGRasterizer() as r:
        return r.rasterize(svg_string, scale=scale, fmt="png")


def svg_to_jpg(svg_string: str, scale: int = 2, quality: int = 95) -> bytes:
    """Convenience function: convert SVG string to JPG bytes."""
    with SVGRasterizer() as r:
        return r.rasterize(svg_string, scale=scale, fmt="jpg",
                           jpg_quality=quality)


def _parse_svg_dimensions(svg_string: str) -> tuple:
    """Extract width and height from an SVG string.

    Tries the width/height attributes first, then falls back to viewBox.

    Returns:
        (width, height) in pixels as floats.
    """
    w_match = re.search(r'<svg[^>]+width="([\d.]+)"', svg_string)
    h_match = re.search(r'<svg[^>]+height="([\d.]+)"', svg_string)

    if w_match and h_match:
        return float(w_match.group(1)), float(h_match.group(1))

    # Fall back to viewBox
    vb_match = re.search(
        r'viewBox="([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)"',
        svg_string,
    )
    if vb_match:
        return float(vb_match.group(3)), float(vb_match.group(4))

    # Default fallback
    return 800.0, 600.0
