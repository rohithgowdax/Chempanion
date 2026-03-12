"""
Converter - High-level API for converting CDXML files to SVG, PNG, JPG.

Enhanced to support fragment-based conversion where each chemical fragment
(molecule) in a CDXML file is converted to a separate image.

For SVG, generates clean SVG markup via the SVGRenderer.
For PNG/JPG, rasterizes the SVG output using headless Chromium (Playwright)
for pixel-perfect results. Falls back to a Pillow-based renderer if
Playwright is not installed.
"""

import io
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from .parser import CDXMLParser, CDXMLDocument, Node, Bond
from .renderer import SVGRenderer

# Check Playwright availability once at import time
_HAS_PLAYWRIGHT = False
try:
    from .svg_rasterizer import SVGRasterizer
    _HAS_PLAYWRIGHT = True
except ImportError:
    pass


def _extract_fragment_elements(cdxml_path: str):
    """Extract all fragment elements from a CDXML file.
    
    Returns:
        List of tuples: (fragment_index, fragment_element)
    """
    tree = ET.parse(cdxml_path)
    root = tree.getroot()
    
    fragments = []
    fragment_idx = 0
    
    # Find all fragment elements in the document
    for page in root.iter("page"):
        for fragment in page.iter("fragment"):
            fragments.append((fragment_idx, fragment))
            fragment_idx += 1
    
    return fragments


def _get_fragment_nodes_and_bonds(fragment_elem):
    """Extract node IDs and bonds from a fragment element.
    
    Returns:
        (set of node IDs, list of bond tuples (begin_id, end_id))
    """
    node_ids = set()
    bonds = []
    
    for child in fragment_elem:
        if child.tag == "n":
            node_id = child.get("id")
            if node_id:
                node_ids.add(node_id)
        elif child.tag == "b":
            begin = child.get("B")
            end = child.get("E")
            if begin and end:
                bonds.append((begin, end))
    
    return node_ids, bonds


def _create_fragment_document(full_doc: CDXMLDocument, node_ids: set) -> CDXMLDocument:
    """Create a new CDXMLDocument containing only the specified nodes and their bonds.
    
    Args:
        full_doc: The complete parsed document
        node_ids: Set of node IDs to include in the fragment
    
    Returns:
        A new CDXMLDocument containing only the fragment's elements
    """
    from dataclasses import replace, field
    
    # Create new document with same global settings
    frag_doc = CDXMLDocument(
        colors=full_doc.colors,
        fonts=full_doc.fonts,
        line_width=full_doc.line_width,
        bold_width=full_doc.bold_width,
        bond_length=full_doc.bond_length,
        bond_spacing=full_doc.bond_spacing,
        hash_spacing=full_doc.hash_spacing,
        margin_width=full_doc.margin_width,
        label_font=full_doc.label_font,
        label_size=full_doc.label_size,
        caption_size=full_doc.caption_size,
    )
    
    # Copy only the nodes that belong to this fragment
    for node_id in node_ids:
        if node_id in full_doc.nodes:
            frag_doc.nodes[node_id] = full_doc.nodes[node_id]
    
    # Copy only the bonds that connect nodes in this fragment
    for bond in full_doc.bonds:
        if bond.begin_id in node_ids and bond.end_id in node_ids:
            frag_doc.bonds.append(bond)
    
    # Calculate bounding box for this fragment
    if frag_doc.nodes:
        xs = [node.position[0] for node in frag_doc.nodes.values()]
        ys = [node.position[1] for node in frag_doc.nodes.values()]
        
        # Add some padding
        padding = 20.0
        min_x = min(xs) - padding
        min_y = min(ys) - padding
        max_x = max(xs) + padding
        max_y = max(ys) + padding
        
        frag_doc.bounding_box = (min_x, min_y, max_x, max_y)
    
    return frag_doc


def _apply_line_width_override(
    doc: CDXMLDocument,
    line_width: Optional[float],
) -> None:
    """Override document stroke sizing while preserving style ratios."""
    if line_width is None:
        return
    if line_width <= 0:
        raise ValueError("line_width must be greater than 0")

    original_line_width = doc.line_width
    scale_factor = line_width / original_line_width if original_line_width > 0 else 1.0

    doc.line_width = line_width
    doc.bold_width *= scale_factor
    doc.hash_spacing *= scale_factor
    doc.margin_width *= scale_factor


def convert_cdxml(
    input_path: str,
    output_dir: str = None,
    formats: list = None,
    scale: float = 2.0,
    padding: float = 15.0,
    bg_color: str = "white",
    png_scale: int = 2,
    per_fragment: bool = False,
    font_family: str = "Arial",
    line_width: Optional[float] = None,
    _rasterizer=None,
) -> dict:
    """
    Convert a CDXML file to SVG, PNG, and/or JPG.

    Args:
        input_path: Path to the input CDXML file.
        output_dir: Directory for output files (defaults to same as input).
        formats: List of output formats ('svg', 'png', 'jpg'). Defaults to all.
        scale: SVG rendering scale factor.
        padding: Padding around the diagram in points.
        bg_color: Background color for the SVG.
        png_scale: Scale factor for PNG/JPG rasterization.
        per_fragment: If True, create separate images for each fragment (molecule).
                     If False, create a single image of the entire document.
        line_width: Optional override for structure line width. When set,
            related bold/hash/margin stroke dimensions are scaled to match.
        _rasterizer: Optional pre-created SVGRasterizer instance for batch
            processing. If None, a new one is created per call.

    Returns:
        If per_fragment=False: Dict mapping format names to output file paths.
        If per_fragment=True: Dict mapping format names to lists of file paths,
                             one per fragment.
    """
    if formats is None:
        formats = ["svg", "png", "jpg"]

    input_path = Path(input_path)
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem

    # Parse the full document
    parser = CDXMLParser()
    full_doc = parser.parse(str(input_path))
    _apply_line_width_override(full_doc, line_width)

    if not per_fragment:
        # Original behavior: single image for entire document
        return _convert_single(
            full_doc, output_dir, stem, formats, 
            scale, padding, bg_color, png_scale, font_family, _rasterizer
        )
    
    # New behavior: separate image for each fragment
    fragments = _extract_fragment_elements(str(input_path))
    
    if not fragments:
        # No fragments found, fall back to single image
        return _convert_single(
            full_doc, output_dir, stem, formats,
            scale, padding, bg_color, png_scale,font_family, _rasterizer
        )
    
    results = {fmt: [] for fmt in formats}
    
    for frag_idx, frag_elem in fragments:
        node_ids, bonds = _get_fragment_nodes_and_bonds(frag_elem)
        
        if not node_ids:
            continue
        
        # Create a document for this fragment only
        frag_doc = _create_fragment_document(full_doc, node_ids)
        
        # Generate output filename
        frag_stem = f"{stem}_fragment_{frag_idx + 1}"
        
        # Convert this fragment
        frag_results = _convert_single(
            frag_doc, output_dir, frag_stem, formats,
            scale, padding, bg_color, png_scale,font_family, _rasterizer
        )
        
        # Collect results
        for fmt in formats:
            if fmt in frag_results:
                results[fmt].append(frag_results[fmt])
    
    return results


def _convert_single(
    doc: CDXMLDocument,
    output_dir: Path,
    stem: str,
    formats: list,
    scale: float,
    padding: float,
    bg_color: str,
    png_scale: int,
    font_family: str,
    _rasterizer=None,
) -> dict:
    """Convert a single document to the requested formats."""
    
    # Render SVG
    renderer = SVGRenderer(scale=scale, padding=padding, bg_color=bg_color, font_family=font_family)
    svg_content = renderer.render(doc)

    results = {}

    # Save SVG
    if "svg" in formats:
        svg_path = output_dir / f"{stem}.svg"
        svg_path.write_text(svg_content, encoding="utf-8")
        results["svg"] = str(svg_path)

    # Convert to PNG / JPG
    need_raster = "png" in formats or "jpg" in formats
    if need_raster and _HAS_PLAYWRIGHT:
        # High-quality path: rasterize the SVG via headless Chromium
        own_rasterizer = False
        try:
            if _rasterizer is None:
                _rasterizer = SVGRasterizer()
                _rasterizer.__enter__()
                own_rasterizer = True

            if "png" in formats:
                png_bytes = _rasterizer.rasterize(
                    svg_content, scale=png_scale, fmt="png")
                png_path = output_dir / f"{stem}.png"
                png_path.write_bytes(png_bytes)
                results["png"] = str(png_path)

            if "jpg" in formats:
                jpg_bytes = _rasterizer.rasterize(
                    svg_content, scale=png_scale, fmt="jpg", jpg_quality=95)
                jpg_path = output_dir / f"{stem}.jpg"
                jpg_path.write_bytes(jpg_bytes)
                results["jpg"] = str(jpg_path)
        finally:
            if own_rasterizer:
                _rasterizer.__exit__(None, None, None)

    elif need_raster:
        # Fallback: PIL-based direct rendering (lower quality)
        from .rasterizer import PILRenderer
        pil_renderer = PILRenderer(
            scale=scale * png_scale,
            padding=padding,
            bg_color=(255, 255, 255),
        )
        img = pil_renderer.render(doc)

        if "png" in formats:
            png_path = output_dir / f"{stem}.png"
            img.save(str(png_path), "PNG")
            results["png"] = str(png_path)

        if "jpg" in formats:
            if img.mode != "RGB":
                img = img.convert("RGB")
            jpg_path = output_dir / f"{stem}.jpg"
            img.save(str(jpg_path), "JPEG", quality=95)
            results["jpg"] = str(jpg_path)

    return results


def convert_cdxml_to_svg_string(
    input_path: str,
    scale: float = 2.0,
    padding: float = 15.0,
    bg_color: str = "white",
    line_width: Optional[float] = None,
) -> str:
    """
    Convert a CDXML file and return the SVG string directly.

    Args:
        input_path: Path to the input CDXML file.
        scale: SVG rendering scale factor.
        padding: Padding around the diagram in points.
        bg_color: Background color for the SVG.
        line_width: Optional override for structure line width.

    Returns:
        SVG markup string.
    """
    parser = CDXMLParser()
    doc = parser.parse(str(input_path))
    _apply_line_width_override(doc, line_width)
    renderer = SVGRenderer(scale=scale, padding=padding, bg_color=bg_color)
    return renderer.render(doc)


def convert_cdxml_fragments_to_svg_strings(
    input_path: str,
    scale: float = 2.0,
    padding: float = 15.0,
    bg_color: str = "white",
    line_width: Optional[float] = None,
) -> list:
    """
    Convert a CDXML file and return a list of SVG strings, one per fragment.

    Args:
        input_path: Path to the input CDXML file.
        scale: SVG rendering scale factor.
        padding: Padding around the diagram in points.
        bg_color: Background color for the SVG.
        line_width: Optional override for structure line width.

    Returns:
        List of SVG markup strings, one per fragment.
    """
    parser = CDXMLParser()
    full_doc = parser.parse(str(input_path))
    _apply_line_width_override(full_doc, line_width)
    
    fragments = _extract_fragment_elements(str(input_path))
    
    if not fragments:
        # No fragments, return single SVG
        renderer = SVGRenderer(scale=scale, padding=padding, bg_color=bg_color)
        return [renderer.render(full_doc)]
    
    svg_strings = []
    renderer = SVGRenderer(scale=scale, padding=padding, bg_color=bg_color)
    
    for frag_idx, frag_elem in fragments:
        node_ids, bonds = _get_fragment_nodes_and_bonds(frag_elem)
        
        if not node_ids:
            continue
        
        frag_doc = _create_fragment_document(full_doc, node_ids)
        svg_content = renderer.render(frag_doc)
        svg_strings.append(svg_content)
    
    return svg_strings
