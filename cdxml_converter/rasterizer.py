"""
PIL-based rasterizer - Renders CDXMLDocument directly to PNG/JPG using Pillow.

This avoids the need for cairo or any external SVG renderer.
"""

import math
from PIL import Image, ImageDraw, ImageFont

from .parser import CDXMLDocument, Node, Bond, Arrow, TextLabel, Color, ELEMENT_MAP


def _hex_to_tuple(color: Color):
    """Convert Color to RGB tuple."""
    return (int(color.r * 255), int(color.g * 255), int(color.b * 255))


def _get_color(color_idx: int, doc: CDXMLDocument):
    """Get RGB tuple for a CDXML color index."""
    color = doc.get_color(color_idx)
    return _hex_to_tuple(color)


def _try_get_font(size: int, bold: bool = False):
    """Try to get the best available font."""
    font_names = []
    if bold:
        font_names = ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"]
    else:
        font_names = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]

    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            pass

    # Fallback to default
    try:
        return ImageFont.load_default()
    except Exception:
        return None


class PILRenderer:
    """Renders CDXMLDocument directly to a PIL Image."""

    def __init__(self, scale: float = 4.0, padding: float = 15.0,
                 bg_color=(255, 255, 255)):
        self.scale = scale
        self.padding = padding
        self.bg_color = bg_color

    def render(self, doc: CDXMLDocument) -> Image.Image:
        """Render the document to a PIL Image."""
        bb = doc.bounding_box
        x1, y1, x2, y2 = bb

        # Calculate image dimensions
        w = x2 - x1 + 2 * self.padding
        h = y2 - y1 + 2 * self.padding
        img_w = int(w * self.scale)
        img_h = int(h * self.scale)

        # Create image
        img = Image.new("RGB", (img_w, img_h), self.bg_color)
        draw = ImageDraw.Draw(img)

        # Offset to shift coordinates
        ox = -x1 + self.padding
        oy = -y1 + self.padding

        # Render bonds
        for bond in doc.bonds:
            self._draw_bond(draw, bond, doc, ox, oy)

        # Render node labels (with background)
        for node_id, node in doc.nodes.items():
            self._draw_node_label(draw, node, doc, ox, oy)

        # Render arrows
        for arrow in doc.arrows:
            self._draw_arrow(draw, arrow, doc, ox, oy)

        # Render standalone text
        for text in doc.texts:
            self._draw_text(draw, text, doc, ox, oy)

        return img

    def _tx(self, x, ox):
        """Transform X coordinate to pixel space."""
        return (x + ox) * self.scale

    def _ty(self, y, oy):
        """Transform Y coordinate to pixel space."""
        return (y + oy) * self.scale

    def _node_has_visible_label(self, node: Node) -> bool:
        if node.has_label_text:
            return True
        if node.element != 6:
            return True
        if node.node_type in ("Fragment", "Nickname", "GenericNickname",
                              "Unspecified"):
            return True
        return False

    def _get_label_text(self, node: Node) -> str:
        if node.label:
            return node.label.get_text()
        if node.element != 6:
            return ELEMENT_MAP.get(node.element, f"?{node.element}")
        return ""

    def _compute_bond_endpoints(self, bond: Bond, doc: CDXMLDocument):
        """Compute bond endpoints, shortened at labeled atoms."""
        begin_node = doc.nodes.get(bond.begin_id)
        end_node = doc.nodes.get(bond.end_id)

        if not begin_node or not end_node:
            return None

        x1, y1 = begin_node.position
        x2, y2 = end_node.position

        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.001:
            return None

        ux = dx / length
        uy = dy / length

        start_offset = 0
        end_offset = 0

        if self._node_has_visible_label(begin_node):
            if begin_node.label and begin_node.label.bounding_box:
                bb = begin_node.label.bounding_box
                hw = (bb[2] - bb[0]) / 2 + 1
                hh = (bb[3] - bb[1]) / 2 + 1
                if abs(ux) > abs(uy):
                    start_offset = hw
                else:
                    start_offset = hh
            else:
                text = self._get_label_text(begin_node)
                start_offset = len(text) * doc.label_size * 0.3 + 1

        if self._node_has_visible_label(end_node):
            if end_node.label and end_node.label.bounding_box:
                bb = end_node.label.bounding_box
                hw = (bb[2] - bb[0]) / 2 + 1
                hh = (bb[3] - bb[1]) / 2 + 1
                if abs(ux) > abs(uy):
                    end_offset = hw
                else:
                    end_offset = hh
            else:
                text = self._get_label_text(end_node)
                end_offset = len(text) * doc.label_size * 0.3 + 1

        ax = x1 + ux * start_offset
        ay = y1 + uy * start_offset
        bx = x2 - ux * end_offset
        by = y2 - uy * end_offset

        return ax, ay, bx, by, ux, uy, length

    def _draw_bond(self, draw, bond: Bond, doc: CDXMLDocument, ox, oy):
        """Draw a bond."""
        result = self._compute_bond_endpoints(bond, doc)
        if not result:
            return

        x1, y1, x2, y2, ux, uy, length = result
        color = _get_color(bond.color_idx, doc)
        lw = max(1, int(doc.line_width * self.scale))

        if bond.display == "WedgeBegin":
            self._draw_wedge_bond(draw, x1, y1, x2, y2, ux, uy, color, doc, ox, oy)
        elif bond.display == "WedgedHashBegin":
            self._draw_hash_bond(draw, x1, y1, x2, y2, ux, uy, color, doc, ox, oy)
        elif bond.display == "Wavy":
            self._draw_wavy_bond(draw, x1, y1, x2, y2, color, doc, ox, oy)
        elif bond.display == "Dash":
            # Draw dashed line by segments
            self._draw_dashed_line(draw, x1, y1, x2, y2, color, lw, ox, oy)
        elif bond.display == "Bold":
            bold_w = max(2, int(doc.bold_width * self.scale))
            draw.line(
                [(self._tx(x1, ox), self._ty(y1, oy)),
                 (self._tx(x2, ox), self._ty(y2, oy))],
                fill=color, width=bold_w
            )
        elif bond.order == 1:
            draw.line(
                [(self._tx(x1, ox), self._ty(y1, oy)),
                 (self._tx(x2, ox), self._ty(y2, oy))],
                fill=color, width=lw
            )
        elif bond.order == 2:
            self._draw_double_bond(draw, x1, y1, x2, y2, ux, uy, color, lw, doc, ox, oy)
        elif bond.order == 3:
            self._draw_triple_bond(draw, x1, y1, x2, y2, ux, uy, color, lw, doc, ox, oy)
        else:
            draw.line(
                [(self._tx(x1, ox), self._ty(y1, oy)),
                 (self._tx(x2, ox), self._ty(y2, oy))],
                fill=color, width=lw
            )

    def _draw_double_bond(self, draw, x1, y1, x2, y2, ux, uy, color, lw,
                          doc, ox, oy):
        px = -uy
        py = ux
        spacing = doc.bond_length * doc.bond_spacing / 100 / 2

        for sign in (1, -1):
            draw.line(
                [(self._tx(x1 + px * spacing * sign, ox),
                  self._ty(y1 + py * spacing * sign, oy)),
                 (self._tx(x2 + px * spacing * sign, ox),
                  self._ty(y2 + py * spacing * sign, oy))],
                fill=color, width=lw
            )

    def _draw_triple_bond(self, draw, x1, y1, x2, y2, ux, uy, color, lw,
                          doc, ox, oy):
        px = -uy
        py = ux
        spacing = doc.bond_length * doc.bond_spacing / 100 / 2

        # Center line
        draw.line(
            [(self._tx(x1, ox), self._ty(y1, oy)),
             (self._tx(x2, ox), self._ty(y2, oy))],
            fill=color, width=lw
        )
        for sign in (1, -1):
            draw.line(
                [(self._tx(x1 + px * spacing * sign, ox),
                  self._ty(y1 + py * spacing * sign, oy)),
                 (self._tx(x2 + px * spacing * sign, ox),
                  self._ty(y2 + py * spacing * sign, oy))],
                fill=color, width=lw
            )

    def _draw_wedge_bond(self, draw, x1, y1, x2, y2, ux, uy, color,
                         doc, ox, oy):
        px = -uy
        py = ux
        ww = doc.bold_width * 1.5

        points = [
            (self._tx(x1, ox), self._ty(y1, oy)),
            (self._tx(x2 + px * ww, ox), self._ty(y2 + py * ww, oy)),
            (self._tx(x2 - px * ww, ox), self._ty(y2 - py * ww, oy)),
        ]
        draw.polygon(points, fill=color, outline=color)

    def _draw_hash_bond(self, draw, x1, y1, x2, y2, ux, uy, color,
                        doc, ox, oy):
        px = -uy
        py = ux
        ww = doc.bold_width * 1.5
        lw = max(1, int(doc.line_width * self.scale))

        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        num_lines = max(3, int(length / doc.hash_spacing))

        for i in range(num_lines + 1):
            t = i / num_lines
            mx = x1 + dx * t
            my = y1 + dy * t
            w = ww * t

            draw.line(
                [(self._tx(mx + px * w, ox), self._ty(my + py * w, oy)),
                 (self._tx(mx - px * w, ox), self._ty(my - py * w, oy))],
                fill=color, width=lw
            )

    def _draw_wavy_bond(self, draw, x1, y1, x2, y2, color, doc, ox, oy):
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.001:
            return

        ux_d = dx / length
        uy_d = dy / length
        px = -uy_d
        py = ux_d

        amplitude = doc.bold_width * 0.8
        num_waves = max(3, int(length / 3))
        lw = max(1, int(doc.line_width * self.scale))

        points = []
        for i in range(num_waves * 4 + 1):
            t = i / (num_waves * 4)
            angle = t * num_waves * 2 * math.pi
            offset = amplitude * math.sin(angle)

            cx = x1 + dx * t + px * offset
            cy = y1 + dy * t + py * offset
            points.append((self._tx(cx, ox), self._ty(cy, oy)))

        if len(points) >= 2:
            draw.line(points, fill=color, width=lw)

    def _draw_dashed_line(self, draw, x1, y1, x2, y2, color, lw, ox, oy):
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.001:
            return

        dash_len = 2.0
        gap_len = 2.0
        t = 0
        drawing = True
        while t < length:
            seg_len = dash_len if drawing else gap_len
            t_end = min(t + seg_len, length)

            if drawing:
                sx = x1 + dx * (t / length)
                sy = y1 + dy * (t / length)
                ex = x1 + dx * (t_end / length)
                ey = y1 + dy * (t_end / length)
                draw.line(
                    [(self._tx(sx, ox), self._ty(sy, oy)),
                     (self._tx(ex, ox), self._ty(ey, oy))],
                    fill=color, width=lw
                )

            t = t_end
            drawing = not drawing

    def _draw_node_label(self, draw, node: Node, doc: CDXMLDocument, ox, oy):
        """Draw label for an atom node."""
        if not self._node_has_visible_label(node):
            return

        text = self._get_label_text(node)
        if not text:
            return

        x, y = node.position

        # Determine color
        if node.label and node.label.spans:
            color = _get_color(node.label.spans[0].color_idx, doc)
        elif node.color_idx:
            color = _get_color(node.color_idx, doc)
        else:
            color = (0, 0, 0)

        # Font size
        font_size = int(doc.label_size * self.scale)
        if node.label and node.label.spans:
            font_size = int(node.label.spans[0].size * self.scale)

        bold = False
        if node.label and node.label.spans:
            bold = node.label.spans[0].is_bold

        font = _try_get_font(font_size, bold)

        # Use label bounding box for background
        if node.label and node.label.bounding_box:
            bb = node.label.bounding_box
            pad = 1.5
            draw.rectangle(
                [(self._tx(bb[0] - pad, ox), self._ty(bb[1] - pad, oy)),
                 (self._tx(bb[2] + pad, ox), self._ty(bb[3] + pad, oy))],
                fill=self.bg_color
            )

        # Get position from label or node
        if node.label:
            tx, ty = node.label.position
        else:
            tx, ty = x, y

        # Draw text
        px = self._tx(tx, ox)
        py = self._ty(ty, oy)

        if font:
            draw.text((px, py), text, fill=color, font=font)
        else:
            draw.text((px, py), text, fill=color)

    def _draw_arrow(self, draw, arrow: Arrow, doc: CDXMLDocument, ox, oy):
        """Draw a reaction arrow."""
        tx_s, ty_s = arrow.tail
        hx, hy = arrow.head
        lw = max(1, int(doc.line_width * 1.5 * self.scale))

        # Arrow shaft
        draw.line(
            [(self._tx(tx_s, ox), self._ty(ty_s, oy)),
             (self._tx(hx, ox), self._ty(hy, oy))],
            fill=(0, 0, 0), width=lw
        )

        # Arrowhead
        dx = hx - tx_s
        dy = hy - ty_s
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.001:
            return

        ux = dx / length
        uy = dy / length
        px = -uy
        py = ux

        head_len = arrow.head_size / 1000 * doc.bond_length * 0.6
        head_width = head_len * 0.35

        tip = (self._tx(hx, ox), self._ty(hy, oy))
        left = (self._tx(hx - ux * head_len + px * head_width, ox),
                self._ty(hy - uy * head_len + py * head_width, oy))
        right = (self._tx(hx - ux * head_len - px * head_width, ox),
                 self._ty(hy - uy * head_len - py * head_width, oy))

        draw.polygon([tip, left, right], fill=(0, 0, 0), outline=(0, 0, 0))

    def _draw_text(self, draw, text_label: TextLabel, doc: CDXMLDocument,
                   ox, oy):
        """Draw standalone text annotation."""
        if not text_label.spans:
            return

        full_text = text_label.get_text()
        if not full_text.strip():
            return

        tx, ty = text_label.position

        # Use first span's properties for font
        span = text_label.spans[0]
        font_size = int(span.size * self.scale)
        color = _get_color(span.color_idx, doc)
        bold = span.is_bold
        font = _try_get_font(font_size, bold)

        px = self._tx(tx, ox)
        py = self._ty(ty, oy)

        # Handle multi-line text
        text_lines = full_text.split("\n")
        line_height = font_size * 1.3

        for i, line in enumerate(text_lines):
            line_y = py + i * line_height
            if font:
                # Handle justification
                just = text_label.caption_justification or text_label.justification
                if just == "Center":
                    bbox = font.getbbox(line)
                    tw = bbox[2] - bbox[0]
                    draw.text((px - tw / 2, line_y), line, fill=color, font=font)
                elif just == "Right":
                    bbox = font.getbbox(line)
                    tw = bbox[2] - bbox[0]
                    draw.text((px - tw, line_y), line, fill=color, font=font)
                else:
                    draw.text((px, line_y), line, fill=color, font=font)
            else:
                draw.text((px, line_y), line, fill=color)
