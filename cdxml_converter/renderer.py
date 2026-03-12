"""
SVG Renderer - Converts a parsed CDXMLDocument into SVG markup.

Uses ChemDraw-style rendering:
- Bonds as filled hexagonal paths with miter joins at atoms
- Double bonds: primary filled path + shortened inner line
- Triple bonds: primary filled path + two shortened inner lines
- Wedge bonds (stereochemistry)
- Hash/dashed wedge bonds
- Wavy bonds
- Atom labels with white background rectangles
- Reaction arrows with arrowheads
- Text annotations (multi-line, styled)
- Colored elements
- Graphic elements (rectangles)
"""

import math
import re
from collections import defaultdict
from typing import Optional

from .parser import (
    CDXMLDocument, Node, Bond, Arrow, TextLabel, TextSpan,
    Color, Graphic, ELEMENT_MAP,
)


class SVGRenderer:
    """Renders a CDXMLDocument to SVG."""

    def __init__(self, scale: float = 2.0, padding: float = 15.0,
                 bg_color: str = "white",  font_family: str = "Arial"):
        self.scale = scale
        self.padding = padding
        self.bg_color = bg_color
        self.font_family = font_family
       

    def render(self, doc: CDXMLDocument) -> str:
        """Render the document to an SVG string."""
        bb = doc.bounding_box
        x1, y1, x2, y2 = bb

        # 5% padding on all sides based on content dimensions
        content_w = x2 - x1
        content_h = y2 - y1
        pad_x = content_w * 0.05
        pad_y = content_h * 0.05

        width = (content_w + 2 * pad_x) * self.scale
        height = (content_h + 2 * pad_y) * self.scale

        vb_x = x1 - pad_x
        vb_y = y1 - pad_y
        vb_w = content_w + 2 * pad_x
        vb_h = content_h + 2 * pad_y

        # Pre-compute adjacency and miter points
        adjacency = self._build_adjacency(doc)
        miter_map = self._compute_all_miter_points(doc, adjacency)

        # Build SVG
        lines = []
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width:.1f}" height="{height:.1f}" '
            f'viewBox="{vb_x:.2f} {vb_y:.2f} {vb_w:.2f} {vb_h:.2f}" '
            f'preserveAspectRatio="xMidYMid">'
        )

        # Background
        if self.bg_color:
            lines.append(
                f'  <rect x="{vb_x:.2f}" y="{vb_y:.2f}" '
                f'width="{vb_w:.2f}" height="{vb_h:.2f}" '
                f'fill="{self.bg_color}"/>'
            )

        # Style definitions
        lines.append('  <defs/>')

        # Render graphics (rectangles) first (background)
        for g in doc.graphics:
            if g.graphic_type == "Rectangle":
                lines.extend(self._render_graphic(g, doc))

        # Render bonds using miter-joined paths
        for bond in doc.bonds:
            lines.extend(self._render_bond(bond, doc, miter_map, adjacency))

        # Render bond crossing gaps (white bands on back bonds,
        # then re-render front bonds on top)
        crossings = self._find_crossings(doc)
        for back_bond, front_bond, cross_point in crossings:
            lines.extend(self._render_crossing_gap(
                cross_point, front_bond, back_bond, doc))
            lines.extend(
                self._render_bond(front_bond, doc, miter_map, adjacency))

        # Render atom labels (on top of bonds)
        for node_id, node in doc.nodes.items():
            label_lines = self._render_node_label(node, doc)
            if label_lines:
                lines.extend(label_lines)

        # Render arrows
        for arrow in doc.arrows:
            lines.extend(self._render_arrow(arrow, doc))

        # Render standalone text
        for text in doc.texts:
            lines.extend(self._render_text(text, doc))

        lines.append('</svg>')
        return '\n'.join(lines)

    # ------------------------------------------------------------------ #
    #  Color helpers
    # ------------------------------------------------------------------ #

    def _get_color_str(self, color_idx: int, doc: CDXMLDocument) -> str:
        """Get color string for a color index."""
        color = doc.get_color(color_idx)
        if color.r == 0 and color.g == 0 and color.b == 0:
            return "rgb(0, 0, 0)"
        if color.r == 1 and color.g == 1 and color.b == 1:
            return "rgb(255, 255, 255)"
        return color.to_rgb()

    # ------------------------------------------------------------------ #
    #  Label bounding-box clipping
    # ------------------------------------------------------------------ #

    def _clip_bond_endpoint(self, node: 'Node', other_x: float,
                            other_y: float, doc: CDXMLDocument,
                            pad: float = 1.0) -> tuple:
        """Clip a bond endpoint to the edge of the node's label bounding box.

        If the node has a visible label with a bounding box, return the
        intersection of the line (node→other) with the padded bounding box.
        Otherwise return the node position unchanged.
        """
        if not self._node_has_visible_label(node):
            return node.position
        if not node.label or not node.label.bounding_box:
            return node.position

        nx, ny = node.position
        bb = node.label.bounding_box
        # bb = (left, top, right, bottom)
        left = bb[0] - pad
        top = bb[1] - pad
        right = bb[2] + pad
        bottom = bb[3] + pad

        # Direction from node toward the other end
        dx = other_x - nx
        dy = other_y - ny
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-10:
            return node.position

        # Find the parametric t where the ray from node exits the bbox
        # We solve for each edge and take the smallest positive t
        t_min = float('inf')
        if abs(dx) > 1e-10:
            # Left edge
            t = (left - nx) / dx
            if t > 0:
                y_at = ny + t * dy
                if top <= y_at <= bottom:
                    t_min = min(t_min, t)
            # Right edge
            t = (right - nx) / dx
            if t > 0:
                y_at = ny + t * dy
                if top <= y_at <= bottom:
                    t_min = min(t_min, t)
        if abs(dy) > 1e-10:
            # Top edge
            t = (top - ny) / dy
            if t > 0:
                x_at = nx + t * dx
                if left <= x_at <= right:
                    t_min = min(t_min, t)
            # Bottom edge
            t = (bottom - ny) / dy
            if t > 0:
                x_at = nx + t * dx
                if left <= x_at <= right:
                    t_min = min(t_min, t)

        if t_min < float('inf'):
            return (nx + t_min * dx, ny + t_min * dy)
        return node.position

    # ------------------------------------------------------------------ #
    #  Adjacency & miter computation
    # ------------------------------------------------------------------ #

    def _build_adjacency(self, doc: CDXMLDocument) -> dict:
        """Build adjacency map: {node_id: [(bond, other_node_id), ...]}."""
        adj = defaultdict(list)
        for bond in doc.bonds:
            adj[bond.begin_id].append((bond, bond.end_id))
            adj[bond.end_id].append((bond, bond.begin_id))
        return adj

    def _compute_all_miter_points(self, doc: CDXMLDocument,
                                   adjacency: dict) -> dict:
        """Compute miter join points at every node for every bond.

        Returns {node_id: {bond_id: (left_miter, right_miter)}} where
        left/right are defined looking OUTWARD from the node along the bond.

        The miter point between two adjacent bonds is computed by
        intersecting the offset edge lines (offset by LineWidth/2) of the
        two bonds.  This produces seamless joins at vertices, matching
        ChemDraw's rendering.
        """
        miter_map = {}
        half_lw = doc.line_width / 2

        for node_id, bonds_at_node in adjacency.items():
            node = doc.nodes.get(node_id)
            if not node:
                continue
            nx, ny = node.position

            if len(bonds_at_node) == 0:
                miter_map[node_id] = {}
                continue

            # Collect valid bond directions from this node
            bond_data = []
            for bond, other_id in bonds_at_node:
                other = doc.nodes.get(other_id)
                if not other:
                    continue
                ox, oy = other.position
                dx, dy = ox - nx, oy - ny
                length = math.sqrt(dx * dx + dy * dy)
                if length < 1e-10:
                    continue
                ux, uy = dx / length, dy / length
                angle = math.atan2(dy, dx)
                bond_data.append((angle, bond, ux, uy))

            if len(bond_data) == 0:
                miter_map[node_id] = {}
                continue

            if len(bond_data) == 1:
                # Terminal atom: simple perpendicular offsets
                _, bond, ux, uy = bond_data[0]
                lx = nx + (-uy) * half_lw
                ly = ny + ux * half_lw
                rx = nx + uy * half_lw
                ry = ny + (-ux) * half_lw
                miter_map[node_id] = {bond.id: ((lx, ly), (rx, ry))}
                continue

            # Sort bonds by outward angle (CCW order)
            bond_data.sort(key=lambda x: x[0])
            n = len(bond_data)

            # Compute miter point between each consecutive pair
            # miter_points[i] = intersection between right edge of bond i
            #                    and left edge of bond (i+1) % n
            miter_points = []
            for i in range(n):
                _, b1, u1x, u1y = bond_data[i]
                _, b2, u2x, u2y = bond_data[(i + 1) % n]

                # Right perpendicular of bond 1 (CW rotation): (u1y, -u1x)
                r1x, r1y = u1y, -u1x
                # Left perpendicular of bond 2 (CCW rotation): (-u2y, u2x)
                l2x, l2y = -u2y, u2x

                # Points on the offset edge lines
                p1x = nx + r1x * half_lw
                p1y = ny + r1y * half_lw
                p2x = nx + l2x * half_lw
                p2y = ny + l2y * half_lw

                # Intersect: p1 + t*u1 == p2 + s*u2
                det = u1x * u2y - u1y * u2x
                if abs(det) < 1e-10:
                    # Nearly parallel: use midpoint
                    mx = (p1x + p2x) / 2
                    my = (p1y + p2y) / 2
                else:
                    dpx = p2x - p1x
                    dpy = p2y - p1y
                    t = (dpx * u2y - dpy * u2x) / det
                    mx = p1x + t * u1x
                    my = p1y + t * u1y

                miter_points.append((mx, my))

            # Assign left/right miter to each bond at this node
            node_miters = {}
            for i, (angle, bond, ux, uy) in enumerate(bond_data):
                # Right miter = between bond i and bond (i+1)
                right_miter = miter_points[i]
                # Left miter = between bond (i-1) and bond i
                left_miter = miter_points[(i - 1) % n]
                node_miters[bond.id] = (left_miter, right_miter)

            miter_map[node_id] = node_miters

        return miter_map

    # ------------------------------------------------------------------ #
    #  Bond rendering
    # ------------------------------------------------------------------ #

    def _render_bond(self, bond: Bond, doc: CDXMLDocument,
                     miter_map: dict, adjacency: dict) -> list:
        """Render a single bond using ChemDraw-style filled paths."""
        begin_node = doc.nodes.get(bond.begin_id)
        end_node = doc.nodes.get(bond.end_id)
        if not begin_node or not end_node:
            return []

        # Original node positions (for miter lookup)
        orig_bx, orig_by = begin_node.position
        orig_ex, orig_ey = end_node.position

        # Clip endpoints at label bounding boxes
        bx, by = self._clip_bond_endpoint(begin_node, orig_ex, orig_ey, doc)
        ex, ey = self._clip_bond_endpoint(end_node, orig_bx, orig_by, doc)

        dx = ex - bx
        dy = ey - by
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-10:
            return []
        ux, uy = dx / length, dy / length

        color = self._get_color_str(bond.color_idx, doc)

        # Special display types
        if bond.display == "WedgeBegin":
            return self._render_wedge_bond(bx, by, ex, ey, ux, uy, color, doc)
        elif bond.display == "WedgedHashBegin":
            return self._render_hash_bond(bx, by, ex, ey, ux, uy, color, doc)
        elif bond.display == "Wavy":
            return self._render_wavy_bond(bx, by, ex, ey, color, doc)

        # Get miter points
        begin_miters = miter_map.get(bond.begin_id, {}).get(bond.id)
        end_miters = miter_map.get(bond.end_id, {}).get(bond.id)

        half_lw = doc.line_width / 2

        if not begin_miters or not end_miters:
            # Fallback: simple line
            return [
                f'  <line x1="{bx:.2f}" y1="{by:.2f}" '
                f'x2="{ex:.2f}" y2="{ey:.2f}" '
                f'stroke="{color}" stroke-width="{doc.line_width:.2f}"/>'
            ]

        # Check if begin endpoint was clipped
        b_clipped = (abs(bx - orig_bx) > 0.01 or abs(by - orig_by) > 0.01)
        e_clipped = (abs(ex - orig_ex) > 0.01 or abs(ey - orig_ey) > 0.01)

        # Path points at begin node (looking outward from begin toward end)
        if b_clipped:
            # Use simple perpendicular wings at clipped position
            left_b = (bx + (-uy) * half_lw, by + ux * half_lw)
            right_b = (bx + uy * half_lw, by + (-ux) * half_lw)
        else:
            left_b = begin_miters[0]
            right_b = begin_miters[1]

        # At end node: swap because outward from end is reversed
        if e_clipped:
            right_e = (ex + (-uy) * half_lw, ey + ux * half_lw)
            left_e = (ex + uy * half_lw, ey + (-ux) * half_lw)
        else:
            right_e = end_miters[0]
            left_e = end_miters[1]

        lines = []

        if bond.display == "Dash":
            lines.append(
                f'  <line x1="{bx:.2f}" y1="{by:.2f}" '
                f'x2="{ex:.2f}" y2="{ey:.2f}" '
                f'stroke="{color}" stroke-width="{doc.line_width:.2f}" '
                f'stroke-dasharray="2,2"/>'
            )
            return lines

        if bond.display == "Bold":
            half_bw = doc.bold_width / 2
            lb = (bx + (-uy) * half_bw, by + ux * half_bw)
            rb = (bx + uy * half_bw, by + (-ux) * half_bw)
            le = (ex + (-uy) * half_bw, ey + ux * half_bw)
            re = (ex + uy * half_bw, ey + (-ux) * half_bw)
            if b_clipped:
                bp = f'M{lb[0]:.2f} {lb[1]:.2f} L{rb[0]:.2f} {rb[1]:.2f} '
            else:
                bp = (f'M{lb[0]:.2f} {lb[1]:.2f} '
                      f'L{bx:.2f} {by:.2f} L{rb[0]:.2f} {rb[1]:.2f} ')
            if e_clipped:
                ep = f'L{re[0]:.2f} {re[1]:.2f} L{le[0]:.2f} {le[1]:.2f} Z'
            else:
                ep = (f'L{re[0]:.2f} {re[1]:.2f} '
                      f'L{ex:.2f} {ey:.2f} L{le[0]:.2f} {le[1]:.2f} Z')
            path = bp + ep
            lines.append(
                f'  <path d="{path}" stroke="{color}" stroke-width="0" '
                f'fill-rule="evenodd" stroke-linejoin="miter" fill="{color}" '
                f'shape-rendering="geometricPrecision" />'
            )
            return lines

        # Check if double bond is in a ring (has BondCircularOrdering)
        is_ring_double = (bond.order == 2 and
                          len(bond.bond_circular_ordering) > 0)
        is_centered_double = (bond.order == 2 and not is_ring_double)

        if is_centered_double:
            # Centered double bond: two symmetric lines fanning from vertex
            return self._render_centered_double_bond(
                bond, doc, adjacency, ux, uy, bx, by, ex, ey,
                left_b, right_b, left_e, right_e, color)

        # Primary filled hexagonal path (single, ring-double, triple)
        # For single bonds with any clipped endpoint, use a simple line
        # instead of hexagonal path to avoid visual artifacts
        if bond.order == 1 and (b_clipped or e_clipped):
            lines.append(
                f'  <line x1="{bx:.2f}" y1="{by:.2f}" '
                f'x2="{ex:.2f}" y2="{ey:.2f}" '
                f'stroke="{color}" stroke-width="{doc.line_width:.2f}"/>'
            )
            return lines

        # At clipped endpoints, skip the center vertex for a flat cut
        if b_clipped:
            begin_part = f'M{left_b[0]:.2f} {left_b[1]:.2f} L{right_b[0]:.2f} {right_b[1]:.2f} '
        else:
            begin_part = (
                f'M{left_b[0]:.2f} {left_b[1]:.2f} '
                f'L{bx:.2f} {by:.2f} '
                f'L{right_b[0]:.2f} {right_b[1]:.2f} '
            )
        if e_clipped:
            end_part = f'L{right_e[0]:.2f} {right_e[1]:.2f} L{left_e[0]:.2f} {left_e[1]:.2f} Z'
        else:
            end_part = (
                f'L{right_e[0]:.2f} {right_e[1]:.2f} '
                f'L{ex:.2f} {ey:.2f} '
                f'L{left_e[0]:.2f} {left_e[1]:.2f} Z'
            )
        path = begin_part + end_part
        lines.append(
            f'  <path d="{path}" stroke="{color}" stroke-width="0" '
            f'fill-rule="evenodd" stroke-linejoin="miter" fill="{color}" '
            f'shape-rendering="geometricPrecision" />'
        )

        if bond.order == 2:
            inner = self._compute_inner_double_bond_line(
                bond, doc, adjacency, ux, uy, bx, by, ex, ey)
            if inner:
                ix1, iy1, ix2, iy2 = inner
                lines.append(
                    f'  <line x1="{ix1:.2f}" y1="{iy1:.2f}" '
                    f'x2="{ix2:.2f}" y2="{iy2:.2f}" '
                    f'stroke="{color}" stroke-width="{doc.line_width:.2f}" '
                    f'shape-rendering="auto" />'
                )

        elif bond.order == 3:
            inner_lines = self._compute_inner_triple_bond_lines(
                bond, doc, adjacency, ux, uy, bx, by, ex, ey)
            for ix1, iy1, ix2, iy2 in inner_lines:
                lines.append(
                    f'  <line x1="{ix1:.2f}" y1="{iy1:.2f}" '
                    f'x2="{ix2:.2f}" y2="{iy2:.2f}" '
                    f'stroke="{color}" stroke-width="{doc.line_width:.2f}" '
                    f'shape-rendering="auto" />'
                )

        return lines

    # ------------------------------------------------------------------ #
    #  Centered double bond (non-ring)
    # ------------------------------------------------------------------ #

    def _render_centered_double_bond(self, bond, doc, adjacency,
                                      ux, uy, bx, by, ex, ey,
                                      left_b, right_b, left_e, right_e,
                                      color):
        """Render a non-ring double bond as two parallel lines with fan-out.

        Both lines maintain constant perpendicular distance from the bond
        center.  At vertices with adjacent bonds, only the along-bond
        starting position is shifted so that the start point lies on
        the adjacent bond's centerline — this creates the fan-out while
        keeping the lines strictly parallel.
        """
        half_spacing = doc.bond_spacing / 100 * doc.bond_length / 2
        lw = doc.line_width
        px, py = -uy, ux  # Left perpendicular (CCW)
        lines = []

        for sign in [1, -1]:
            s = half_spacing * sign

            # Default: simple perpendicular offset
            sx = bx + px * s
            sy = by + py * s
            fx = ex + px * s
            fy = ey + py * s

            # Fan-out at begin vertex — shift along bond direction only
            delta_b = self._compute_fanout_along_bond(
                bond.begin_id, bond, ux, uy, px, py, s, doc, adjacency)
            sx += ux * delta_b
            sy += uy * delta_b

            # Fan-out at end vertex
            delta_e = self._compute_fanout_along_bond(
                bond.end_id, bond, ux, uy, px, py, s, doc, adjacency)
            fx += ux * delta_e
            fy += uy * delta_e

            lines.append(
                f'  <line x1="{sx:.2f}" y1="{sy:.2f}" '
                f'x2="{fx:.2f}" y2="{fy:.2f}" '
                f'stroke="{color}" stroke-width="{lw:.2f}" '
                f'shape-rendering="auto" />'
            )

        return lines

    def _compute_fanout_along_bond(self, node_id, bond, ux, uy, px, py,
                                    s, doc, adjacency):
        """Compute along-bond shift for parallel fan-out at a vertex.

        Finds the adjacent bond on the same side as perpendicular offset s,
        then computes delta_t such that the point
            vertex + p*s + u*delta_t
        lies on the adjacent bond's centerline.

        Formula:  delta_t = -s * dot(adj, u) / cross(adj, u)

        This keeps perpendicular offset constant at s, only shifting
        along the bond direction.
        """
        adj_bonds = adjacency.get(node_id, [])
        if len(adj_bonds) <= 1:
            return 0

        node = doc.nodes.get(node_id)
        if not node:
            return 0

        nx, ny = node.position

        # Find adjacent bond on the same side as offset s
        best_adj = None
        best_side = 0

        for b, adj_other_id in adj_bonds:
            if b.id == bond.id:
                continue
            adj_node = doc.nodes.get(adj_other_id)
            if not adj_node:
                continue
            ax, ay = adj_node.position
            adx, ady = ax - nx, ay - ny
            adj_len = math.sqrt(adx * adx + ady * ady)
            if adj_len < 1e-10:
                continue
            adj_ux, adj_uy = adx / adj_len, ady / adj_len

            # Projection onto perpendicular = which side this adj is on
            side = adj_ux * px + adj_uy * py

            # Select the adjacent bond most aligned with our offset side
            if s > 0 and side > best_side:
                best_side = side
                best_adj = (adj_ux, adj_uy)
            elif s < 0 and side < best_side:
                best_side = side
                best_adj = (adj_ux, adj_uy)

        if best_adj is None:
            return 0

        adj_ux, adj_uy = best_adj

        # cross(adj, u)
        cross = adj_ux * uy - adj_uy * ux
        if abs(cross) < 1e-10:
            return 0

        # dot(adj, u)
        dot_au = adj_ux * ux + adj_uy * uy

        # Along-bond shift to place point on adjacent bond's centerline
        return -s * dot_au / cross

    # ------------------------------------------------------------------ #
    #  Double bond inner line
    # ------------------------------------------------------------------ #

    def _compute_inner_double_bond_line(self, bond, doc, adjacency,
                                         ux, uy, bx, by, ex, ey):
        """Compute the inner line of a double bond.

        The inner line is offset perpendicular to the bond toward the side
        with more neighboring atoms (ring interior), then shortened at each
        end so it doesn't protrude past adjacent bonds.

        Shortening formula: offset * tan(angle_from_perp_to_adjacent_bond)
        """
        offset = doc.bond_spacing / 100 * doc.bond_length

        # Determine which side
        side = self._determine_double_bond_side(bond, doc, adjacency,
                                                 ux, uy, bx, by, ex, ey)
        if side == "right":
            px, py = uy, -ux   # CW perpendicular
        else:
            px, py = -uy, ux   # CCW perpendicular

        # Offset start/end
        isx = bx + px * offset
        isy = by + py * offset
        iex = ex + px * offset
        iey = ey + py * offset

        # Shorten at each end
        begin_short = self._compute_inner_line_shortening(
            bond.begin_id, bond, px, py, doc, adjacency)
        end_short = self._compute_inner_line_shortening(
            bond.end_id, bond, px, py, doc, adjacency)

        isx += ux * begin_short
        isy += uy * begin_short
        iex -= ux * end_short
        iey -= uy * end_short

        return isx, isy, iex, iey

    def _determine_double_bond_side(self, bond, doc, adjacency,
                                     ux, uy, bx, by, ex, ey) -> str:
        """Determine which side of the bond the inner line should be on.

        Looks at neighbor atoms of both endpoints; the inner line goes
        toward the side where neighbors cluster (ring interior).
        """
        neighbors = []
        for b, other_id in adjacency.get(bond.begin_id, []):
            if b.id != bond.id:
                other = doc.nodes.get(other_id)
                if other:
                    neighbors.append(other.position)
        for b, other_id in adjacency.get(bond.end_id, []):
            if b.id != bond.id:
                other = doc.nodes.get(other_id)
                if other:
                    neighbors.append(other.position)

        if not neighbors:
            return "right"

        avg_x = sum(p[0] for p in neighbors) / len(neighbors)
        avg_y = sum(p[1] for p in neighbors) / len(neighbors)
        mid_x = (bx + ex) / 2
        mid_y = (by + ey) / 2
        vx = avg_x - mid_x
        vy = avg_y - mid_y

        cross = ux * vy - uy * vx
        return "left" if cross > 0 else "right"

    def _compute_inner_line_shortening(self, node_id, bond, px, py,
                                        doc, adjacency) -> float:
        """Compute shortening of the inner double-bond line at one endpoint.

        Finds the adjacent bond on the same side as the offset and computes:
            shortening = offset * tan(angle_from_perpendicular_to_adjacent)
        """
        node = doc.nodes.get(node_id)
        if not node:
            return 0

        nx, ny = node.position
        offset = doc.bond_spacing / 100 * doc.bond_length

        best_angle = None
        for b, adj_other_id in adjacency.get(node_id, []):
            if b.id == bond.id:
                continue
            adj_node = doc.nodes.get(adj_other_id)
            if not adj_node:
                continue
            ax, ay = adj_node.position
            adx, ady = ax - nx, ay - ny
            adj_len = math.sqrt(adx * adx + ady * ady)
            if adj_len < 1e-10:
                continue
            adj_ux, adj_uy = adx / adj_len, ady / adj_len

            # Is this bond on the same side as the offset?
            dot = adj_ux * px + adj_uy * py
            if dot > 0.01:
                angle = math.acos(max(-1.0, min(1.0, dot)))
                if best_angle is None or angle < best_angle:
                    best_angle = angle

        if best_angle is None:
            return 0

        return offset * math.tan(best_angle)

    # ------------------------------------------------------------------ #
    #  Triple bond inner lines
    # ------------------------------------------------------------------ #

    def _compute_inner_triple_bond_lines(self, bond, doc, adjacency,
                                          ux, uy, bx, by, ex, ey):
        """Compute two inner lines for a triple bond (one on each side)."""
        offset = doc.bond_spacing / 100 * doc.bond_length
        result = []

        for side_sign in [1, -1]:
            px = uy * side_sign
            py = -ux * side_sign

            isx = bx + px * offset
            isy = by + py * offset
            iex = ex + px * offset
            iey = ey + py * offset

            begin_short = self._compute_inner_line_shortening(
                bond.begin_id, bond, px, py, doc, adjacency)
            end_short = self._compute_inner_line_shortening(
                bond.end_id, bond, px, py, doc, adjacency)

            isx += ux * begin_short
            isy += uy * begin_short
            iex -= ux * end_short
            iey -= uy * end_short

            result.append((isx, isy, iex, iey))

        return result

    # ------------------------------------------------------------------ #
    #  Wedge bond
    # ------------------------------------------------------------------ #

    def _render_wedge_bond(self, x1, y1, x2, y2, ux, uy, color,
                           doc: CDXMLDocument) -> list:
        """Render a wedge (solid) bond - filled triangle."""
        px = -uy
        py = ux
        wedge_width = doc.bold_width * 0.75

        p1x, p1y = x1, y1
        p2x, p2y = x2 + px * wedge_width, y2 + py * wedge_width
        p3x, p3y = x2 - px * wedge_width, y2 - py * wedge_width

        return [
            f'  <polygon points="{p1x:.2f},{p1y:.2f} '
            f'{p2x:.2f},{p2y:.2f} {p3x:.2f},{p3y:.2f}" '
            f'fill="{color}" stroke="{color}" stroke-width="0.3"/>'
        ]

    # ------------------------------------------------------------------ #
    #  Hash bond
    # ------------------------------------------------------------------ #

    def _render_hash_bond(self, x1, y1, x2, y2, ux, uy, color,
                          doc: CDXMLDocument) -> list:
        """Render a hashed wedge bond - series of lines getting wider."""
        px = -uy
        py = ux
        wedge_width = doc.bold_width * 0.75
        lw = doc.line_width

        ddx = x2 - x1
        ddy = y2 - y1
        length = math.sqrt(ddx * ddx + ddy * ddy)
        num_lines = max(3, int(length / doc.hash_spacing))

        lines = []
        for i in range(num_lines + 1):
            t = i / num_lines
            mx = x1 + ddx * t
            my = y1 + ddy * t
            w = wedge_width * t

            lx1 = mx + px * w
            ly1 = my + py * w
            lx2 = mx - px * w
            ly2 = my - py * w

            lines.append(
                f'  <line x1="{lx1:.2f}" y1="{ly1:.2f}" '
                f'x2="{lx2:.2f}" y2="{ly2:.2f}" '
                f'stroke="{color}" stroke-width="{lw:.2f}"/>'
            )
        return lines

    # ------------------------------------------------------------------ #
    #  Wavy bond
    # ------------------------------------------------------------------ #

    def _render_wavy_bond(self, x1, y1, x2, y2, color,
                          doc: CDXMLDocument) -> list:
        """Render a wavy bond using a sine-wave path."""
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.001:
            return []

        ux = dx / length
        uy = dy / length
        px = -uy
        py = ux

        amplitude = doc.bold_width * 0.8
        num_waves = max(3, int(length / 3))

        points = []
        for i in range(num_waves * 4 + 1):
            t = i / (num_waves * 4)
            angle = t * num_waves * 2 * math.pi
            off = amplitude * math.sin(angle)

            cx = x1 + dx * t + px * off
            cy = y1 + dy * t + py * off
            points.append(f"{cx:.2f},{cy:.2f}")

        path_d = "M " + " L ".join(points)
        return [
            f'  <path d="{path_d}" fill="none" '
            f'stroke="{color}" stroke-width="{doc.line_width:.2f}"/>'
        ]

    # ------------------------------------------------------------------ #
    #  Bond crossing detection & gap rendering
    # ------------------------------------------------------------------ #

    def _find_crossings(self, doc: CDXMLDocument) -> list:
        """Find all bond pairs that cross without sharing a node.

        Returns [(back_bond, front_bond, intersection_point), ...]
        where bond appearing later in the CDXML is considered 'in front'.
        """
        crossings = []
        bonds = doc.bonds

        for i in range(len(bonds)):
            b1 = bonds[i]
            n1b = doc.nodes.get(b1.begin_id)
            n1e = doc.nodes.get(b1.end_id)
            if not n1b or not n1e:
                continue

            for j in range(i + 1, len(bonds)):
                b2 = bonds[j]

                # Skip if they share a node (connected bonds don't cross)
                if (b1.begin_id in (b2.begin_id, b2.end_id) or
                        b1.end_id in (b2.begin_id, b2.end_id)):
                    continue

                n2b = doc.nodes.get(b2.begin_id)
                n2e = doc.nodes.get(b2.end_id)
                if not n2b or not n2e:
                    continue

                pt = self._segment_intersection(
                    n1b.position, n1e.position,
                    n2b.position, n2e.position)

                if pt is not None:
                    # Later bond (j) is in front, earlier bond (i) is behind
                    crossings.append((b1, b2, pt))

        return crossings

    @staticmethod
    def _segment_intersection(p1, p2, p3, p4):
        """Return intersection point of segments p1-p2 and p3-p4, or None."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3

        denom = dx1 * dy2 - dy1 * dx2
        if abs(denom) < 1e-10:
            return None  # parallel or coincident

        t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / denom
        u = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / denom

        # Strict interior intersection (not at endpoints)
        if 0 < t < 1 and 0 < u < 1:
            return (x1 + t * dx1, y1 + t * dy1)

        return None

    def _render_crossing_gap(self, cross_point, front_bond, back_bond,
                              doc: CDXMLDocument) -> list:
        """Render a white gap band on the back bond at the crossing point.

        Draws a short white line along the front bond's direction, centered
        at the crossing, wide enough to erase the back bond underneath.
        """
        fb = doc.nodes.get(front_bond.begin_id)
        fe = doc.nodes.get(front_bond.end_id)
        if not fb or not fe:
            return []

        dx = fe.position[0] - fb.position[0]
        dy = fe.position[1] - fb.position[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-10:
            return []
        ux, uy = dx / length, dy / length

        cx, cy = cross_point

        # Gap dimensions — proportional to line width
        gap_half_len = doc.line_width * 5.0   # half-length along front bond
        gap_width = doc.line_width * 8.0      # stroke-width to cover back bond

        # Widen gap based on back bond type
        bond_spacing = doc.bond_spacing / 100 * doc.bond_length
        if back_bond.order == 2:
            # Double bond: gap must cover both parallel lines
            gap_width += bond_spacing
        elif back_bond.order == 3:
            # Triple bond: gap must cover all three lines
            gap_width += bond_spacing * 2
        if back_bond.display == "Bold":
            # Bold/thick bond: gap must cover the wider stroke
            gap_width = max(gap_width, doc.bold_width + doc.line_width * 6.0)

        # Widen gap length based on front bond type
        if front_bond.order == 2:
            gap_half_len += bond_spacing / 2
        elif front_bond.order == 3:
            gap_half_len += bond_spacing
        if front_bond.display == "Bold":
            gap_half_len = max(gap_half_len, doc.bold_width / 2 + doc.line_width * 4.0)

        x1 = cx - ux * gap_half_len
        y1 = cy - uy * gap_half_len
        x2 = cx + ux * gap_half_len
        y2 = cy + uy * gap_half_len

        return [
            f'  <line x1="{x1:.2f}" y1="{y1:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="white" stroke-width="{gap_width:.2f}" '
            f'stroke-linecap="butt"/>'
        ]

    # ------------------------------------------------------------------ #
    #  Node labels
    # ------------------------------------------------------------------ #

    def _node_has_visible_label(self, node: Node) -> bool:
        """Check if a node should display a label."""
        if node.has_label_text:
            return True
        if node.element != 6:
            return True
        if node.node_type in ("Fragment", "Nickname", "GenericNickname",
                              "Unspecified"):
            return True
        return False

    def _get_label_text(self, node: Node) -> str:
        """Get the display text for a node."""
        if node.label:
            return node.label.get_text()
        if node.element != 6:
            return ELEMENT_MAP.get(node.element, f"?{node.element}")
        return ""

    def _render_node_label(self, node: Node, doc: CDXMLDocument) -> list:
        """Render the label for an atom node."""
        if not self._node_has_visible_label(node):
            return []

        x, y = node.position
        lines = []

        if node.label and node.label.spans:
            lines.extend(self._render_atom_text(node, doc))
        elif node.element != 6:
            symbol = ELEMENT_MAP.get(node.element, f"?{node.element}")
            color = self._get_color_str(node.color_idx, doc)
            font_size = doc.label_size
            lines.append(
                f'  <text x="{x:.2f}" y="{y:.2f}" '
                f'font-family="{self.font_family}" font-size="{font_size:.1f}" '
                f'fill="{color}" text-anchor="middle" '
                f'dominant-baseline="central" '
                f'font-weight="bold">{symbol}</text>'
            )

        return lines

    def _render_atom_text(self, node: Node, doc: CDXMLDocument) -> list:
        """Render text for an atom that has a label with spans.

        Auto-subscripts digits in chemical formulas (e.g. NH2 -> NH₂).
        Supports multi-line labels via LineStarts (e.g. NH with
        LabelAlignment="Below" renders N on line 1, H on line 2).
        """
        if not node.label:
            return []

        label = node.label
        x, y = node.position
        lines = []

        tx, ty = label.position
        ta = self._get_text_anchor(label)

        # Get the full concatenated text for LineStarts splitting
        full_text = label.get_text()

        # Determine line breaks from LineStarts
        # LineStarts values are 1-based character indices where new lines begin
        line_breaks = []
        if label.line_starts and len(label.line_starts) >= 2:
            # Convert 1-based char positions to 0-based indices
            line_breaks = [ls - 1 for ls in label.line_starts]

        if line_breaks and len(line_breaks) >= 2:
            # Multi-line label: split text by LineStarts positions
            line_texts = []
            for i in range(len(line_breaks)):
                start = line_breaks[i] - 1  # LineStarts is 1-based
                if i + 1 < len(line_breaks):
                    end = line_breaks[i + 1] - 1
                else:
                    end = len(full_text)
                line_texts.append(full_text[start:end])

            # Determine font properties from first span
            first_span = label.spans[0] if label.spans else None
            if not first_span:
                return lines

            s_color = self._get_color_str(first_span.color_idx, doc)
            font_size = first_span.size
            weight = "bold" if first_span.is_bold else "normal"
            style = "italic" if first_span.is_italic else "normal"

            # Derive line height from BoundingBox when available
            n_lines = len(line_texts)
            if label.bounding_box and n_lines >= 2:
                bb_top = label.bounding_box[1]
                bb_bottom = label.bounding_box[3]
                line_height = (bb_bottom - bb_top) / n_lines
            else:
                line_height = font_size * 1.2

            for line_idx, line_text in enumerate(line_texts):
                escaped = self._xml_escape(line_text)
                segments = self._split_label_segments(escaped)
                text_parts = []
                pending_dy = 0.0
                for seg_text, is_digit in segments:
                    if is_digit:
                        sub_size = font_size * 0.75
                        sub_dy = font_size * 0.25
                        total_dy = sub_dy + pending_dy
                        text_parts.append(
                            f'<tspan fill="{s_color}" '
                            f'font-size="{sub_size:.1f}" '
                            f'font-weight="{weight}" font-style="{style}" '
                            f'dy="{total_dy:.1f}">'
                            f'{seg_text}</tspan>'
                        )
                        pending_dy = -sub_dy
                    else:
                        dy_attr = f' dy="{pending_dy:.1f}"' if pending_dy else ''
                        text_parts.append(
                            f'<tspan fill="{s_color}" '
                            f'font-size="{font_size:.1f}" '
                            f'font-weight="{weight}" font-style="{style}"{dy_attr}>'
                            f'{seg_text}</tspan>'
                        )
                        pending_dy = 0.0
                # Final reset if label ends with a subscript digit
                if pending_dy:
                    text_parts.append(
                        f'<tspan dy="{pending_dy:.1f}"></tspan>'
                    )

                text_content = ''.join(text_parts)
                ly = ty + line_idx * line_height

                lines.append(
                    f'  <text x="{tx:.2f}" y="{ly:.2f}" '
                    f'font-family="{self.font_family}" '
                    f'text-anchor="{ta}" '
                    f'dominant-baseline="auto">'
                    f'{text_content}</text>'
                )

            return lines

        # Single-line label (original path)
        # When LabelAlignment="Right", reverse atom groups for display
        # so the connecting atom ends up at the right edge where bonds attach.
        # E.g. CDXML stores "OMe" → display as "MeO"
        #      CDXML stores "NH2" → display as "H2N"
        is_right_aligned = (label.label_alignment == "Right")

        text_parts = []
        pending_dy = 0.0
        for span in label.spans:
            s_color = self._get_color_str(span.color_idx, doc)
            font_size = span.size
            weight = "bold" if span.is_bold else "normal"
            style = "italic" if span.is_italic else "normal"
            raw_text = span.text

            # Reverse atom groups if right-aligned
            if is_right_aligned:
                raw_text = self._reverse_atom_groups(raw_text)

            escaped = self._xml_escape(raw_text)

            # Auto-subscript: split text into letter/digit segments
            segments = self._split_label_segments(escaped)
            for seg_text, is_digit in segments:
                if is_digit:
                    sub_size = font_size * 0.75
                    sub_dy = font_size * 0.25
                    total_dy = sub_dy + pending_dy
                    text_parts.append(
                        f'<tspan fill="{s_color}" '
                        f'font-size="{sub_size:.1f}" '
                        f'font-weight="{weight}" font-style="{style}" '
                        f'dy="{total_dy:.1f}">'
                        f'{seg_text}</tspan>'
                    )
                    pending_dy = -sub_dy
                else:
                    dy_attr = f' dy="{pending_dy:.1f}"' if pending_dy else ''
                    text_parts.append(
                        f'<tspan fill="{s_color}" '
                        f'font-size="{font_size:.1f}" '
                        f'font-weight="{weight}" font-style="{style}"{dy_attr}>'
                        f'{seg_text}</tspan>'
                    )
                    pending_dy = 0.0

        text_content = ''.join(text_parts)

        lines.append(
            f'  <text x="{tx:.2f}" y="{ty:.2f}" '
            f'font-family="{self.font_family}" '
            f'text-anchor="{ta}" '
            f'dominant-baseline="auto">'
            f'{text_content}</text>'
        )

        return lines

    @staticmethod
    def _split_label_segments(text: str) -> list:
        """Split atom label text into (text, is_digit) segments.

        E.g. 'NH2' -> [('NH', False), ('2', True)]
             'CH3' -> [('CH', False), ('3', True)]
             'OH'  -> [('OH', False)]
        """
        segments = []
        current = ''
        current_is_digit = False

        for ch in text:
            is_d = ch.isdigit()
            if current and is_d != current_is_digit:
                segments.append((current, current_is_digit))
                current = ''
            current += ch
            current_is_digit = is_d

        if current:
            segments.append((current, current_is_digit))

        return segments

    @staticmethod
    def _reverse_atom_groups(text: str) -> str:
        """Reverse atom groups in a chemical label for right-aligned display.

        CDXML stores labels with the connecting atom first.  When
        LabelAlignment='Right', the display must be reversed at the
        atom-group level so the connecting atom appears at the right
        edge (where the bond attaches).

        An atom group starts with an uppercase letter followed by
        optional lowercase letters and digits.

        Examples:
            'OMe'  -> 'MeO'
            'NH2'  -> 'H2N'
            'OH'   -> 'HO'
            'OTIPS' -> 'TIPSO'
            'Br'   -> 'Br'   (single group, no change)
            'Cbz'  -> 'Cbz'  (single group, no change)
        """
        import re
        # Split into atom groups: uppercase letter, then optional lowercase + digits
        groups = re.findall(r'[A-Z][a-z]*\d*', text)
        if len(groups) <= 1:
            return text
        # Check that we captured the full string (no unexpected chars)
        if ''.join(groups) != text:
            return text  # Don't reverse if parsing is ambiguous
        return ''.join(reversed(groups))

    # ------------------------------------------------------------------ #
    #  Arrows
    # ------------------------------------------------------------------ #

    def _render_arrow(self, arrow: Arrow, doc: CDXMLDocument) -> list:
        """Render a reaction arrow matching ChemDraw's exact geometry.

        ChemDraw uses cubic Bezier curves for arrowheads (not polygons).
        The scaling factor for arrow dimensions is: sf = line_width / 100.
        All CDXML arrow attributes (HeadSize, ArrowheadCenterSize,
        ArrowheadWidth, ArrowShaftSpacing) are multiplied by sf to get
        CDXML coordinate distances.

        Supports:
        - Full arrowhead (curved-back filled shape)
        - HalfLeft / HalfRight arrowheads (half-triangles with curve)
        - Equilibrium (double-shaft with opposing half-arrowheads)
        - NoGo="Cross" (does-not-proceed X mark)
        """
        tx, ty = arrow.tail
        hx, hy = arrow.head
        lw = doc.line_width
        lines: list = []

        dx = hx - tx
        dy = hy - ty
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.001:
            return lines

        # Unit vectors: u = along arrow, p = perpendicular (left of u)
        ux = dx / length
        uy = dy / length
        px = -uy
        py = ux

        # Scale factor: CDXML attribute value * sf = distance in CDXML coords
        sf = lw / 100.0
        head_len = arrow.head_size * sf
        indent = arrow.head_center_size * sf
        hw = arrow.head_width * sf          # half-width from center to wing
        hlw = lw / 2.0                      # half line-width (shaft edge offset)

        is_equilibrium = (arrow.shaft_spacing > 0 and
                          arrow.arrowhead_tail != "")

        if is_equilibrium:
            self._render_equilibrium_arrow(
                lines, tx, ty, hx, hy, ux, uy, px, py,
                head_len, indent, hw, hlw, lw, sf, arrow
            )
        else:
            self._render_standard_arrow(
                lines, tx, ty, hx, hy, ux, uy, px, py,
                head_len, indent, hw, hlw, lw, arrow
            )

        # NoGo cross mark (does-not-proceed): X across the shaft
        if arrow.nogo == "Cross":
            self._render_nogo_cross(
                lines, tx, ty, hx, hy, ux, uy, px, py,
                head_len, hlw, lw
            )

        return lines

    def _render_arrow(self, arrow: Arrow, doc: CDXMLDocument) -> list:
        """Render a reaction arrow or plain line.

        ChemDraw uses cubic Bezier curves for arrowheads (not polygons).
        The scaling factor for arrow dimensions is: sf = line_width / 100.

        Supports:
        - Plain lines (no arrowhead on either end)
        - Dashed lines (LineType="Dashed")
        - Full arrowhead (curved-back filled shape)
        - HalfLeft / HalfRight arrowheads (half-triangles with curve)
        - Tail-only arrowheads (ArrowheadTail without ArrowheadHead)
        - Equilibrium (double-shaft with opposing half-arrowheads)
        - NoGo="Cross" (does-not-proceed X mark)
        """
        tx, ty = arrow.tail
        hx, hy = arrow.head
        lw = doc.line_width
        lines: list = []

        dx = hx - tx
        dy = hy - ty
        length = math.sqrt(dx * dx + dy * dy)
        if length < 0.001:
            return lines

        # Unit vectors: u = along arrow, p = perpendicular (left of u)
        ux = dx / length
        uy = dy / length
        px = -uy
        py = ux

        # Scale factor: CDXML attribute value * sf = distance in CDXML coords
        sf = lw / 100.0
        head_len = arrow.head_size * sf
        indent = arrow.head_center_size * sf
        hw = arrow.head_width * sf          # half-width from center to wing
        hlw = lw / 2.0                      # half line-width (shaft edge offset)

        has_head = arrow.arrowhead_head != ""
        has_tail = arrow.arrowhead_tail != ""

        is_equilibrium = (arrow.shaft_spacing > 0 and has_tail)
        is_plain_line = not has_head and not has_tail

        if is_plain_line:
            self._render_plain_line(
                lines, tx, ty, hx, hy, lw, arrow, doc
            )
        elif is_equilibrium:
            self._render_equilibrium_arrow(
                lines, tx, ty, hx, hy, ux, uy, px, py,
                head_len, indent, hw, hlw, lw, sf, arrow
            )
        else:
            self._render_standard_arrow(
                lines, tx, ty, hx, hy, ux, uy, px, py,
                head_len, indent, hw, hlw, lw, arrow
            )

        # NoGo cross mark (does-not-proceed): X across the shaft
        if arrow.nogo == "Cross":
            self._render_nogo_cross(
                lines, tx, ty, hx, hy, ux, uy, px, py,
                head_len, hlw, lw
            )

        return lines

    def _render_plain_line(self, lines, tx, ty, hx, hy, lw, arrow, doc):
        """Render a plain line (no arrowheads) — solid or dashed."""
        dash = ''
        if arrow.line_type == "Dashed":
            dash = f' stroke-dasharray="{lw * 4:.2f},{lw * 3:.2f}"'

        color = self._get_color_str(arrow.color_idx, doc)
        lines.append(
            f'  <line x1="{tx:.2f}" y1="{ty:.2f}" '
            f'x2="{hx:.2f}" y2="{hy:.2f}" '
            f'stroke="{color}" stroke-width="{lw:.2f}"{dash}/>'
        )

    def _render_standard_arrow(self, lines, tx, ty, hx, hy,
                               ux, uy, px, py,
                               head_len, indent, hw, hlw, lw, arrow):
        """Render a single-shaft arrow with arrowhead at head and/or tail."""
        has_head = arrow.arrowhead_head != ""
        has_tail_only = (not has_head and arrow.arrowhead_tail != "")

        if has_tail_only:
            # Tail-only arrow: draw arrowhead at tail end
            # Flip direction: arrowhead points from head toward tail
            # Shaft runs from head to the arrowhead indent at tail
            sex = tx + ux * indent
            sey = ty + uy * indent

            s1x = hx + px * hlw;  s1y = hy + py * hlw
            s2x = hx - px * hlw;  s2y = hy - py * hlw
            s3x = sex - px * hlw; s3y = sey - py * hlw
            s4x = sex + px * hlw; s4y = sey + py * hlw

            lines.append(
                f'  <path d="M {s1x:.2f},{s1y:.2f} '
                f'L {s2x:.2f},{s2y:.2f} '
                f'L {s3x:.2f},{s3y:.2f} '
                f'L {s4x:.2f},{s4y:.2f} Z" '
                f'fill="rgb(0,0,0)" stroke="none"/>'
            )

            # Draw arrowhead at tail pointing in reverse direction
            tail_type = arrow.arrowhead_tail
            if tail_type == "Full":
                self._draw_full_arrowhead(
                    lines, tx, ty, -ux, -uy, -px, -py,
                    head_len, indent, hw
                )
            elif tail_type in ("HalfLeft", "HalfRight"):
                side = "left" if tail_type == "HalfLeft" else "right"
                self._draw_half_arrowhead(
                    lines, tx, ty, -ux, -uy, -px, -py,
                    head_len, indent, hw, hlw, side
                )
        else:
            # Standard head arrow (possibly with tail arrowhead too)
            # Shaft from tail to arrowhead indent
            sex = hx - ux * indent
            sey = hy - uy * indent

            s1x = tx + px * hlw;  s1y = ty + py * hlw
            s2x = tx - px * hlw;  s2y = ty - py * hlw
            s3x = sex - px * hlw; s3y = sey - py * hlw
            s4x = sex + px * hlw; s4y = sey + py * hlw

            lines.append(
                f'  <path d="M {s1x:.2f},{s1y:.2f} '
                f'L {s2x:.2f},{s2y:.2f} '
                f'L {s3x:.2f},{s3y:.2f} '
                f'L {s4x:.2f},{s4y:.2f} Z" '
                f'fill="rgb(0,0,0)" stroke="none"/>'
            )

            # Head arrowhead
            head_type = arrow.arrowhead_head
            if head_type == "Full":
                self._draw_full_arrowhead(
                    lines, hx, hy, ux, uy, px, py,
                    head_len, indent, hw
                )
            elif head_type in ("HalfLeft", "HalfRight"):
                side = "left" if head_type == "HalfLeft" else "right"
                self._draw_half_arrowhead(
                    lines, hx, hy, ux, uy, px, py,
                    head_len, indent, hw, hlw, side
                )

    def _render_equilibrium_arrow(self, lines, tx, ty, hx, hy,
                                  ux, uy, px, py,
                                  head_len, indent, hw, hlw, lw, sf, arrow):
        """Render an equilibrium arrow (two parallel shafts with opposing heads)."""
        shaft_offset = arrow.shaft_spacing * sf / 2.0

        # Top shaft center (forward, arrowhead at head end)
        t1cx = tx + px * shaft_offset
        t1cy = ty + py * shaft_offset
        h1cx = hx + px * shaft_offset
        h1cy = hy + py * shaft_offset

        # Bottom shaft center (reverse, arrowhead at tail end)
        t2cx = tx - px * shaft_offset
        t2cy = ty - py * shaft_offset
        h2cx = hx - px * shaft_offset
        h2cy = hy - py * shaft_offset

        # Forward half-arrowhead tip extends beyond head_x by shaft_offset
        fwd_tip_x = h1cx + ux * shaft_offset
        fwd_tip_y = h1cy + uy * shaft_offset

        # Reverse half-arrowhead tip extends beyond tail_x by shaft_offset
        rev_tip_x = t2cx - ux * shaft_offset
        rev_tip_y = t2cy - uy * shaft_offset

        # Forward shaft: from tail to arrowhead junction
        fwd_junc_x = fwd_tip_x - ux * indent
        fwd_junc_y = fwd_tip_y - uy * indent

        # Reverse shaft: from arrowhead junction to head position
        rev_junc_x = rev_tip_x + ux * indent
        rev_junc_y = rev_tip_y + uy * indent

        # Draw top shaft (forward) as rectangle
        self._draw_shaft_rect(lines, t1cx, t1cy, fwd_junc_x, fwd_junc_y,
                              px, py, hlw)

        # Draw bottom shaft (reverse) as rectangle
        self._draw_shaft_rect(lines, rev_junc_x, rev_junc_y, h2cx, h2cy,
                              px, py, hlw)

        # Forward half-arrowhead (on top shaft, wing extends outward = left)
        self._draw_half_arrowhead(
            lines, fwd_tip_x, fwd_tip_y, ux, uy, px, py,
            head_len, indent, hw, hlw, "left"
        )

        # Reverse half-arrowhead (on bottom shaft, wing extends outward = left
        # in reverse direction, which is "right" relative to forward)
        self._draw_half_arrowhead(
            lines, rev_tip_x, rev_tip_y, -ux, -uy, -px, -py,
            head_len, indent, hw, hlw, "left"
        )

    @staticmethod
    def _draw_shaft_rect(lines, x1, y1, x2, y2, px, py, hlw):
        """Draw a shaft as a filled rectangle along (x1,y1)→(x2,y2)."""
        s1x = x1 + px * hlw;  s1y = y1 + py * hlw
        s2x = x1 - px * hlw;  s2y = y1 - py * hlw
        s3x = x2 - px * hlw;  s3y = y2 - py * hlw
        s4x = x2 + px * hlw;  s4y = y2 + py * hlw

        lines.append(
            f'  <path d="M {s1x:.2f},{s1y:.2f} '
            f'L {s2x:.2f},{s2y:.2f} '
            f'L {s3x:.2f},{s3y:.2f} '
            f'L {s4x:.2f},{s4y:.2f} Z" '
            f'fill="rgb(0,0,0)" stroke="none"/>'
        )

    @staticmethod
    def _draw_full_arrowhead(lines, tip_x, tip_y, ux, uy, px, py,
                             head_len, indent, hw):
        """Draw a filled full arrowhead with cubic-Bezier curved back.

        Shape: tip → bottom_wing (straight) → indent_center (curve) →
               top_wing (curve) → tip (straight).
        The curve control points are at 7/16 of half_width from center,
        positioned at the indent x-position.
        """
        # Key points
        # Bottom wing (right side when looking along arrow, positive p)
        bwx = tip_x - ux * head_len + px * hw
        bwy = tip_y - uy * head_len + py * hw
        # Top wing (left side, negative p)
        twx = tip_x - ux * head_len - px * hw
        twy = tip_y - uy * head_len - py * hw
        # Indent center (notch at back of arrowhead)
        ix = tip_x - ux * indent
        iy = tip_y - uy * indent
        # Curve control points at indent x, 7/16 hw from center
        k = hw * 7.0 / 16.0
        cb_x = ix + px * k;  cb_y = iy + py * k   # bottom ctrl
        ct_x = ix - px * k;  ct_y = iy - py * k   # top ctrl

        # Build cubic Bezier path (matching ChemDraw exactly)
        # Segment 1: tip → bottom_wing (straight line via degenerate cubic)
        # Segment 2: bottom_wing → indent_center (curved)
        # Segment 3: indent_center → top_wing (curved)
        # Segment 4: top_wing → tip (straight line via degenerate cubic)
        d = (f'M {tip_x:.2f},{tip_y:.2f} '
             f'C {tip_x:.2f},{tip_y:.2f} {bwx:.2f},{bwy:.2f} {bwx:.2f},{bwy:.2f} '
             f'C {bwx:.2f},{bwy:.2f} {cb_x:.2f},{cb_y:.2f} {ix:.2f},{iy:.2f} '
             f'C {ct_x:.2f},{ct_y:.2f} {twx:.2f},{twy:.2f} {twx:.2f},{twy:.2f} '
             f'C {twx:.2f},{twy:.2f} {tip_x:.2f},{tip_y:.2f} {tip_x:.2f},{tip_y:.2f}')

        # Filled shape
        lines.append(
            f'  <path d="{d}" fill="rgb(0,0,0)" stroke="none"/>'
        )
        # Outline stroke (thin, matching ChemDraw)
        lines.append(
            f'  <path d="{d}" fill="none" stroke="rgb(0,0,0)" '
            f'stroke-width="0.05"/>'
        )

    @staticmethod
    def _draw_half_arrowhead(lines, tip_x, tip_y, ux, uy, px, py,
                             head_len, indent, hw, hlw, side):
        """Draw a filled half-arrowhead with curved back.

        The half-arrowhead sits on the shaft edge. The tip and junction
        are on the inner edge (toward the other shaft for equilibrium),
        and the wing extends outward.

        For 'left' side: wing extends in +p direction (outward/up).
        For 'right' side: wing extends in -p direction (outward/down).
        """
        if side == "left":
            sp = 1.0    # sign for perpendicular
        else:
            sp = -1.0

        # Tip on shaft inner edge
        tip_ex = tip_x - px * sp * hlw
        tip_ey = tip_y - py * sp * hlw

        # Junction on shaft inner edge (where shaft rectangle meets arrowhead)
        jx = tip_ex - ux * indent
        jy = tip_ey - uy * indent

        # Wing point (extends outward by hw from shaft CENTER, not edge)
        # The shaft center is at tip position. Wing = center + hw outward.
        wing_x = tip_x - ux * head_len + px * sp * hw
        wing_y = tip_y - uy * head_len + py * sp * hw

        # Curve control point: at junction x-position, 7/16 hw from center
        k = hw * 7.0 / 16.0
        ctrl_x = jx + px * sp * (k + hlw)  # offset from junction edge
        ctrl_y = jy + py * sp * (k + hlw)

        # Path: tip_edge → junction (straight along shaft edge) →
        #        wing (curve) → tip_edge (straight diagonal)
        d = (f'M {tip_ex:.2f},{tip_ey:.2f} '
             f'C {tip_ex:.2f},{tip_ey:.2f} {jx:.2f},{jy:.2f} {jx:.2f},{jy:.2f} '
             f'C {jx:.2f},{ctrl_y:.2f} {wing_x:.2f},{wing_y:.2f} {wing_x:.2f},{wing_y:.2f} '
             f'C {wing_x:.2f},{wing_y:.2f} {tip_ex:.2f},{tip_ey:.2f} {tip_ex:.2f},{tip_ey:.2f}')

        lines.append(
            f'  <path d="{d}" fill="rgb(0,0,0)" stroke="none"/>'
        )
        lines.append(
            f'  <path d="{d}" fill="none" stroke="rgb(0,0,0)" '
            f'stroke-width="0.05"/>'
        )

    @staticmethod
    def _render_nogo_cross(lines, tx, ty, hx, hy, ux, uy, px, py,
                           head_len, hlw, lw):
        """Render a does-not-proceed X mark at the center of the shaft."""
        cx = (tx + hx) / 2
        cy = (ty + hy) / 2
        # Cross half-extent ≈ 3.2 CDXML for HeadSize=1000, sf=0.006
        cross_half = head_len * 0.533
        # Cross line half-width
        clw = lw * 0.33

        # Four diagonal directions (±u ± p) * cross_half
        # Line 1: (+u+p) to (-u-p) with width
        d1ax = cx + (ux + px) * cross_half
        d1ay = cy + (uy + py) * cross_half
        d1bx = cx - (ux + px) * cross_half
        d1by = cy - (uy + py) * cross_half

        # Line 2: (+u-p) to (-u+p) with width
        d2ax = cx + (ux - px) * cross_half
        d2ay = cy + (uy - py) * cross_half
        d2bx = cx - (ux - px) * cross_half
        d2by = cy - (uy - py) * cross_half

        # Draw as two rectangles (rotated)
        # Normal to line 1: direction perpendicular to (u+p)
        n1x = -(uy + py); n1y = (ux + px)
        n1_len = math.sqrt(n1x * n1x + n1y * n1y)
        if n1_len > 0:
            n1x /= n1_len; n1y /= n1_len
        n2x = -(uy - py); n2y = (ux - px)
        n2_len = math.sqrt(n2x * n2x + n2y * n2y)
        if n2_len > 0:
            n2x /= n2_len; n2y /= n2_len

        for (ax, ay, bx, by, nx, ny) in [
            (d1ax, d1ay, d1bx, d1by, n1x, n1y),
            (d2ax, d2ay, d2bx, d2by, n2x, n2y),
        ]:
            p1x = ax + nx * clw; p1y = ay + ny * clw
            p2x = bx + nx * clw; p2y = by + ny * clw
            p3x = bx - nx * clw; p3y = by - ny * clw
            p4x = ax - nx * clw; p4y = ay - ny * clw
            lines.append(
                f'  <path d="M {p1x:.2f},{p1y:.2f} '
                f'L {p2x:.2f},{p2y:.2f} '
                f'L {p3x:.2f},{p3y:.2f} '
                f'L {p4x:.2f},{p4y:.2f} Z" '
                f'fill="rgb(0,0,0)" stroke="none"/>'
            )

    # ------------------------------------------------------------------ #
    #  Standalone text
    # ------------------------------------------------------------------ #

    def _render_text(self, text_label: TextLabel, doc: CDXMLDocument) -> list:
        """Render standalone text annotation.

        Line breaks:
        1. Each \\n in the CDXML text produces a line break.
        2. If a line exceeds the available width (WordWrapWidth or
           BoundingBox width), it is word-wrapped at space boundaries.

        Handles subscript (face=32) and superscript (face=64) spans.
        Tspans on the same visual line are joined without whitespace
        to avoid SVG renderers inserting visible space characters.
        """
        if not text_label.spans:
            return []

        lines = []
        tx, ty = text_label.position

        ta = "start"
        just = text_label.caption_justification or text_label.justification
        if just == "Center":
            ta = "middle"
        elif just == "Right":
            ta = "end"

        # Determine line height from CDXML LineHeight attribute or fallback
        if text_label.line_height != "auto":
            try:
                line_height = float(text_label.line_height)
            except ValueError:
                line_height = doc.caption_size * 1.4
        else:
            line_height = doc.caption_size * 1.4

        # Determine max line width for word-wrapping (CDXML coordinate units).
        # Priority: WordWrapWidth > BoundingBox width > 0 (no wrap).
        # Only word-wrap when WordWrapWidth is explicitly set (page-level overflow).
        max_width = text_label.word_wrap_width if text_label.word_wrap_width > 0 else 0.0

        # Walk spans, splitting on \n to produce visual lines.
        # Each visual line is a list of (span, text_fragment) pairs.
        visual_lines = []          # list of [(span, text), ...]
        current_line_parts = []    # (span, text) for current line

        for span in text_label.spans:
            fragments = span.text.split("\n")
            for i, frag in enumerate(fragments):
                if i > 0:
                    # \n encountered — flush current line
                    visual_lines.append(current_line_parts)
                    current_line_parts = []
                if frag:
                    current_line_parts.append((span, frag))
        if current_line_parts:
            visual_lines.append(current_line_parts)

        # Word-wrap lines that exceed max_width.
        if max_width > 0:
            wrapped = []
            for line_parts in visual_lines:
                wrapped.extend(
                    self._word_wrap_line(line_parts, max_width))
            visual_lines = wrapped

        # Expand formula-mode spans: auto-subscript digits after letters.
        visual_lines = [self._expand_formula_spans(parts)
                        for parts in visual_lines]

        is_single_line = len(visual_lines) <= 1

        if is_single_line:
            # Single-line text
            text_parts = []
            pending_dy = 0.0
            parts = visual_lines[0] if visual_lines else []
            for span, text_frag in parts:
                s_color = self._get_color_str(span.color_idx, doc)
                font_size = span.size
                weight = "bold" if span.is_bold else "normal"
                style = "italic" if span.is_italic else "normal"
                escaped = self._xml_escape(text_frag)

                if span.is_subscript:
                    sub_size = font_size * 0.75
                    sub_dy = font_size * 0.25
                    total_dy = sub_dy + pending_dy
                    text_parts.append(
                        f'<tspan fill="{s_color}" font-size="{sub_size:.1f}" '
                        f'font-weight="{weight}" font-style="{style}" '
                        f'dy="{total_dy:.1f}">'
                        f'{escaped}</tspan>'
                    )
                    pending_dy = -sub_dy
                elif span.is_superscript:
                    sup_size = font_size * 0.75
                    sup_dy = -font_size * 0.35
                    total_dy = sup_dy + pending_dy
                    text_parts.append(
                        f'<tspan fill="{s_color}" font-size="{sup_size:.1f}" '
                        f'font-weight="{weight}" font-style="{style}" '
                        f'dy="{total_dy:.1f}">'
                        f'{escaped}</tspan>'
                    )
                    pending_dy = -sup_dy
                else:
                    dy_attr = f' dy="{pending_dy:.1f}"' if pending_dy else ''
                    text_parts.append(
                        f'<tspan fill="{s_color}" font-size="{font_size:.1f}" '
                        f'font-weight="{weight}" font-style="{style}"{dy_attr}>'
                        f'{escaped}</tspan>'
                    )
                    pending_dy = 0.0

            if pending_dy:
                text_parts.append(f'<tspan dy="{pending_dy:.1f}"></tspan>')

            text_content = ''.join(text_parts)
            lines.append(
                f'  <text x="{tx:.2f}" y="{ty:.2f}" '
                f'font-family="{self.font_family}" '
                f'text-anchor="{ta}" '
                f'dominant-baseline="auto">'
                f'{text_content}</text>'
            )
        else:
            # Multi-line text
            rendered_lines = []
            pending_dy = 0.0

            for line_idx, line_parts in enumerate(visual_lines):
                if not line_parts and line_idx > 0:
                    continue

                current_parts = []
                for span, text_frag in line_parts:
                    s_color = self._get_color_str(span.color_idx, doc)
                    font_size = span.size
                    weight = "bold" if span.is_bold else "normal"
                    style = "italic" if span.is_italic else "normal"
                    escaped = self._xml_escape(text_frag)

                    if span.is_subscript:
                        sub_size = font_size * 0.75
                        sub_dy = font_size * 0.25
                        total_dy = sub_dy + pending_dy
                        current_parts.append(
                            f'<tspan fill="{s_color}" '
                            f'font-size="{sub_size:.1f}" '
                            f'font-weight="{weight}" font-style="{style}" '
                            f'dy="{total_dy:.1f}">'
                            f'{escaped}</tspan>')
                        pending_dy = -sub_dy
                    elif span.is_superscript:
                        sup_size = font_size * 0.75
                        sup_dy = -font_size * 0.35
                        total_dy = sup_dy + pending_dy
                        current_parts.append(
                            f'<tspan fill="{s_color}" '
                            f'font-size="{sup_size:.1f}" '
                            f'font-weight="{weight}" font-style="{style}" '
                            f'dy="{total_dy:.1f}">'
                            f'{escaped}</tspan>')
                        pending_dy = -sup_dy
                    else:
                        dy_attr = (f' dy="{pending_dy:.1f}"'
                                   if pending_dy else '')
                        current_parts.append(
                            f'<tspan fill="{s_color}" '
                            f'font-size="{font_size:.1f}" '
                            f'font-weight="{weight}" font-style="{style}"'
                            f'{dy_attr}>'
                            f'{escaped}</tspan>')
                        pending_dy = 0.0

                # Reset baseline between lines
                if pending_dy and line_idx < len(visual_lines) - 1:
                    current_parts.append(
                        f'<tspan dy="{pending_dy:.1f}"></tspan>')
                    pending_dy = 0.0

                rendered_lines.append(''.join(current_parts))

            # Trailing baseline reset
            if pending_dy and rendered_lines:
                rendered_lines[-1] += (
                    f'<tspan dy="{pending_dy:.1f}"></tspan>')

            # Build <text> element
            lines.append(
                f'  <text x="{tx:.2f}" y="{ty:.2f}" '
                f'font-family="{self.font_family}" '
                f'text-anchor="{ta}" '
                f'dominant-baseline="auto">'
            )
            for i, rl in enumerate(rendered_lines):
                if i == 0:
                    lines.append(f'    {rl}')
                else:
                    # Inject x-reset and dy for line break into the
                    # first tspan.  If that tspan already carries a dy
                    # (e.g. from a subscript shift), merge the two
                    # values instead of adding a duplicate attribute.
                    first_tag = re.match(r'<tspan\s[^>]*>', rl)
                    if first_tag and 'dy="' in first_tag.group(0):
                        # First tspan already has dy — combine values
                        tag_str = first_tag.group(0)
                        dm = re.search(r'dy="([^"]+)"', tag_str)
                        existing_dy = float(dm.group(1))
                        combined = line_height + existing_dy
                        new_tag = tag_str.replace(
                            dm.group(0),
                            f'dy="{combined:.1f}"')
                        new_tag = new_tag.replace(
                            '<tspan ',
                            f'<tspan x="{tx:.2f}" ', 1)
                        rl = new_tag + rl[first_tag.end():]
                    else:
                        rl = rl.replace(
                            '<tspan ',
                            f'<tspan x="{tx:.2f}" dy="{line_height:.1f}" ',
                            1)
                    lines.append(f'    {rl}')
            lines.append('  </text>')

        return lines

    @staticmethod
    def _expand_formula_spans(line_parts):
        """Expand formula-mode spans into normal + subscript sub-parts.

        In ChemDraw formula mode (face bits 5+6 = 96), digits that
        follow letters are rendered as subscripts.  This method splits
        such spans into alternating normal/subscript TextSpan objects
        so the downstream tspan renderer handles them automatically.

        Non-formula spans are passed through unchanged.

        Cross-span context: when a formula digit starts a span, the
        last character of the preceding part's text is checked so that
        e.g. ``<s>SO</s><s face="96">3</s>`` correctly subscripts the 3.
        """
        result = []
        # Track the last character emitted across all parts so we can
        # do cross-span preceding-character checks.
        prev_last_char = ''

        for span, text in line_parts:
            if not span.is_formula:
                result.append((span, text))
                if text:
                    prev_last_char = text[-1]
                continue

            # Split text into segments: letters/symbols vs trailing digits
            segments = re.findall(r'[A-Za-z()\[\]{}/·•\-,;: ]+|\d+', text)
            if not segments:
                result.append((span, text))
                if text:
                    prev_last_char = text[-1]
                continue

            # Walk through the original text to decide which digits
            # should be subscripted.  Rule: a run of digits is subscripted
            # if it immediately follows a letter or ')'.
            pos = 0
            for seg in segments:
                if not seg:
                    continue
                seg_start = text.find(seg, pos)
                # Emit any skipped characters as normal
                if seg_start > pos:
                    skipped = text[pos:seg_start]
                    result.append((span, skipped))

                if seg[0].isdigit():
                    # Check if preceded by a letter or ')' → subscript.
                    # Use cross-span context when at position 0.
                    if seg_start > 0:
                        preceding = text[seg_start - 1]
                    else:
                        preceding = prev_last_char
                    if preceding and (preceding.isalpha() or preceding in ')]}'):
                        # Create a subscript variant of this span
                        sub_span = TextSpan(
                            text=seg,
                            font_id=span.font_id,
                            size=span.size,
                            color_idx=span.color_idx,
                            face=(span.face & ~96) | 32,  # clear formula bits, set subscript
                        )
                        result.append((sub_span, seg))
                    else:
                        result.append((span, seg))
                else:
                    result.append((span, seg))
                pos = seg_start + len(seg)

            # Any remaining text
            if pos < len(text):
                result.append((span, text[pos:]))

            # Update prev_last_char from the original text of this part
            if text:
                prev_last_char = text[-1]

        return result

    @staticmethod
    def _estimate_text_width(text: str, font_size: float) -> float:
        """Estimate text width in CDXML units using approximate Arial metrics.

        Average character width for Arial is roughly 0.52 * font_size.
        This is a heuristic — narrow chars (i, l, 1) are ~0.3x and
        wide chars (M, W) are ~0.8x, but 0.52 works well on average.
        """
        return len(text) * font_size * 0.52

    def _word_wrap_line(self, line_parts, max_width):
        """Word-wrap a single visual line (list of (span, text) pairs).

        Returns a list of visual lines, each a list of (span, text) pairs.
        Splits only at space boundaries within span text fragments.
        """
        # First estimate the total width of this line
        total_width = sum(
            self._estimate_text_width(text, span.size)
            for span, text in line_parts
        )
        if total_width <= max_width:
            return [line_parts]

        # Need to wrap. Flatten all (span, word) pairs, then re-assemble
        # into lines that fit within max_width.
        # Split each span's text on spaces, keeping the space at the end
        # of each word so spacing is preserved.
        word_units = []  # [(span, word_text), ...] where word_text may end with space
        for span, text in line_parts:
            words = text.split(' ')
            for wi, w in enumerate(words):
                if wi < len(words) - 1:
                    word_units.append((span, w + ' '))
                else:
                    if w:  # skip empty trailing fragment
                        word_units.append((span, w))

        result_lines = []
        current_line = []
        current_width = 0.0

        for span, word in word_units:
            w = self._estimate_text_width(word, span.size)
            if current_line and current_width + w > max_width:
                # Strip trailing space from last part of current line
                if current_line:
                    last_span, last_text = current_line[-1]
                    current_line[-1] = (last_span, last_text.rstrip(' '))
                result_lines.append(current_line)
                current_line = []
                current_width = 0.0
            current_line.append((span, word))
            current_width += w

        if current_line:
            # Strip trailing space
            last_span, last_text = current_line[-1]
            current_line[-1] = (last_span, last_text.rstrip(' '))
            result_lines.append(current_line)

        # Merge consecutive parts that belong to the same span
        merged = []
        for line in (result_lines or [line_parts]):
            merged_line = []
            for span, text in line:
                if merged_line and merged_line[-1][0] is span:
                    prev_span, prev_text = merged_line[-1]
                    merged_line[-1] = (prev_span, prev_text + text)
                else:
                    merged_line.append((span, text))
            merged.append(merged_line)

        return merged

    # ------------------------------------------------------------------ #
    #  Graphics
    # ------------------------------------------------------------------ #

    def _render_graphic(self, graphic: Graphic, doc: CDXMLDocument) -> list:
        """Render a graphic element."""
        if graphic.graphic_type != "Rectangle":
            return []

        if not graphic.center or not graphic.major_axis_end or not graphic.minor_axis_end:
            return []

        cx, cy = graphic.center
        mx, my = graphic.major_axis_end
        nxx, ny = graphic.minor_axis_end

        w = abs(mx - cx) * 2
        h = abs(ny - cy) * 2
        rx = cx - w / 2
        ry = cy - h / 2

        # Determine fill and stroke from rectangle type and color
        fill_color = "white"
        stroke_color = "rgb(0, 0, 0)"
        stroke_width = doc.line_width
        corner_radius = 1

        if graphic.color_idx:
            color = doc.get_color(graphic.color_idx)
            color_str = color.to_rgb()
        else:
            color_str = None

        is_filled = "Filled" in graphic.rectangle_type
        is_round = "RoundEdge" in graphic.rectangle_type

        if is_round:
            corner_radius = min(w, h) * 0.1  # 10% of smaller dimension

        if is_filled and color_str:
            fill_color = color_str
            stroke_color = "none"
            stroke_width = 0
        elif is_filled:
            fill_color = "#cccccc"
            stroke_color = "none"
            stroke_width = 0
        else:
            fill_color = "none"  # outline-only rectangle: transparent fill

        lines = []
        if graphic.rectangle_type == "Shadow":
            lines.append(
                f'  <rect x="{rx + 2:.2f}" y="{ry + 2:.2f}" '
                f'width="{w:.2f}" height="{h:.2f}" '
                f'fill="#cccccc" stroke="none" rx="2"/>'
            )
        lines.append(
            f'  <rect x="{rx:.2f}" y="{ry:.2f}" '
            f'width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill_color}" stroke="{stroke_color}" '
            f'stroke-width="{stroke_width:.2f}" rx="{corner_radius:.1f}"/>'
        )
        return lines

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _get_text_anchor(self, label: TextLabel) -> str:
        """Determine SVG text-anchor from label justification."""
        just = label.justification
        if just == "Center":
            return "middle"
        elif just == "Right":
            return "end"
        return "start"

    @staticmethod
    def _xml_escape(text: str) -> str:
        """Escape text for XML/SVG."""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("'", "&apos;")
                .replace('"', "&quot;"))
