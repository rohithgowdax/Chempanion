"""
CDXML Parser - Parses ChemDraw CDXML files into an intermediate representation.

Handles:
- Color tables
- Font tables
- Fragments (molecule groups)
- Nodes (atoms) with element types, positions, labels
- Bonds with order, display type (wedge, hash, wavy, etc.)
- Reaction arrows
- Text annotations
- Graphic elements (rectangles, etc.)
- Groups and nested fragments
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional


# Map of element numbers to symbols
ELEMENT_MAP = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    11: "Na", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I",
}


@dataclass
class Color:
    r: float
    g: float
    b: float

    def to_hex(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(
            int(self.r * 255), int(self.g * 255), int(self.b * 255)
        )

    def to_rgb(self) -> str:
        return f"rgb({int(self.r*255)},{int(self.g*255)},{int(self.b*255)})"


@dataclass
class Font:
    id: str
    name: str
    charset: str = "iso-8859-1"


@dataclass
class TextSpan:
    text: str
    font_id: str = "3"
    size: float = 10.0
    color_idx: int = 0
    face: int = 0  # bitmask: 1=bold, 2=italic, 32=subscript, 64=superscript, 96=both?

    @property
    def is_bold(self) -> bool:
        return bool(self.face & 1)

    @property
    def is_italic(self) -> bool:
        return bool(self.face & 2)

    @property
    def is_subscript(self) -> bool:
        # face=32 (bit 5 only) for explicit subscript
        # face=96 (bits 5+6) is formula mode, NOT subscript
        return bool(self.face & 32) and not bool(self.face & 64)

    @property
    def is_superscript(self) -> bool:
        return bool(self.face & 64) and not bool(self.face & 32)

    @property
    def is_formula(self) -> bool:
        """Formula mode (face bits 5+6 both set = 96).

        In formula mode, digits following letters are auto-subscripted.
        Can combine with bold (97) or italic (98).
        """
        return bool(self.face & 32) and bool(self.face & 64)


@dataclass
class TextLabel:
    id: str
    position: tuple  # (x, y)
    bounding_box: Optional[tuple] = None  # (x1, y1, x2, y2)
    spans: list = field(default_factory=list)
    justification: str = "Left"
    caption_justification: str = ""
    label_alignment: str = ""  # Left, Right, Above, Below
    line_starts: list = field(default_factory=list)  # character positions for line breaks
    line_height: str = "auto"
    word_wrap_width: float = 0.0  # max width before word-wrap (CDXML units)
    warning: str = ""
    color_idx: int = 0

    def get_text(self) -> str:
        return "".join(s.text for s in self.spans)


@dataclass
class Node:
    id: str
    position: tuple  # (x, y)
    element: int = 6  # default carbon
    num_hydrogens: int = -1  # -1 means implicit
    label: Optional[TextLabel] = None
    color_idx: int = 0
    node_type: str = ""  # Fragment, Nickname, GenericNickname, etc.
    z_order: int = 0
    charge: int = 0
    has_label_text: bool = False  # whether the node has visible label text


@dataclass
class Bond:
    id: str
    begin_id: str
    end_id: str
    order: float = 1
    display: str = ""  # WedgedHashBegin, WedgeBegin, Wavy, Bold, Dash, etc.
    color_idx: int = 0
    z_order: int = 0
    bond_circular_ordering: list = field(default_factory=list)  # non-empty if in ring


@dataclass
class Arrow:
    id: str
    head: tuple  # (x, y)
    tail: tuple  # (x, y)
    bounding_box: Optional[tuple] = None
    arrowhead_head: str = ""        # Full, HalfLeft, HalfRight, Angle, "" = none
    arrowhead_tail: str = ""         # same options (for equilibrium tail)
    arrowhead_type: str = "Solid"    # Solid, Hollow, Angle
    head_size: int = 1000            # overall size factor (1000 = 1x)
    head_center_size: int = 875      # indent for filled arrowheads
    head_width: int = 250            # half-width of arrowhead
    shaft_spacing: int = 0           # for equilibrium double-shaft
    nogo: str = ""                   # "Cross" for does-not-proceed
    line_type: str = ""              # "" = solid, "Dashed", "Wavy", etc.
    color_idx: int = 0               # color table index (0 = foreground/black)
    z_order: int = 0
    fill_type: str = "None"


@dataclass
class Graphic:
    id: str
    graphic_type: str = ""
    bounding_box: Optional[tuple] = None
    rectangle_type: str = ""
    center: Optional[tuple] = None
    major_axis_end: Optional[tuple] = None
    minor_axis_end: Optional[tuple] = None
    color_idx: int = 0
    z_order: int = 0


@dataclass
class CDXMLDocument:
    bounding_box: tuple = (0, 0, 540, 720)
    colors: list = field(default_factory=list)
    fonts: list = field(default_factory=list)
    nodes: dict = field(default_factory=dict)  # id -> Node
    bonds: list = field(default_factory=list)
    arrows: list = field(default_factory=list)
    texts: list = field(default_factory=list)
    graphics: list = field(default_factory=list)
    line_width: float = 0.6
    bold_width: float = 2.0
    bond_length: float = 14.4
    bond_spacing: float = 18.0
    hash_spacing: float = 2.5
    margin_width: float = 1.6
    label_font: str = "3"
    label_size: float = 10.0
    caption_size: float = 10.0

    def get_color(self, idx: int) -> Color:
        """Get color by CDXML color attribute value.

        In CDXML, color attribute values use a special mapping:
        - 0 = foreground (maps to colortable[1], typically black)
        - 1 = background (maps to colortable[0], typically white)
        - N >= 2 = colortable[N-2] (first two colortable entries are bg/fg)
        """
        if idx == 0:
            # Foreground = colortable[1] (black)
            if len(self.colors) > 1:
                return self.colors[1]
            return Color(0, 0, 0)
        elif idx == 1:
            # Background = colortable[0] (white)
            if len(self.colors) > 0:
                return self.colors[0]
            return Color(1, 1, 1)
        elif idx >= 2 and (idx - 2) < len(self.colors):
            return self.colors[idx - 2]
        return Color(0, 0, 0)  # default black


class CDXMLParser:
    """Parses a CDXML file into a CDXMLDocument."""

    def parse(self, filepath: str) -> CDXMLDocument:
        tree = ET.parse(filepath)
        root = tree.getroot()
        doc = CDXMLDocument()

        self._parse_document_attrs(root, doc)
        self._parse_color_table(root, doc)
        self._parse_font_table(root, doc)

        # Parse page contents
        for page in root.iter("page"):
            self._parse_page(page, doc)

        return doc

    def parse_string(self, xml_string: str) -> CDXMLDocument:
        root = ET.fromstring(xml_string)
        doc = CDXMLDocument()

        self._parse_document_attrs(root, doc)
        self._parse_color_table(root, doc)
        self._parse_font_table(root, doc)

        for page in root.iter("page"):
            self._parse_page(page, doc)

        return doc

    def _parse_document_attrs(self, root, doc: CDXMLDocument):
        bb = root.get("BoundingBox", "")
        if bb:
            parts = [float(x) for x in bb.split()]
            if len(parts) == 4:
                doc.bounding_box = tuple(parts)

        doc.line_width = float(root.get("LineWidth", "0.6"))
        doc.bold_width = float(root.get("BoldWidth", "2"))
        doc.bond_length = float(root.get("BondLength", "14.4"))
        doc.bond_spacing = float(root.get("BondSpacing", "18"))
        doc.hash_spacing = float(root.get("HashSpacing", "2.5"))
        doc.margin_width = float(root.get("MarginWidth", "1.6"))
        doc.label_font = root.get("LabelFont", "3")
        doc.label_size = float(root.get("LabelSize", "10"))
        doc.caption_size = float(root.get("CaptionSize", "10"))

    def _parse_color_table(self, root, doc: CDXMLDocument):
        for ct in root.iter("colortable"):
            for c in ct.findall("color"):
                doc.colors.append(Color(
                    r=float(c.get("r", "0")),
                    g=float(c.get("g", "0")),
                    b=float(c.get("b", "0")),
                ))

    def _parse_font_table(self, root, doc: CDXMLDocument):
        for ft in root.iter("fonttable"):
            for f in ft.findall("font"):
                doc.fonts.append(Font(
                    id=f.get("id", ""),
                    name=f.get("name", "Arial"),
                    charset=f.get("charset", "iso-8859-1"),
                ))

    def _parse_page(self, page, doc: CDXMLDocument):
        """Recursively parse all elements within a page."""
        self._parse_elements(page, doc)

    def _parse_elements(self, parent, doc: CDXMLDocument):
        """Parse child elements of a container (page, group, fragment)."""
        for child in parent:
            tag = child.tag
            if tag == "fragment":
                self._parse_fragment(child, doc)
            elif tag == "group":
                # Groups contain fragments
                self._parse_elements(child, doc)
            elif tag == "n":
                self._parse_node(child, doc)
            elif tag == "b":
                self._parse_bond(child, doc)
            elif tag == "t":
                self._parse_text(child, doc)
            elif tag == "arrow":
                self._parse_arrow(child, doc)
            elif tag == "graphic":
                self._parse_graphic(child, doc)
            elif tag in ("scheme", "step", "colortable", "fonttable",
                         "chemicalproperty"):
                # Skip meta elements
                pass

    def _parse_fragment(self, frag, doc: CDXMLDocument):
        """Parse a fragment (molecule) and its children."""
        for child in frag:
            tag = child.tag
            if tag == "n":
                self._parse_node(child, doc)
            elif tag == "b":
                self._parse_bond(child, doc)
            elif tag == "t":
                self._parse_text(child, doc)

    def _parse_node(self, node_elem, doc: CDXMLDocument):
        """Parse an atom node."""
        node_id = node_elem.get("id", "")
        pos_str = node_elem.get("p", "")

        if not pos_str:
            return

        pos = tuple(float(x) for x in pos_str.split())
        if len(pos) != 2:
            return

        element = int(node_elem.get("Element", "6"))
        num_h = int(node_elem.get("NumHydrogens", "-1"))
        color_idx = int(node_elem.get("color", "0"))
        node_type = node_elem.get("NodeType", "")
        z_order = int(node_elem.get("Z", "0"))
        charge = int(node_elem.get("Charge", "0"))

        # Look for text label
        label = None
        has_label_text = False
        for t_elem in node_elem.findall("t"):
            label = self._parse_text_element(t_elem)
            has_label_text = True
            break

        # Check for nested fragment (nickname expansions) - we only want the label
        node = Node(
            id=node_id,
            position=pos,
            element=element,
            num_hydrogens=num_h,
            label=label,
            color_idx=color_idx,
            node_type=node_type,
            z_order=z_order,
            charge=charge,
            has_label_text=has_label_text,
        )

        doc.nodes[node_id] = node

    def _parse_bond(self, bond_elem, doc: CDXMLDocument):
        """Parse a bond element."""
        bond_id = bond_elem.get("id", "")
        begin = bond_elem.get("B", "")
        end = bond_elem.get("E", "")
        order = float(bond_elem.get("Order", "1"))
        display = bond_elem.get("Display", "")
        color_idx = int(bond_elem.get("color", "0"))
        z_order = int(bond_elem.get("Z", "0"))

        # Parse BondCircularOrdering (present for ring bonds)
        bco_str = bond_elem.get("BondCircularOrdering", "")
        bco = bco_str.split() if bco_str else []

        # Skip bonds to ExternalConnectionPoint nodes
        if not begin or not end:
            return

        doc.bonds.append(Bond(
            id=bond_id,
            begin_id=begin,
            end_id=end,
            order=order,
            display=display,
            color_idx=color_idx,
            z_order=z_order,
            bond_circular_ordering=bco,
        ))

    def _parse_text(self, text_elem, doc: CDXMLDocument):
        """Parse a standalone text element (caption/annotation)."""
        label = self._parse_text_element(text_elem)
        if label and label.get_text().strip():
            doc.texts.append(label)

    def _parse_text_element(self, t_elem) -> Optional[TextLabel]:
        """Parse a <t> element with <s> children into a TextLabel."""
        t_id = t_elem.get("id", "")
        pos_str = t_elem.get("p", "")
        bb_str = t_elem.get("BoundingBox", "")
        justification = t_elem.get("Justification", "Left")
        caption_just = t_elem.get("CaptionJustification", "")
        label_alignment = t_elem.get("LabelAlignment", "")
        line_starts_str = t_elem.get("LineStarts", "")
        line_starts = [int(x) for x in line_starts_str.split()] if line_starts_str else []
        line_height_str = t_elem.get("LineHeight", "auto")
        word_wrap_width = float(t_elem.get("WordWrapWidth", "0"))
        color_idx = int(t_elem.get("color", "0"))

        pos = (0, 0)
        if pos_str:
            parts = pos_str.split()
            if len(parts) >= 2:
                pos = (float(parts[0]), float(parts[1]))

        bb = None
        if bb_str:
            parts = [float(x) for x in bb_str.split()]
            if len(parts) == 4:
                bb = tuple(parts)

        spans = []
        for s_elem in t_elem.findall("s"):
            text = s_elem.text or ""
            # Also get tail text
            if s_elem.tail:
                text += s_elem.tail
            font_id = s_elem.get("font", "3")
            size = float(s_elem.get("size", "10"))
            s_color = int(s_elem.get("color", "0"))
            face = int(s_elem.get("face", "0"))

            spans.append(TextSpan(
                text=text,
                font_id=font_id,
                size=size,
                color_idx=s_color,
                face=face,
            ))

        if not spans:
            # Element might have direct text
            if t_elem.text and t_elem.text.strip():
                spans.append(TextSpan(text=t_elem.text))

        if not spans:
            return None

        return TextLabel(
            id=t_id,
            position=pos,
            bounding_box=bb,
            spans=spans,
            justification=justification,
            caption_justification=caption_just,
            label_alignment=label_alignment,
            line_starts=line_starts,
            line_height=line_height_str,
            word_wrap_width=word_wrap_width,
            color_idx=color_idx,
        )

    def _parse_arrow(self, arrow_elem, doc: CDXMLDocument):
        """Parse a reaction arrow."""
        arrow_id = arrow_elem.get("id", "")
        z_order = int(arrow_elem.get("Z", "0"))
        arrowhead_head = arrow_elem.get("ArrowheadHead", "")
        arrowhead_tail = arrow_elem.get("ArrowheadTail", "")
        arrowhead_type = arrow_elem.get("ArrowheadType", "Solid")
        head_size = int(arrow_elem.get("HeadSize", "1000"))
        head_center_size = int(arrow_elem.get("ArrowheadCenterSize", str(int(head_size * 0.875))))
        head_width = int(arrow_elem.get("ArrowheadWidth", "250"))
        shaft_spacing = int(arrow_elem.get("ArrowShaftSpacing", "0"))
        nogo = arrow_elem.get("NoGo", "")
        line_type = arrow_elem.get("LineType", "")
        color_idx = int(arrow_elem.get("color", "0"))
        fill_type = arrow_elem.get("FillType", "None")

        head_str = arrow_elem.get("Head3D", "")
        tail_str = arrow_elem.get("Tail3D", "")

        if not head_str or not tail_str:
            return

        head_parts = [float(x) for x in head_str.split()]
        tail_parts = [float(x) for x in tail_str.split()]

        head = (head_parts[0], head_parts[1])
        tail = (tail_parts[0], tail_parts[1])

        bb_str = arrow_elem.get("BoundingBox", "")
        bb = None
        if bb_str:
            parts = [float(x) for x in bb_str.split()]
            if len(parts) == 4:
                bb = tuple(parts)

        doc.arrows.append(Arrow(
            id=arrow_id,
            head=head,
            tail=tail,
            bounding_box=bb,
            arrowhead_head=arrowhead_head,
            arrowhead_tail=arrowhead_tail,
            arrowhead_type=arrowhead_type,
            head_size=head_size,
            head_center_size=head_center_size,
            head_width=head_width,
            shaft_spacing=shaft_spacing,
            nogo=nogo,
            line_type=line_type,
            color_idx=color_idx,
            z_order=z_order,
            fill_type=fill_type,
        ))

    def _parse_graphic(self, graphic_elem, doc: CDXMLDocument):
        """Parse a graphic element (rectangle, etc.)."""
        g_id = graphic_elem.get("id", "")
        g_type = graphic_elem.get("GraphicType", "")
        rect_type = graphic_elem.get("RectangleType", "")
        z_order = int(graphic_elem.get("Z", "0"))

        bb_str = graphic_elem.get("BoundingBox", "")
        bb = None
        if bb_str:
            parts = [float(x) for x in bb_str.split()]
            if len(parts) == 4:
                bb = tuple(parts)

        center = None
        center_str = graphic_elem.get("Center3D", "")
        if center_str:
            parts = [float(x) for x in center_str.split()]
            center = (parts[0], parts[1])

        major = None
        major_str = graphic_elem.get("MajorAxisEnd3D", "")
        if major_str:
            parts = [float(x) for x in major_str.split()]
            major = (parts[0], parts[1])

        minor = None
        minor_str = graphic_elem.get("MinorAxisEnd3D", "")
        if minor_str:
            parts = [float(x) for x in minor_str.split()]
            minor = (parts[0], parts[1])

        color_idx = int(graphic_elem.get("color", "0"))

        doc.graphics.append(Graphic(
            id=g_id,
            graphic_type=g_type,
            bounding_box=bb,
            rectangle_type=rect_type,
            center=center,
            major_axis_end=major,
            minor_axis_end=minor,
            color_idx=color_idx,
            z_order=z_order,
        ))
