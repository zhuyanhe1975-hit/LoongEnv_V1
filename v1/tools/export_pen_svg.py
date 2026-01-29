#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _norm_padding(padding: Any) -> Tuple[float, float, float, float]:
    if padding is None:
        return (0.0, 0.0, 0.0, 0.0)
    if isinstance(padding, (int, float)):
        v = float(padding)
        return (v, v, v, v)
    if isinstance(padding, list) and len(padding) == 2:
        v = float(padding[0])
        h = float(padding[1])
        return (v, h, v, h)
    if isinstance(padding, list) and len(padding) == 4:
        return (float(padding[0]), float(padding[1]), float(padding[2]), float(padding[3]))
    return (0.0, 0.0, 0.0, 0.0)


_SIZING_RE = re.compile(r"^(fit_content|fill_container)(?:\(([-0-9.]+)\))?$")


def _parse_sizing(value: Any) -> Tuple[str, Optional[float]] | Tuple[str, float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return ("fixed", float(value))
    if isinstance(value, str):
        m = _SIZING_RE.match(value)
        if m:
            kind = m.group(1)
            fb = float(m.group(2)) if m.group(2) is not None else None
            return (kind, fb)
    return None


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name or "page"


def _resolve_color(value: Any, varmap: Dict[str, Any]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        if value.startswith("$--"):
            key = value[1:]  # remove leading $
            v = varmap.get(key)
            if isinstance(v, dict) and "value" in v:
                return str(v["value"])
            if isinstance(v, str):
                return v
            return None
        return value
    if isinstance(value, dict) and value.get("type") == "color":
        if value.get("enabled") is False:
            return None
        return _resolve_color(value.get("color"), varmap)
    return None


def _stroke_info(stroke: Any, varmap: Dict[str, Any]) -> Tuple[Optional[str], float]:
    if not isinstance(stroke, dict):
        return (None, 0.0)
    color = _resolve_color(stroke.get("fill"), varmap)
    thickness = stroke.get("thickness", 0)
    if isinstance(thickness, (int, float)):
        return (color, float(thickness))
    if isinstance(thickness, dict):
        vals = [v for v in thickness.values() if isinstance(v, (int, float))]
        return (color, float(max(vals)) if vals else 0.0)
    return (color, 0.0)


def _corner_radius(value: Any, varmap: Dict[str, Any], w: float, h: float) -> float:
    if value is None:
        return 0.0
    if isinstance(value, str) and value.startswith("$--"):
        v = varmap.get(value[1:])
        if isinstance(v, dict) and isinstance(v.get("value"), (int, float)):
            value = v["value"]
    if isinstance(value, (int, float)):
        r = float(value)
        if r >= 999:
            return min(w, h) / 2.0
        return max(0.0, r)
    return 0.0


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float


def _text_metrics(text: str, font_size: float) -> Tuple[float, float]:
    # Rough heuristic; good enough for our UI wireframe export.
    lines = text.splitlines() or [""]
    longest = max((len(line) for line in lines), default=0)
    width = longest * font_size * 0.6
    height = len(lines) * font_size * 1.35
    return (width, height)


def _iter_nodes(root: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        children = n.get("children")
        if isinstance(children, list) and children:
            for c in reversed(children):
                if isinstance(c, dict):
                    stack.append(c)


def _build_id_map(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for child in doc.get("children", []):
        if isinstance(child, dict):
            for n in _iter_nodes(child):
                nid = n.get("id")
                if isinstance(nid, str):
                    out[nid] = n
    return out


def _clone_component_instance(
    instance: Dict[str, Any], components: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    instance_id = instance["id"]
    comp_id = instance["ref"]
    base = components.get(comp_id)
    if base is None:
        raise RuntimeError(f"Missing component ref: {comp_id}")

    base_copy = copy.deepcopy(base)
    base_copy["id"] = instance_id
    if "name" in instance:
        base_copy["name"] = instance["name"]

    # Apply top-level sizing/position overrides if present.
    for k in ("x", "y", "width", "height", "fill", "stroke", "cornerRadius", "layout", "gap", "padding"):
        if k in instance:
            base_copy[k] = instance[k]

    # Prefix all descendant ids with instance_id + "/".
    orig_to_new: Dict[str, Dict[str, Any]] = {}
    for n in _iter_nodes(base_copy):
        if n is base_copy:
            continue
        nid = n.get("id")
        if isinstance(nid, str):
            orig_to_new[nid] = n
            n["id"] = f"{instance_id}/{nid}"

    # Apply descendant overrides by original id.
    descendants = instance.get("descendants") or {}
    if isinstance(descendants, dict):
        for key, patch in descendants.items():
            if not isinstance(patch, dict):
                continue
            target_key = str(key).split("/")[-1]
            target = orig_to_new.get(target_key)
            if target is None:
                continue
            for pk, pv in patch.items():
                # Children overrides are not supported by this exporter.
                if pk == "children":
                    continue
                target[pk] = pv

    # Resolve nested refs inside component copy.
    def resolve_in_place(node: Dict[str, Any]) -> Dict[str, Any]:
        if node.get("type") == "ref":
            # For nested refs, keep the id-prefixing local to that nested instance.
            return _clone_component_instance(node, components)
        children = node.get("children")
        if isinstance(children, list):
            new_children: List[Dict[str, Any]] = []
            for c in children:
                if not isinstance(c, dict):
                    continue
                new_children.append(resolve_in_place(c))
            node["children"] = new_children
        return node

    return resolve_in_place(base_copy)


def _resolve_refs(root: Dict[str, Any], components: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if root.get("type") == "ref":
        return _clone_component_instance(root, components)
    children = root.get("children")
    if isinstance(children, list):
        out_children: List[Dict[str, Any]] = []
        for c in children:
            if not isinstance(c, dict):
                continue
            out_children.append(_resolve_refs(c, components))
        root = copy.deepcopy(root)
        root["children"] = out_children
    return root


def _compute_layout(
    node: Dict[str, Any],
    varmap: Dict[str, Any],
    origin: Box,
    parent_content: Optional[Box] = None,
) -> Dict[str, Box]:
    """
    Compute approximate absolute layout boxes for a node subtree.
    For `layout: none`, uses x/y. For flex, uses padding+gap and basic alignment.
    """
    layouts: Dict[str, Box] = {}

    def size_of(n: Dict[str, Any], available: Box) -> Tuple[float, float]:
        t = n.get("type")
        if t == "text":
            text = str(n.get("content") or "")
            fs = float(n.get("fontSize") or 14)
            w, h = _text_metrics(text, fs)
            return (w, h)

        w_spec = _parse_sizing(n.get("width"))
        h_spec = _parse_sizing(n.get("height"))

        w: float
        h: float
        if w_spec is None:
            w = 0.0
        elif w_spec[0] == "fixed":
            w = float(w_spec[1])  # type: ignore[index]
        elif w_spec[0] == "fill_container":
            w = available.w
        elif w_spec[0] == "fit_content":
            w = float(w_spec[1] or 0.0)  # type: ignore[index]
        else:
            w = 0.0

        if h_spec is None:
            h = 0.0
        elif h_spec[0] == "fixed":
            h = float(h_spec[1])  # type: ignore[index]
        elif h_spec[0] == "fill_container":
            h = available.h
        elif h_spec[0] == "fit_content":
            h = float(h_spec[1] or 0.0)  # type: ignore[index]
        else:
            h = 0.0

        # Defaults for shapes.
        if t in ("rectangle", "ellipse") and (w == 0.0 or h == 0.0):
            w = float(n.get("width") or w or 10.0)
            h = float(n.get("height") or h or 10.0)

        # If fit_content, compute from children for frames.
        if t == "frame":
            layout = n.get("layout")
            if isinstance(layout, str) and layout in ("vertical", "horizontal"):
                pad_t, pad_r, pad_b, pad_l = _norm_padding(n.get("padding"))
                gap = float(n.get("gap") or 0.0)
                children = [c for c in (n.get("children") or []) if isinstance(c, dict)]
                if w_spec is not None and w_spec[0] == "fit_content":
                    if layout == "vertical":
                        child_ws = []
                        for c in children:
                            cw, _ch = size_of(c, Box(0, 0, 0, 0))
                            child_ws.append(cw)
                        w = pad_l + (max(child_ws) if child_ws else 0.0) + pad_r
                    else:
                        total_w = 0.0
                        for idx, c in enumerate(children):
                            cw, _ch = size_of(c, Box(0, 0, 0, 0))
                            total_w += cw
                            if idx > 0:
                                total_w += gap
                        w = pad_l + total_w + pad_r
                if h_spec is not None and h_spec[0] == "fit_content":
                    if layout == "vertical":
                        total_h = 0.0
                        for idx, c in enumerate(children):
                            _cw, ch = size_of(c, Box(0, 0, 0, 0))
                            total_h += ch
                            if idx > 0:
                                total_h += gap
                        h = pad_t + total_h + pad_b
                    else:
                        child_hs = []
                        for c in children:
                            _cw, ch = size_of(c, Box(0, 0, 0, 0))
                            child_hs.append(ch)
                        h = pad_t + (max(child_hs) if child_hs else 0.0) + pad_b

        # Fallback: if size is still 0, use available for fill_container parents.
        if w == 0.0 and w_spec is not None and w_spec[0] == "fill_container":
            w = available.w
        if h == 0.0 and h_spec is not None and h_spec[0] == "fill_container":
            h = available.h
        return (w, h)

    def layout_children(n: Dict[str, Any], box: Box) -> None:
        nid = n.get("id")
        if isinstance(nid, str):
            layouts[nid] = box

        if n.get("type") != "frame":
            return

        children = [c for c in (n.get("children") or []) if isinstance(c, dict)]
        if not children:
            return

        layout = n.get("layout") or "none"
        pad_t, pad_r, pad_b, pad_l = _norm_padding(n.get("padding"))
        gap = float(n.get("gap") or 0.0)
        content = Box(box.x + pad_l, box.y + pad_t, max(0.0, box.w - pad_l - pad_r), max(0.0, box.h - pad_t - pad_b))

        if layout == "none":
            for c in children:
                cx = float(c.get("x") or 0.0)
                cy = float(c.get("y") or 0.0)
                cw, ch = size_of(c, content)
                cbox = Box(content.x + cx, content.y + cy, cw, ch)
                layout_children(c, cbox)
            return

        if layout == "horizontal":
            # Measure fixed widths first.
            child_sizes: List[Tuple[Dict[str, Any], float, float]] = []
            for c in children:
                cw, ch = size_of(c, Box(0, 0, content.w, content.h))
                child_sizes.append((c, cw, ch))

            total_w = sum(cw for _c, cw, _ch in child_sizes)
            total_w += gap * max(0, len(child_sizes) - 1)

            justify = n.get("justifyContent") or "start"
            x_cursor = content.x
            if justify == "center":
                x_cursor = content.x + max(0.0, (content.w - total_w) / 2.0)
            elif justify == "end":
                x_cursor = content.x + max(0.0, content.w - total_w)
            elif justify == "space_between" and len(child_sizes) > 1:
                gap = (content.w - sum(cw for _c, cw, _ch in child_sizes)) / (len(child_sizes) - 1)
                gap = max(0.0, gap)

            align = n.get("alignItems") or "start"
            for c, cw, ch in child_sizes:
                cy = content.y
                if align == "center":
                    cy = content.y + max(0.0, (content.h - ch) / 2.0)
                elif align == "end":
                    cy = content.y + max(0.0, content.h - ch)
                cbox = Box(x_cursor, cy, cw, ch)
                layout_children(c, cbox)
                x_cursor += cw + gap
            return

        if layout == "vertical":
            child_sizes = []
            for c in children:
                cw, ch = size_of(c, Box(0, 0, content.w, content.h))
                child_sizes.append((c, cw, ch))

            total_h = sum(ch for _c, _cw, ch in child_sizes)
            total_h += gap * max(0, len(child_sizes) - 1)

            justify = n.get("justifyContent") or "start"
            y_cursor = content.y
            if justify == "center":
                y_cursor = content.y + max(0.0, (content.h - total_h) / 2.0)
            elif justify == "end":
                y_cursor = content.y + max(0.0, content.h - total_h)
            elif justify == "space_between" and len(child_sizes) > 1:
                gap = (content.h - sum(ch for _c, _cw, ch in child_sizes)) / (len(child_sizes) - 1)
                gap = max(0.0, gap)

            align = n.get("alignItems") or "start"
            for c, cw, ch in child_sizes:
                cx = content.x
                if align == "center":
                    cx = content.x + max(0.0, (content.w - cw) / 2.0)
                elif align == "end":
                    cx = content.x + max(0.0, content.w - cw)
                cbox = Box(cx, y_cursor, cw if _parse_sizing(c.get("width")) != ("fill_container", None) else content.w, ch)
                # If child is fill_container width, stretch.
                wspec = _parse_sizing(c.get("width"))
                if isinstance(wspec, tuple) and wspec[0] == "fill_container":
                    cbox.w = content.w
                layout_children(c, cbox)
                y_cursor += ch + gap
            return

    # Determine node box.
    available = parent_content or origin
    w, h = size_of(node, available)
    # Root node position: if it's at origin, clamp to origin; otherwise respect x/y.
    if parent_content is None:
        x = origin.x
        y = origin.y
    else:
        x = origin.x
        y = origin.y
    box = Box(x, y, w or origin.w, h or origin.h)
    layout_children(node, box)
    return layouts


def _render_svg(
    root: Dict[str, Any],
    layouts: Dict[str, Box],
    varmap: Dict[str, Any],
    width: float,
    height: float,
) -> str:
    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" height="{height:g}" viewBox="0 0 {width:g} {height:g}">')
    parts.append('<rect x="0" y="0" width="100%" height="100%" fill="transparent"/>')

    def draw(node: Dict[str, Any]) -> None:
        nid = node.get("id")
        if not isinstance(nid, str):
            return
        box = layouts.get(nid)
        if box is None:
            # Some fit_content wrappers may be missed; skip.
            return

        t = node.get("type")
        if t in ("frame", "rectangle"):
            fill = _resolve_color(node.get("fill"), varmap)
            stroke_color, stroke_w = _stroke_info(node.get("stroke"), varmap)
            rx = _corner_radius(node.get("cornerRadius"), varmap, box.w, box.h)
            if fill or (stroke_color and stroke_w > 0):
                fill_attr = fill if fill else "none"
                stroke_attr = stroke_color if (stroke_color and stroke_w > 0) else "none"
                sw_attr = f'{stroke_w:g}' if (stroke_color and stroke_w > 0) else "0"
                parts.append(
                    f'<rect x="{box.x:g}" y="{box.y:g}" width="{box.w:g}" height="{box.h:g}" rx="{rx:g}" ry="{rx:g}" fill="{fill_attr}" stroke="{stroke_attr}" stroke-width="{sw_attr}"/>'
                )
        elif t == "ellipse":
            fill = _resolve_color(node.get("fill"), varmap) or "none"
            stroke_color, stroke_w = _stroke_info(node.get("stroke"), varmap)
            cx = box.x + box.w / 2.0
            cy = box.y + box.h / 2.0
            rx = box.w / 2.0
            ry = box.h / 2.0
            stroke_attr = stroke_color if (stroke_color and stroke_w > 0) else "none"
            sw_attr = f'{stroke_w:g}' if (stroke_color and stroke_w > 0) else "0"
            parts.append(
                f'<ellipse cx="{cx:g}" cy="{cy:g}" rx="{rx:g}" ry="{ry:g}" fill="{fill}" stroke="{stroke_attr}" stroke-width="{sw_attr}"/>'
            )
        elif t == "text":
            content = str(node.get("content") or "")
            if not content:
                return
            fill = _resolve_color(node.get("fill"), varmap) or "#FFFFFF"
            font_family = _escape_xml(str(node.get("fontFamily") or "Inter"))
            font_size = float(node.get("fontSize") or 14)
            font_weight = str(node.get("fontWeight") or "normal")
            # Baseline heuristic: use top + font_size for baseline.
            x = box.x
            y = box.y + font_size
            parts.append(
                f'<text x="{x:g}" y="{y:g}" fill="{fill}" font-family="{font_family}" font-size="{font_size:g}" font-weight="{_escape_xml(font_weight)}">{_escape_xml(content)}</text>'
            )

        children = node.get("children")
        if isinstance(children, list):
            for c in children:
                if isinstance(c, dict):
                    draw(c)

    draw(root)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _find_screens(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    children = doc.get("children") or []
    for c in children:
        if not isinstance(c, dict):
            continue
        if c.get("type") == "frame" and c.get("name") == "Robot Motion Control UI (Pencil)":
            root = c
            break
    else:
        root = children[0] if children and isinstance(children[0], dict) else None
    if not isinstance(root, dict):
        return []

    # Find a frame named "Screens" under root (shallow search).
    for n in _iter_nodes(root):
        if n.get("type") == "frame" and n.get("name") == "Screens":
            screens = n.get("children") or []
            return [s for s in screens if isinstance(s, dict) and s.get("type") == "frame"]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description="Export screens from a .pen file to SVG (approximate).")
    ap.add_argument("pen", type=Path, help="Path to .pen file (JSON).")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for SVGs.")
    args = ap.parse_args()

    doc = json.loads(args.pen.read_text(encoding="utf-8"))
    varmap = doc.get("variables") or {}

    id_map = _build_id_map(doc)
    components = {nid: n for nid, n in id_map.items() if n.get("reusable") is True}

    screens = _find_screens(doc)
    if not screens:
        raise SystemExit("No screens found (expected a frame named 'Screens').")

    args.out.mkdir(parents=True, exist_ok=True)

    for screen in screens:
        resolved = _resolve_refs(screen, components)
        w = float(resolved.get("width") or 1440)
        h = float(resolved.get("height") or 900)
        layouts = _compute_layout(resolved, varmap, origin=Box(0, 0, w, h))
        svg = _render_svg(resolved, layouts, varmap, width=w, height=h)
        name = str(resolved.get("name") or resolved.get("id") or "screen")
        out_path = args.out / f"{_slugify(name)}.svg"
        out_path.write_text(svg, encoding="utf-8")
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

