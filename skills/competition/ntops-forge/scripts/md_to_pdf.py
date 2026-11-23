#!/usr/bin/env python3
"""Convert markdown report to PDF (ReportLab, Chinese + tables + images)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as RLImage,
)
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

FONT_PATH = Path("C:/Windows/Fonts/simhei.ttf")
FONT = "SimHei"
PAGE_W, PAGE_H = A4
CONTENT_W = PAGE_W - 3.6 * cm


def _register_font() -> None:
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"Font not found: {FONT_PATH}")
    pdfmetrics.registerFont(TTFont(FONT, str(FONT_PATH)))


def _esc(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = text.replace("`", "")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    return text


def _styles() -> dict[str, ParagraphStyle]:
    base = dict(fontName=FONT, alignment=TA_LEFT)
    return {
        "title": ParagraphStyle(
            "title", fontSize=18, leading=24, spaceAfter=10, **base
        ),
        "meta": ParagraphStyle(
            "meta", fontSize=10, leading=14, textColor=colors.HexColor("#333"), **base
        ),
        "h2": ParagraphStyle(
            "h2", fontSize=14, leading=20, spaceBefore=14, spaceAfter=8, **base
        ),
        "h3": ParagraphStyle(
            "h3", fontSize=11, leading=16, spaceBefore=8, spaceAfter=4, **base
        ),
        "body": ParagraphStyle("body", fontSize=10, leading=15, spaceAfter=6, **base),
        "caption": ParagraphStyle(
            "caption",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#555"),
            spaceAfter=8,
            **base,
        ),
        "code": ParagraphStyle(
            "code",
            fontSize=8,
            leading=11,
            backColor=colors.HexColor("#f5f5f5"),
            leftIndent=6,
            spaceAfter=6,
            **base,
        ),
    }


def _parse_table_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_table_sep(line: str) -> bool:
    return bool(re.match(r"^\|[\s\-:|]+\|\s*$", line))


def _make_image(
    path: Path, max_w: float = CONTENT_W, max_h: float = 8.5 * cm
) -> RLImage:
    with PILImage.open(path) as im:
        w, h = im.size
    ratio = h / w if w else 1
    width = max_w
    height = width * ratio
    if height > max_h:
        height = max_h
        width = height / ratio if ratio else max_w
    return RLImage(str(path), width=width, height=height)


def _table_flowable(rows: list[list[str]], st: dict[str, ParagraphStyle]) -> Table:
    ncols = len(rows[0])
    col_w = CONTENT_W / ncols
    data = [[Paragraph(_esc(c), st["body"]) for c in row] for row in rows]
    tbl = Table(data, colWidths=[col_w] * ncols, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), FONT, 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def convert(md_path: Path, out_path: Path) -> int:
    _register_font()
    st = _styles()
    base_dir = md_path.parent
    story: list = []
    img_count = 0

    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    in_code = False
    code_buf: list[str] = []
    first_h1 = True

    while i < len(lines):
        line = lines[i].rstrip()

        if line.startswith("```"):
            if in_code:
                story.append(Paragraph(_esc("\n".join(code_buf)), st["code"]))
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # table block
        if line.startswith("|"):
            table_rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not _is_table_sep(lines[i]):
                    table_rows.append(_parse_table_row(lines[i]))
                i += 1
            if table_rows:
                story.append(Spacer(1, 4))
                story.append(_table_flowable(table_rows, st))
                story.append(Spacer(1, 8))
            continue

        m_img = re.match(r"^!\[(.*?)\]\((.+?)\)\s*$", line)
        if m_img:
            alt, rel = m_img.group(1), m_img.group(2)
            img_path = (base_dir / rel).resolve()
            story.append(PageBreak())
            if img_path.is_file():
                try:
                    story.append(_make_image(img_path))
                    story.append(Spacer(1, 4))
                    story.append(Paragraph(_esc(alt), st["caption"]))
                    img_count += 1
                except Exception as exc:
                    story.append(
                        Paragraph(f"[图片渲染失败: {rel} - {exc}]", st["body"])
                    )
            else:
                story.append(Paragraph(f"[图片缺失: {rel}]", st["body"]))
            i += 1
            continue

        m_cap = re.match(r"^\*\*(图\d+[^*]*)\*\*\s*(.*)$", line)
        if m_cap:
            story.append(
                Paragraph(_esc(f"{m_cap.group(1)} {m_cap.group(2)}"), st["caption"])
            )
            i += 1
            continue

        if line.startswith("# "):
            if not first_h1:
                story.append(PageBreak())
            first_h1 = False
            story.append(Paragraph(_esc(line[2:].strip()), st["title"]))
            i += 1
            continue

        if line.startswith("## "):
            story.append(Spacer(1, 6))
            story.append(Paragraph(_esc(line[3:].strip()), st["h2"]))
            i += 1
            continue

        if line.startswith("### "):
            story.append(Paragraph(_esc(line[4:].strip()), st["h3"]))
            i += 1
            continue

        if line.startswith("> "):
            story.append(Paragraph(_esc(line[2:].strip()), st["meta"]))
            i += 1
            continue

        if line.strip() == "---":
            story.append(Spacer(1, 8))
            i += 1
            continue

        if line.startswith("- "):
            story.append(Paragraph(f"&bull; {_esc(line[2:].strip())}", st["body"]))
            i += 1
            continue

        if re.match(r"^\d+\.\s", line):
            story.append(Paragraph(_esc(line.strip()), st["body"]))
            i += 1
            continue

        if line.strip():
            story.append(Paragraph(_esc(line.strip()), st["body"]))

        i += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="中期报告",
        author="于鸿伟",
    )

    def _footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont(FONT, 9)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(PAGE_W / 2, 1.2 * cm, f"- {canvas.getPageNumber()} -")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return img_count


def main() -> None:
    p = argparse.ArgumentParser(description="Convert markdown report to PDF")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    n = convert(args.input, args.output)
    print(f"Created {args.output} ({n} images embedded)")


if __name__ == "__main__":
    main()
