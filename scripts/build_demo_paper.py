#!/usr/bin/env python3
"""Build the synthetic AgenQA demo paper PDF from its Markdown source."""

from __future__ import annotations

from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import CondPageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/papers/layered_thermal_transport_demo.md"
OUTPUT = ROOT / "examples/papers/layered_thermal_transport_demo.pdf"


def _footer(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(0.72 * inch, 0.58 * inch, 7.78 * inch, 0.58 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(0.72 * inch, 0.38 * inch, "AgenQA synthetic public demo paper")
    canvas.drawRightString(7.78 * inch, 0.38 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\b(R1|R2|Ti|Th|Tc|Q'|Q|q|A|L1|L2|k1|k2)\b", r"<i>\1</i>", text)
    return text


def build() -> Path:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "PaperTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0F172A"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER,
        spaceAfter=20,
    )
    h1 = ParagraphStyle(
        "Section",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0F4C81"),
        spaceBefore=12,
        spaceAfter=7,
        keepWithNext=0,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Times-Roman",
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#172033"),
        spaceAfter=8,
    )
    equation = ParagraphStyle(
        "Equation",
        parent=body,
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        leftIndent=24,
        rightIndent=24,
        borderColor=colors.HexColor("#BFDBFE"),
        borderWidth=0.6,
        borderPadding=8,
        backColor=colors.HexColor("#EFF6FF"),
        spaceBefore=4,
        spaceAfter=10,
    )

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = []
    paragraphs: list[str] = []

    def flush() -> None:
        if paragraphs:
            text = " ".join(part.strip() for part in paragraphs).strip()
            if text:
                is_equation = bool(re.match(r"^(R1|Q =|Ti =)", text)) and len(text) < 120
                story.append(Paragraph(_inline(text), equation if is_equation else body))
            paragraphs.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            flush()
            story.append(Spacer(1, 0.42 * inch))
            story.append(Paragraph(_inline(stripped[2:]), title))
        elif stripped.startswith("## A synthetic"):
            flush()
            story.append(Paragraph(_inline(stripped[3:]), subtitle))
            data = [
                ["Artifact type", "Synthetic scientific paper"],
                ["Purpose", "End-to-end AgenQA PDF input"],
                ["Distribution", "Authored for this public repository"],
            ]
            table = Table(data, colWidths=[1.35 * inch, 4.75 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E2E8F0")),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1E293B")),
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 0.18 * inch))
        elif stripped.startswith("## "):
            flush()
            story.append(CondPageBreak(0.72 * inch))
            story.append(Paragraph(_inline(stripped[3:]), h1))
        elif not stripped:
            flush()
        else:
            paragraphs.append(stripped)
    flush()

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.66 * inch,
        bottomMargin=0.76 * inch,
        title="Thermal Transport Through a Two-Layer Composite Rod",
        author="AgenQA public demo",
        subject="Synthetic scientific paper for end-to-end QA synthesis",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return OUTPUT


if __name__ == "__main__":
    print(build())
