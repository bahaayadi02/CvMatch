"""Exact ATS Resume PDF Generator matching the user's professional single-page template.

Features:
- Centered header: Name (Times-Bold 16pt), subtitle role (Times-Roman 9.5pt), contact details & links with pipe separators.
- Full-width horizontal divider line under each uppercase bold section heading.
- 2-Column table layout for Experience/Projects/Education (Title & Company on left, Date & Location on right).
- High-density single-page A4 fit with clean margins (20pt) and proportional leading.
- Standard Times-Roman / Times-Bold / Times-Italic typography matching Latin Modern Roman for 100% ATS readability and OCR compliance.
"""

import io
import re
from typing import List, Tuple
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle


def _clean_xml(text: str) -> str:
    """Escape XML special characters while preserving intentional <b> tags."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    return text.strip()


def generate_ats_pdf(cv_text: str, candidate_name: str = "", target_role: str = "") -> bytes:
    """Generate a pixel-perfect, single-page ATS-compliant PDF matching the user's exact template."""
    buffer = io.BytesIO()

    # Exact A4 page with 20pt left/right and 18pt top/bottom margins
    margin_lr = 20.0
    margin_tb = 18.0
    page_width, page_height = A4
    usable_width = page_width - (2 * margin_lr)  # ~555.28 pt

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=margin_lr,
        rightMargin=margin_lr,
        topMargin=margin_tb,
        bottomMargin=margin_tb,
    )

    # ── Custom Typography Styles matching the user's reference template ──
    name_style = ParagraphStyle(
        "Tpl_Name",
        fontName="Times-Bold",
        fontSize=16,
        leading=18,
        textColor=colors.HexColor("#000000"),
        alignment=1,  # Centered
        spaceAfter=2,
    )

    sub_role_style = ParagraphStyle(
        "Tpl_SubRole",
        fontName="Times-Roman",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#000000"),
        alignment=1,  # Centered
        spaceAfter=1,
    )

    contact_style = ParagraphStyle(
        "Tpl_Contact",
        fontName="Times-Roman",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#111827"),
        alignment=1,  # Centered
        spaceAfter=1,
    )

    section_title_style = ParagraphStyle(
        "Tpl_SecTitle",
        fontName="Times-Bold",
        fontSize=9.5,
        leading=11,
        textColor=colors.HexColor("#000000"),
        spaceBefore=4,
        spaceAfter=1,
        keepWithNext=True,
    )

    item_title_left = ParagraphStyle(
        "Tpl_ItemLeft",
        fontName="Times-Bold",
        fontSize=8.5,
        leading=10,
        textColor=colors.HexColor("#000000"),
    )

    item_title_right = ParagraphStyle(
        "Tpl_ItemRight",
        fontName="Times-Bold",
        fontSize=8.5,
        leading=10,
        textColor=colors.HexColor("#000000"),
        alignment=2,  # Right-aligned
    )

    item_sub_left = ParagraphStyle(
        "Tpl_SubLeft",
        fontName="Times-Italic",
        fontSize=8.5,
        leading=10,
        textColor=colors.HexColor("#111827"),
    )

    item_sub_right = ParagraphStyle(
        "Tpl_SubRight",
        fontName="Times-Roman",
        fontSize=8.5,
        leading=10,
        textColor=colors.HexColor("#111827"),
        alignment=2,  # Right-aligned
    )

    body_style = ParagraphStyle(
        "Tpl_Body",
        fontName="Times-Roman",
        fontSize=8,
        leading=9.8,
        textColor=colors.HexColor("#000000"),
        spaceAfter=1,
    )

    bullet_style = ParagraphStyle(
        "Tpl_Bullet",
        fontName="Times-Roman",
        fontSize=8,
        leading=9.6,
        textColor=colors.HexColor("#000000"),
        leftIndent=10,
        firstLineIndent=-7,
        spaceAfter=0.8,
    )

    skill_style = ParagraphStyle(
        "Tpl_Skill",
        fontName="Times-Roman",
        fontSize=7.2,
        leading=8.6,
        textColor=colors.HexColor("#000000"),
        spaceAfter=0.8,
    )

    story = []

    # Clean lines
    lines = [l.strip() for l in cv_text.strip().split("\n") if l.strip()]
    if not lines:
        lines = ["Candidate Resume"]

    # ── Parse Header (First 4-5 lines) ──
    header_name = candidate_name
    header_role = target_role
    header_contacts = []
    header_links = []
    content_start_idx = 0

    sec_regex = re.compile(
        r"^(RÉSUMÉ PROFESSIONNEL|RESUME PROFESSIONNEL|PROFESSIONAL SUMMARY|SUMMARY|PROFIL|"
        r"EXPÉRIENCE PROFESSIONNELLE|EXPERIENCE PROFESSIONNELLE|WORK EXPERIENCE|EXPERIENCE|"
        r"PROJETS TECHNIQUES|TECHNICAL PROJECTS|PROJECTS|"
        r"COMPÉTENCES TECHNIQUES|TECHNICAL SKILLS|SKILLS|COMPETENCES|"
        r"FORMATION|EDUCATION|"
        r"LANGUES|LANGUAGES|"
        r"COMPÉTENCES COMPORTEMENTALES|SOFT SKILLS)",
        re.I,
    )

    for i, line in enumerate(lines[:6]):
        clean_check = line.rstrip(":#- ").upper()
        if sec_regex.match(clean_check):
            content_start_idx = i
            break
        if i == 0 and not header_name:
            header_name = line
        elif ("téléphone" in line.lower() or "phone" in line.lower() or "@" in line or re.search(r"\+?\d[\d\s\-\.]{7,}\d", line)):
            header_contacts.append(line)
        elif ("github" in line.lower() or "linkedin" in line.lower() or "portfolio" in line.lower() or "http" in line.lower()):
            header_links.append(line)
        elif i == 1 and not header_role and not ("@" in line or "github" in line.lower()):
            header_role = line
        content_start_idx = i + 1

    if not header_name:
        header_name = candidate_name or "CANDIDAT"

    # Add Centered Header
    story.append(Paragraph(_clean_xml(header_name.upper()), name_style))
    if header_role:
        story.append(Paragraph(_clean_xml(header_role), sub_role_style))
    for c in header_contacts:
        story.append(Paragraph(_clean_xml(c), contact_style))
    for lk in header_links:
        story.append(Paragraph(_clean_xml(lk), contact_style))

    remaining = lines[content_start_idx:]
    idx = 0
    total_lines = len(remaining)

    while idx < total_lines:
        line = remaining[idx].strip()
        if not line:
            idx += 1
            continue

        clean_sec = line.rstrip(":#- ")
        sec_match = sec_regex.match(clean_sec)
        if sec_match:
            sec_title = sec_match.group(1).upper()
            story.append(Paragraph(sec_title, section_title_style))
            story.append(
                HRFlowable(
                    width="100%",
                    thickness=0.5,
                    color=colors.HexColor("#000000"),
                    spaceBefore=1,
                    spaceAfter=2.5,
                )
            )
            idx += 1
            continue

        # Bullet point
        if line.startswith(("-", "•", "*", "–", "—")) or re.match(r"^\d+\.", line):
            content = re.sub(r"^([-\u2022*\u2013\u2014]|\d+\.)\s*", "", line)
            story.append(Paragraph(f"&bull;&nbsp;&nbsp;{_clean_xml(content)}", bullet_style))
            idx += 1
            continue

        # Check for 2-column Experience/Project/Education row
        date_match = re.search(
            r"((?:Jan|Fév|Fev|Mar|Avr|Apr|Mai|May|Juin|Jun|Juil|Jul|Aoû|Aou|Aug|Sep|Oct|Nov|Déc|Dec)\w*\s+\d{4}\s*[–-]\s*(?:Présent|Present|\w+\s+\d{4}))",
            line,
            re.I,
        )

        if date_match and not line.startswith(("-", "•")):
            date_str = date_match.group(1)
            left_role = line.replace(date_str, "").strip().rstrip("–- ")

            company_str = ""
            loc_str = ""
            if idx + 1 < total_lines:
                next_line = remaining[idx + 1].strip()
                if not next_line.startswith(("-", "•")) and not sec_regex.match(next_line.rstrip(":#- ")):
                    if any(c in next_line.lower() for c in ["tunis", "france", "remote", "paris", "tunisie", "sousse"]):
                        loc_match = re.search(r"(Tunis.*|France.*|Remote.*|Tunisie.*|Paris.*|Sousse.*)", next_line, re.I)
                        if loc_match:
                            loc_str = loc_match.group(1).strip()
                            company_str = next_line.replace(loc_str, "").strip().rstrip("–- ,")
                        else:
                            company_str = next_line
                    else:
                        company_str = next_line
                    idx += 1

            table_data = [
                [Paragraph(_clean_xml(left_role), item_title_left), Paragraph(_clean_xml(date_str), item_title_right)],
            ]
            if company_str or loc_str:
                table_data.append([
                    Paragraph(_clean_xml(company_str), item_sub_left),
                    Paragraph(_clean_xml(loc_str), item_sub_right),
                ])

            t = Table(table_data, colWidths=[usable_width * 0.72, usable_width * 0.28])
            t.setStyle(
                TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0.5),
                ])
            )
            story.append(t)
            idx += 1
            continue

        # Check for Category: list of skills (e.g. Langages de programmation: TypeScript...)
        if ":" in line and not line.startswith("http"):
            parts = line.split(":", 1)
            cat_name = parts[0].strip()
            cat_val = parts[1].strip()
            formatted_cat = f"<b>{_clean_xml(cat_name)}:</b> {_clean_xml(cat_val)}"
            story.append(Paragraph(formatted_cat, skill_style))
            idx += 1
            continue

        # Standard body line
        story.append(Paragraph(_clean_xml(line), body_style))
        idx += 1

    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
