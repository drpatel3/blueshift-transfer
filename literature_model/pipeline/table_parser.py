import re
import logging
import pdfplumber
import pandas as pd
from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)

SEC_13_START = re.compile(
    r'(?:Table|Figure)\s+13[\-\.]\d+', re.IGNORECASE
)
SEC_14_START = re.compile(
    r'(?:Table|Figure)\s+14[\-\.]\d+', re.IGNORECASE
)


def get_section_pages(pdf_path, section_num, start_page=5, _reader=None):
    """Return the page range [first_section, next_section) for a given NI 43-101 section.

    Uses Table/Figure numbering patterns (e.g., 'Table 17.1', 'Figure 13-2') to detect
    section boundaries. Returns (start_index, end_index) as 0-based page indices,
    or (None, None) if the section is not found.

    Pass _reader to reuse an already-opened PdfReader (avoids re-parsing the PDF).
    """
    sec_start_re = re.compile(
        rf'(?:Table|Figure)\s+{section_num}[\-\.]\d+', re.IGNORECASE
    )
    next_sec = int(section_num) + 1
    sec_end_re = re.compile(
        rf'(?:Table|Figure)\s+{next_sec}[\-\.]\d+', re.IGNORECASE
    )
    reader = _reader or PdfReader(pdf_path)
    first_page = None
    for i in range(start_page, len(reader.pages)):
        text = (reader.pages[i].extract_text() or "")[:500]
        if first_page is None:
            if sec_start_re.search(text):
                first_page = i
        else:
            if sec_end_re.search(text):
                return first_page, i
    if first_page is not None:
        return first_page, len(reader.pages)
    return None, None


def find_multiple_sections(pdf_path, section_nums=("13", "17"), start_page=50):
    """Find page ranges for multiple sections in a single PDF read.

    Returns dict: {section_num: (start, end)} for sections found.
    """
    reader = PdfReader(pdf_path)
    results = {}
    for sec in section_nums:
        start, end = get_section_pages(pdf_path, sec, start_page=start_page, _reader=reader)
        if start is not None:
            results[sec] = (start, end)
    return results


def expand_rows(raw_table):
    """Expand cells with newlines into separate rows."""
    expanded = []
    for row in raw_table:
        split_cells = [str(c or "").split("\n") for c in row]
        num_sub = max(len(parts) for parts in split_cells)
        for j in range(num_sub):
            new_row = []
            for parts in split_cells:
                new_row.append(parts[j].strip() if j < len(parts) else "")
            expanded.append(new_row)

    # Merge subscript orphan rows back into the row above.
    # A subscript row: every filled cell is a short fragment (<=3 chars),
    # e.g. "4" from ZnSO4 and CuSO4 landing on their own row.
    merged = []
    for row in expanded:
        filled = [(i, v) for i, v in enumerate(row) if v.strip()]
        if (merged
                and filled
                and all(len(v) <= 3 for _, v in filled)):
            for col_idx, fragment in filled:
                merged[-1][col_idx] += fragment
        else:
            merged.append(row)
    return merged


def drop_empty_columns(rows):
    """Remove columns that are empty across ALL rows. Preserves positional alignment."""
    if not rows:
        return rows
    num_cols = max(len(row) for row in rows)
    # Pad rows to same length
    padded = [row + [''] * (num_cols - len(row)) for row in rows]
    # Find columns that have at least one non-empty value
    keep = [c for c in range(num_cols)
            if any(padded[r][c].strip() for r in range(len(padded)))]
    return [[row[c] for c in keep] for row in padded]


def parse_table(raw_table):
    """Parse pdfplumber raw table into a DataFrame."""
    rows = expand_rows(raw_table)
    rows = drop_empty_columns(rows)

    if not rows:
        return None

    num_cols = len(rows[0])

    # Find first data row (has numbers)
    def has_numbers(row):
        return any(c.replace(',', '').replace('.', '').replace('-', '').isdigit()
                   for c in row if c.strip())

    # Collect all header rows (non-numeric before first data row)
    header_rows = []
    data_start = 0
    for i, row in enumerate(rows):
        if has_numbers(row):
            data_start = i
            break
        header_rows.append(row)

    data = [row for row in rows[data_start:] if any(c.strip() for c in row)]
    if not data:
        return None

    # Merge all header rows: per column, join non-empty values top-down
    merged_header = []
    for col in range(num_cols):
        parts = [header_rows[r][col].strip()
                 for r in range(len(header_rows))
                 if col < len(header_rows[r]) and header_rows[r][col].strip()]
        merged_header.append(" ".join(parts) if parts else "")

    # Find columns that are empty across ALL data rows (header-only columns).
    # Merge their header text into the nearest data column to the left.
    data_empty_cols = set()
    for col in range(num_cols):
        if all(not data[r][col].strip() for r in range(len(data))):
            data_empty_cols.add(col)

    # For each header-only col, push its header text to the nearest kept col on the left
    for col in sorted(data_empty_cols):
        if not merged_header[col]:
            continue  # no header text to redistribute
        # Find nearest data column to the left
        target = None
        for k in range(col - 1, -1, -1):
            if k not in data_empty_cols:
                target = k
                break
        if target is None:
            # No data column to the left; try right
            for k in range(col + 1, num_cols):
                if k not in data_empty_cols:
                    target = k
                    break
        if target is not None:
            merged_header[target] = merged_header[col] + (" " + merged_header[target] if merged_header[target] else "")

    # Keep only columns that have data
    keep_cols = [c for c in range(num_cols) if c not in data_empty_cols]
    header = [merged_header[c] or f"col_{i}" for i, c in enumerate(keep_cols)]
    data = [[row[c] for c in keep_cols] for row in data]

    # Dedupe duplicate column names
    seen = {}
    for j, h in enumerate(header):
        if h in seen:
            seen[h] += 1
            header[j] = f"{h}_{seen[h]}"
        else:
            seen[h] = 1

    return pd.DataFrame(data, columns=header)



def get_section_13_pages(pdf_path, start_page=60):
    """Return the page range [first_13, first_14) for section 13 in the PDF."""
    reader = PdfReader(pdf_path)
    first_13 = None
    for i in range(start_page, len(reader.pages)):
        text = reader.pages[i].extract_text() or ""
        if first_13 is None:
            if SEC_13_START.search(text):
                first_13 = i
        else:
            if SEC_14_START.search(text):
                return first_13, i
    # Section 13 found but no section 14 boundary — go to end
    if first_13 is not None:
        return first_13, len(reader.pages)
    return None, None


def get_cu_grade_average(pdf_path, start_page=60):
    """Extract tables from section 13 of a PDF and return the average Cu % (0.15–3.0 range).

    Returns the mean of per-table Cu % averages, or None if no values found.
    """
    sec_start, sec_end = get_section_13_pages(pdf_path, start_page)
    if sec_start is None:
        return None

    pdf = pdfplumber.open(pdf_path)

    # Extract all tables from every page in section 13
    all_tables = []
    for i in range(sec_start, sec_end):
        page = pdf.pages[i]
        for raw_table in page.extract_tables():
            df = parse_table(raw_table)
            if df is not None:
                df["_page"] = i
                all_tables.append(df)

    # Average Cu % per table, then average those means
    table_averages = []
    for df in all_tables:
        table_cu = []
        for col in df.columns:
            col_lower = col.lower()
            if "cu" not in col_lower and "copper" not in col_lower:
                continue
            if any(skip in col_lower for skip in ["ppm", "g/t", "dist", "rec", "ratio"]):
                continue
            for val in df[col]:
                cleaned = str(val).replace(",", "").strip()
                try:
                    num = float(cleaned)
                    if 0.15 <= num <= 3.0:
                        table_cu.append(num)
                except ValueError:
                    continue
        if table_cu:
            table_averages.append(sum(table_cu) / len(table_cu))

    pdf.close()

    if not table_averages:
        return None
    return sum(table_averages) / len(table_averages)