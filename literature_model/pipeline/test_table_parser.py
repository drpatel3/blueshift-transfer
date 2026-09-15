from table_parser import get_cu_grade_average, get_section_13_pages, parse_table
import pdfplumber
import logging

logging.getLogger("pypdf").setLevel(logging.ERROR)

pdf_path = "test_pdfs/main_pdfs/10_pdf.pdf"

sec_start, sec_end = get_section_13_pages(pdf_path)
print(f"Section 13 pages: {sec_start} to {sec_end}\n")

if sec_start is not None:
    pdf = pdfplumber.open(pdf_path)
    for i in range(sec_start, sec_end):
        page = pdf.pages[i]
        raw_tables = page.extract_tables()
        if not raw_tables:
            continue
        print(f"{'='*80}")
        print(f"  Page {i} — {len(raw_tables)} table(s)")
        print(f"{'='*80}")
        for t_idx, raw_table in enumerate(raw_tables):
            df = parse_table(raw_table)
            if df is not None:
                print(f"\n--- Table {t_idx + 1} ---")
                print(df.to_string())
            else:
                print(f"\n--- Table {t_idx + 1} --- (no data)")
        print()
    pdf.close()

avg = get_cu_grade_average(pdf_path)
print(f"\nCu % average: {avg}%")
