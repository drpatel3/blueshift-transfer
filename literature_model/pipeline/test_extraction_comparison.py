"""
Compare flowsheet extraction methods to see which finds the most PFS images.

Method 1: find_process_flowsheet() — caption-targeted, handles rotated pages
Method 2: extract_all_figures() — single-pass, tags flowsheet vs other
"""

import time
import shutil
from pathlib import Path
from pdf_extraction import find_process_flowsheet, extract_all_figures

PDF_FOLDER = Path("test_pdfs/main_pdfs")
TEST_PDFS = sorted(PDF_FOLDER.glob("test_*.pdf"))


def run_method1(pdf_path, output_folder):
    """find_process_flowsheet with default settings."""
    return find_process_flowsheet(pdf_location=pdf_path, output_folder=output_folder)


def run_method2(pdf_path, output_folder):
    """extract_all_figures with default settings."""
    results = extract_all_figures(pdf_location=pdf_path, output_folder=output_folder)
    flowsheets = [r for r in results if r['type'] == 'flowsheet']
    return flowsheets


if __name__ == "__main__":
    tmp_base = Path("test_extraction_tmp")
    tmp_base.mkdir(exist_ok=True)

    summary = []

    for pdf_path in TEST_PDFS:
        print(f"\n{'='*60}")
        print(f"  {pdf_path.name}")
        print(f"{'='*60}")

        # Method 1
        m1_dir = tmp_base / "m1" / pdf_path.stem
        m1_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Method 1: find_process_flowsheet ---")
        t0 = time.time()
        m1_results = run_method1(pdf_path, m1_dir)
        m1_time = time.time() - t0
        m1_pages = []
        for p in m1_results:
            # extract page number from filename like stem_flowsheet_1_page213.png
            name = Path(p).stem
            parts = name.split("page")
            if len(parts) > 1:
                m1_pages.append(parts[-1])

        # Method 2
        m2_dir = tmp_base / "m2" / pdf_path.stem
        m2_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Method 2: extract_all_figures ---")
        t0 = time.time()
        m2_results = run_method2(pdf_path, m2_dir)
        m2_time = time.time() - t0
        m2_pages = [str(r.get('page', '?')) for r in m2_results]

        summary.append({
            "pdf": pdf_path.name,
            "m1_count": len(m1_results),
            "m1_pages": m1_pages,
            "m1_time": m1_time,
            "m2_count": len(m2_results),
            "m2_pages": m2_pages,
            "m2_time": m2_time,
        })

    # Print comparison table
    print(f"\n\n{'='*80}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'PDF':<25} {'M1 (find_pfs)':<18} {'M2 (all_figs)':<18} {'Winner':<10}")
    print(f"{'':<25} {'count  time':<18} {'count  time':<18}")
    print(f"{'-'*80}")

    m1_total = 0
    m2_total = 0
    for row in summary:
        m1_total += row['m1_count']
        m2_total += row['m2_count']
        if row['m1_count'] > row['m2_count']:
            winner = "M1"
        elif row['m2_count'] > row['m1_count']:
            winner = "M2"
        else:
            winner = "TIE"

        print(f"{row['pdf']:<25} {row['m1_count']:>3}  {row['m1_time']:>6.1f}s      {row['m2_count']:>3}  {row['m2_time']:>6.1f}s      {winner}")
        if row['m1_pages']:
            print(f"  M1 pages: {', '.join(row['m1_pages'])}")
        if row['m2_pages']:
            print(f"  M2 pages: {', '.join(row['m2_pages'])}")

    print(f"{'-'*80}")
    print(f"{'TOTAL':<25} {m1_total:>3}{'':>15} {m2_total:>3}")
    print(f"\nM1 = find_process_flowsheet  (start_page=120, handles rotation)")
    print(f"M2 = extract_all_figures     (flowsheet_start_page=150, single-pass)")

    # Cleanup
    cleanup = input("\nDelete temp output? (y/n): ").strip().lower()
    if cleanup == 'y':
        shutil.rmtree(tmp_base)
        print("Cleaned up.")