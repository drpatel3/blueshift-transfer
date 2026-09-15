"""
EDGAR Exhibit 96 PDF Downloader

Downloads S-K 1300 Technical Report Summary (TRS) PDFs from SEC EDGAR.
Uses two discovery methods:
  A) EFTS full-text search for "Technical Report Summary" across annual filings
  B) Filing index crawl for known copper mining companies

Usage:
    python edgar_downloader.py [options]
    python edgar_downloader.py --dry-run
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_PATH = Path(__file__).parent / "manifest.json"

EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/{filename}"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/{accession}-index.htm"

FORMS = "10-K,20-F,40-F,10-K/A,S-1,F-1"
FORM_LIST = ["10-K", "20-F", "40-F", "10-K/A", "S-1", "F-1"]
START_DATE = "2021-01-01"

COPPER_COMPANIES = [
    {"name": "Freeport-McMoRan", "cik": "831259"},
    {"name": "Southern Copper", "cik": "1001838"},
    {"name": "BHP Group", "cik": "811809"},
    {"name": "Rio Tinto", "cik": "887028"},
    {"name": "Newmont", "cik": "1164727"},
    {"name": "Teck Resources", "cik": "886986"},
    {"name": "Hudbay Minerals", "cik": "1322422"},
    {"name": "Taseko Mines", "cik": "878518"},
    {"name": "Capstone Copper", "cik": "2075918"},
    {"name": "Ero Copper", "cik": "1853860"},
    {"name": "Ivanhoe Mines", "cik": "1158041"},
    {"name": "Trilogy Metals", "cik": "1543418"},
    {"name": "Western Copper & Gold", "cik": "1364125"},
    {"name": "Faraday Copper", "cik": "1691710"},
    {"name": "Arizona Sonoran Copper", "cik": "1812654"},
    {"name": "Nevada Copper", "cik": "1431847"},
    {"name": "Sandfire Resources", "cik": "1464928"},
    {"name": "First Quantum Minerals", "cik": "1071265"},
    {"name": "Ivanhoe Electric", "cik": "1879016"},
    {"name": "Solaris Resources", "cik": "2019103"},
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent})
    return s


def _get(session: requests.Session, url: str, delay: float = 2.0, **kwargs):
    """GET with rate limiting. Returns response or None on failure."""
    time.sleep(delay)
    try:
        r = session.get(url, timeout=30, **kwargs)
        if r.status_code == 200:
            return r
        print(f"  [warn] HTTP {r.status_code}: {url}")
    except requests.RequestException as e:
        print(f"  [warn] Request failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Discovery: EFTS full-text search
# ---------------------------------------------------------------------------

def search_efts(session: requests.Session) -> dict:
    """Search EDGAR full-text for TRS documents. Returns {key: record}."""
    print("Searching EFTS for Technical Report Summary filings...")
    found = {}
    page_size = 100

    for offset in range(0, 500, page_size):
        params = {
            "q": '"Technical Report Summary"',
            "forms": FORMS,
            "dateRange": "custom",
            "startdt": START_DATE,
            "_source": "adsh,ciks,display_names,form,file_type,file_date",
            "from": offset,
            "size": page_size,
        }
        r = _get(session, EFTS_SEARCH_URL, params=params)
        if not r:
            print(f"  [warn] EFTS page {offset} failed, stopping search")
            break

        data = r.json()
        hits = data.get("hits", {}).get("hits", [])
        total = data.get("hits", {}).get("total", {}).get("value", 0)

        for hit in hits:
            fid = hit["_id"]
            if not fid.lower().endswith(".pdf"):
                continue
            src = hit["_source"]
            # Only keep actual Exhibit 96 files
            file_type = src.get("file_type", "")
            if file_type and not file_type.startswith("EX-96"):
                continue
            accession, filename = fid.split(":", 1)
            cik = src["ciks"][0].lstrip("0") if src.get("ciks") else ""
            key = f"{accession}:{filename}"
            found[key] = {
                "company": src.get("display_names", [""])[0].split("(")[0].strip(),
                "form": src.get("form", ""),
                "exhibit": src.get("file_type", ""),
                "file_date": src.get("file_date", ""),
                "cik": cik,
                "accession": accession,
                "filename": filename,
            }

        print(f"  EFTS page {offset}: {len(hits)} hits (total: {total})")
        if offset + page_size >= total:
            break

    print(f"  EFTS found {len(found)} PDFs\n")
    return found


# ---------------------------------------------------------------------------
# Discovery: Company filing index crawl
# ---------------------------------------------------------------------------

def crawl_company_filings(session: requests.Session) -> dict:
    """Crawl filing indexes for known copper companies. Returns {key: record}."""
    print(f"Crawling filing indexes for {len(COPPER_COMPANIES)} companies...")
    found = {}

    for co in COPPER_COMPANIES:
        cik = co["cik"]
        cik_padded = cik.zfill(10)
        url = SUBMISSIONS_URL.format(cik=cik_padded)
        r = _get(session, url)
        if not r:
            print(f"  {co['name']}: failed to fetch submissions")
            continue

        data = r.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])

        # Filter to relevant forms from 2021+
        filing_list = []
        for i, form in enumerate(forms):
            if form in FORM_LIST and dates[i] >= START_DATE:
                filing_list.append((accessions[i], dates[i], form))

        if not filing_list:
            print(f"  {co['name']}: no filings since {START_DATE}")
            continue

        pdf_count = 0
        for accession, date, form in filing_list:
            accession_path = accession.replace("-", "")
            idx_url = INDEX_URL.format(
                cik=cik, accession_path=accession_path, accession=accession
            )
            r = _get(session, idx_url)
            if not r:
                continue

            links = re.findall(r'href="(/Archives/[^"]+\.pdf)"', r.text, re.IGNORECASE)
            for link in links:
                filename = link.split("/")[-1]
                key = f"{accession}:{filename}"
                if key not in found:
                    found[key] = {
                        "company": co["name"],
                        "form": form,
                        "exhibit": "",
                        "file_date": date,
                        "cik": cik,
                        "accession": accession,
                        "filename": filename,
                    }
                    pdf_count += 1

        print(f"  {co['name']}: {pdf_count} new PDFs from {len(filing_list)} filings")

    print(f"  Crawl found {len(found)} PDFs\n")
    return found


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"last_search": None, "filings": {}}


def save_manifest(manifest: dict):
    manifest["last_search"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_pdfs(session: requests.Session, records: dict, output_dir: Path,
                  manifest: dict, dry_run: bool) -> list:
    """Download PDFs. Returns list of downloaded file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    skipped = 0
    failed = []

    items = sorted(records.items(), key=lambda x: (x[1]["company"], x[1]["file_date"]))

    for i, (key, rec) in enumerate(items, 1):
        filename = rec["filename"]
        filepath = output_dir / filename
        accession_path = rec["accession"].replace("-", "")
        url = ARCHIVES_URL.format(
            cik=rec["cik"], accession_path=accession_path, filename=filename
        )
        rec["download_url"] = url

        # Check manifest — already downloaded and file exists
        existing = manifest["filings"].get(key)
        if existing and existing.get("status") == "downloaded":
            local = output_dir / existing.get("local_path", "")
            if local.exists():
                print(f"  [{i}/{len(items)}] [skip] {rec['company']} | {filename}")
                skipped += 1
                continue

        if dry_run:
            size_str = ""
            print(f"  [{i}/{len(items)}] [dry-run] {rec['company']} | {rec['form']} | {rec['file_date']} | {filename}")
            manifest["filings"][key] = {**rec, "status": "pending", "local_path": filename}
            continue

        print(f"  [{i}/{len(items)}] {rec['company']} | {filename} ...", end=" ", flush=True)
        r = _get(session, url, delay=2.0, stream=True)
        if r:
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            size_kb = filepath.stat().st_size // 1024
            print(f"OK ({size_kb} KB)")
            manifest["filings"][key] = {**rec, "status": "downloaded", "local_path": filename}
            downloaded.append(str(filepath))
        else:
            print("FAILED")
            manifest["filings"][key] = {**rec, "status": "failed", "local_path": filename}
            failed.append((key, rec))

    # Retry failed downloads once
    if failed and not dry_run:
        print(f"\nRetrying {len(failed)} failed download(s)...")
        for key, rec in failed:
            filename = rec["filename"]
            filepath = output_dir / filename
            url = rec["download_url"]
            print(f"  [retry] {rec['company']} | {filename} ...", end=" ", flush=True)
            r = _get(session, url, delay=3.0, stream=True)
            if r:
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                size_kb = filepath.stat().st_size // 1024
                print(f"OK ({size_kb} KB)")
                manifest["filings"][key] = {**rec, "status": "downloaded", "local_path": filename}
                downloaded.append(str(filepath))
            else:
                print("FAILED again")

    print(f"\nDownloaded: {len(downloaded)}, Skipped: {skipped}, Failed: {len(failed) - len([d for d in downloaded if d in [str(output_dir / r['filename']) for _, r in failed]])}")
    return downloaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    print("EDGAR Exhibit 96 PDF Downloader")
    print(f"  Output: {args.output_dir}")
    if args.dry_run:
        print("  Mode: DRY RUN\n")
    else:
        print()

    session = _session(args.user_agent)
    manifest = load_manifest()
    output_dir = Path(args.output_dir)

    # Discovery
    efts_records = search_efts(session)
    crawl_records = crawl_company_filings(session)

    # Merge (crawl takes precedence since it has cleaner company names)
    all_records = {**efts_records, **crawl_records}
    print(f"Total unique PDFs: {len(all_records)} (EFTS: {len(efts_records)}, Crawl: {len(crawl_records)}, Overlap: {len(efts_records) + len(crawl_records) - len(all_records)})\n")

    # Download
    downloaded = download_pdfs(session, all_records, output_dir, manifest, args.dry_run)

    # Save manifest
    save_manifest(manifest)
    print(f"\nManifest saved to {MANIFEST_PATH}")
    print(f"Done. {len(downloaded)} files downloaded to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Exhibit 96 TRS PDFs from SEC EDGAR"
    )
    parser.add_argument(
        "--output-dir", default="../data/edgar_downloads",
        help="Download output directory (default: ../data/edgar_downloads)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List matching PDFs without downloading"
    )
    parser.add_argument(
        "--user-agent",
        default="LiteraturePlatform research@example.com",
        help="User-Agent string (SEC requires name + email)"
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Re-attempt previously failed downloads from manifest"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
