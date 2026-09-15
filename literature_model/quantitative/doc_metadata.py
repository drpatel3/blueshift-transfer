"""Build doc_metadata.json — per-document vintage and study type.

The KNN trace shows *which* analog projects drove a design, but not how old or
how mature the disclosures behind them are. A stage voted in by five 2009
preliminary economic assessments is weaker evidence than the same stage voted
in by three recent feasibility studies, and until now the trace gave no way to
tell those apart.

Two fields per document, both re-derived from the pipeline results already in
S3 (`mineral-pipeline-pipeline/results/`) — nothing new is extracted from the
PDFs:

  year        the report's effective/report date if one is stated on the front
              matter, else the SEDAR filing year parsed out of the object key
              (`..._13_nov_2025_10_36.json`). `year_source` says which.
  study_type  FS / PFS / PEA / MRE / TR, classified from the study-type phrases
              on the front matter. Title pages carry TITLE_WEIGHT so a
              feasibility study that merely cites an earlier PEA still reads
              as FS. `study_type_votes` keeps the raw counts for auditing.

Usage:
    python -m quantitative.doc_metadata                    # full corpus -> json
    python -m quantitative.doc_metadata --limit 40 --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RESULTS_BUCKET = "mineral-pipeline-pipeline"
RESULTS_PREFIX = "results/"
OUTPUT_PATH = Path(__file__).parent / "doc_metadata.json"

# Pages of front matter to read. The QP certificates — the most reliable place
# a report names itself — sit behind the summary and often past page 12, and
# many covers are scanned images with no extractable text at all. Reading
# deeper costs nothing in precision: only certificate titles and title-block
# lines can assign a label, never a loose mention.
FRONT_PAGES = 24
TITLE_PAGES = 3
TITLE_WEIGHT = 5

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

# Filing date embedded in the SEDAR filename: `..._13_nov_2025_10_36.pdf`
_FILING_DATE = re.compile(r"(\d{1,2})_([a-z]{3})_(\d{4})", re.I)

# "effective date: 31 December 2021" / "effective date of March 4, 2019"
_EFFECTIVE_YEAR = re.compile(
    r"effective\s+date[^.\n]{0,60}?(19|20)\d{2}", re.I)
_YEAR_IN = re.compile(r"(19|20)\d{2}")

# Ordered most-specific first: "pre-feasibility" must win before the bare
# "feasibility study" pattern gets a chance at the same span.
_STUDY_PATTERNS = [
    ("PFS", re.compile(r"pre[-\s]?feasibility(?:\s+study)?", re.I)),
    ("PEA", re.compile(r"preliminary\s+economic\s+assessment|scoping\s+study", re.I)),
    ("FS",  re.compile(r"(?<!pre-)(?<!pre\s)(?<!prefeasibility)"
                       r"(?:definitive\s+|bankable\s+)?feasibility\s+study", re.I)),
    ("MRE", re.compile(r"mineral\s+resource\s+(?:estimate|update)", re.I)),
]
_TIE_ORDER = ["FS", "PFS", "PEA", "MRE"]
# The three economic grades, as one alternation, for the declaration patterns.
_STUDY_ALT = (r"pre[-\s]?feasibility\s+study|feasibility\s+study"
              r"|preliminary\s+economic\s+assessment")
# Every economic study also contains a resource estimate, so MRE can never
# outvote FS/PFS/PEA — it is only the answer when none of them appear.
_ECONOMIC = ["FS", "PFS", "PEA"]

# Contents-listing lines mention every study phrase in the document and would
# otherwise swamp the report's own billing — this is what made MRE fire on any
# report merely containing a section 14. Four shapes, all from front matter:
# dot leaders, "Table 14-1 ..." captions, numbered section headings
# ("14.0 MINERAL RESOURCE ESTIMATES"), and short lines ending in a page number
# ("Mineral Resource Estimates 147"). The page-number rule is capped at three
# digits and short lines so prose ending in a year ("... effective date ...
# 2024") survives for the date parser.
_TOC_LINE = re.compile(
    r"\.{4,}"
    r"|^\s*(?:table|figure)\s+\d+[-.]\d+"
    r"|^\s*\d+(?:\.\d+)*\s+\S"
    r"|^.{0,80}?\s\d{1,3}\s*$", re.I)

# 'This certificate applies to the technical report titled "Canariaco Copper
# Project NI 43-101 Technical Report & Preliminary Economic Assessment"'.
# Quotes in the extracted text are whatever the PDF encoder emitted.
_CERT_TITLE = re.compile(
    r"report\s+(?:titled|entitled)[\s:]*[\"“”'`“”]?"
    r"([^\"“”\n\r]{10,200})", re.I)

# Title-block lines that name the instrument itself.
_TITLE_LINE = re.compile(
    r"ni\s*43-?101|technical\s+report|feasibility|preliminary\s+economic", re.I)

# Trailing page/section marker on a running header ("... – May 2021  Page iv").
_HEADER_TAIL = re.compile(r"\b(?:page\s*)?[ivxlcdm]+-?\d*\s*$", re.I)
MIN_HEADER_REPEATS = 3

# Sentences where a report declares its own grade, as opposed to citing someone
# else's study. Each must tie the study phrase to a self-reference ("this
# report presents ...", "results of the feasibility study"), which is what
# keeps a literature review of prior work from labelling the document.
_DECLARATIONS = [
    re.compile(r"(?:this|the)\s+(?:ni\s*43-?101\s+)?(?:technical\s+)?"
               r"(?:report|document|study)\s+(?:presents|summari[sz]es|provides"
               r"|documents|describes|details|reports)"
               r"[^.]{0,140}?(" + _STUDY_ALT + r")", re.I),
    re.compile(r"results?\s+of\s+(?:a|the|an)\s+[^.]{0,60}?(" + _STUDY_ALT + r")", re.I),
    re.compile(r"^\s*this\s+(" + _STUDY_ALT + r")\b", re.I | re.M),
    re.compile(r"(?:prepared|completed|undertaken)\s+(?:a|the|this)\s+("
               + _STUDY_ALT + r")\s+(?:for|on|of)", re.I),
]

# Grades trusted enough to show a stakeholder. FS/PFS are the engineered
# studies and both survive title-anchored identification. PEA is excluded by
# choice, not by evidence quality — it classifies cleanly, so a PEA-backed
# design currently renders the same as one with no stated grade. MRE is
# excluded because no document in the corpus names itself one: every MRE tag
# came from contents-listing headings before the title anchor was added.
# Anything not in this set shows its year alone.
DISPLAYED_STUDY_TYPES = frozenset({"FS", "PFS"})

STUDY_LABELS = {
    "FS": "Feasibility Study",
    "PFS": "Pre-Feasibility Study",
    "PEA": "Preliminary Economic Assessment",
    "MRE": "Mineral Resource Estimate",
    "TR": "Technical Report",
    "UNKNOWN": "Study type not disclosed",
}


def filing_year(name: str) -> int | None:
    """Year from the SEDAR filing date in a result key or filename."""
    m = _FILING_DATE.search(name)
    if not m or m.group(2).lower() not in _MONTHS:
        return None
    return int(m.group(3))


def _strip_toc(text: str) -> str:
    """Drop contents-listing lines, keeping the prose of the front matter."""
    return "\n".join(ln for ln in text.splitlines() if not _TOC_LINE.search(ln))


def _front_matter(doc: dict) -> tuple[str, str]:
    """Return (title_text, front_text) — the first pages of the document, with
    table-of-contents lines removed."""
    pages = [p.get("text") or "" for p in (doc.get("page_text") or [])]
    return (_strip_toc("\n".join(pages[:TITLE_PAGES])),
            _strip_toc("\n".join(pages[:FRONT_PAGES])))


def effective_year(front_text: str) -> int | None:
    """Year of the stated effective/report date, if the front matter has one."""
    m = _EFFECTIVE_YEAR.search(front_text)
    if not m:
        return None
    year = _YEAR_IN.search(m.group(0)[m.group(0).lower().index("date"):])
    return int(year.group(0)) if year else None


def report_titles(title_text: str, front_text: str, filename: str) -> list[str]:
    """The strings where a report states what it *is*, as opposed to anywhere
    the phrase happens to appear.

    Counting mentions across the front matter cannot support a displayed label:
    every feasibility study cites earlier PEAs, and every report with a section
    14 mentions a mineral resource estimate. Three positive-identification
    sources only:

      the QP certificate  'This certificate applies to the technical report
                          titled "X"' — the most reliable, it is the report
                          naming itself under professional signature
      the title block     title-page lines that name the instrument
                          ('... NI 43-101 Technical Report and Preliminary
                          Economic Assessment')
      the filename        the SEDAR document title
    """
    titles = [filename] if filename else []
    titles += [m.group(1) for m in _CERT_TITLE.finditer(front_text)]
    titles += [ln.strip() for ln in title_text.splitlines()
               if _TITLE_LINE.search(ln)]
    return [t for t in titles if t.strip()]


def running_header_titles(pages: list[str]) -> list[str]:
    """Report titles taken from running headers. A line repeated across many
    pages is the report stating its own name — which recovers the reports whose
    cover page is a scanned image with no extractable text."""
    counts: Counter = Counter()
    for page in pages:
        for line in {re.sub(r"\s+", " ", ln).strip() for ln in page.splitlines()}:
            line = _HEADER_TAIL.sub("", line).strip()
            if 15 <= len(line) <= 160 and _TITLE_LINE.search(line):
                counts[line] += 1
    return [line for line, n in counts.items() if n >= MIN_HEADER_REPEATS]


def declared_studies(front_text: str) -> list[str]:
    """Study grades the report claims in its own summary prose."""
    return [m.group(1) for rx in _DECLARATIONS for m in rx.finditer(front_text)]


def classify_study_type(title_text: str, front_text: str, filename: str = "",
                        pages: list[str] | None = None) -> tuple[str, dict]:
    """Return (label, votes) — the study grade the report claims for itself.

    A label is only assigned when a study phrase appears in one of the report's
    own titles (see `report_titles`); otherwise the answer is TR, "no study
    grade stated", which is an honest reading of a technical report that never
    bills itself as one. `votes` records the per-label title hits."""
    evidence = report_titles(title_text, front_text, filename)
    evidence += running_header_titles(pages or [])
    evidence += declared_studies(front_text)
    votes: Counter = Counter()
    for text in evidence:
        for label, pattern in _STUDY_PATTERNS:
            if pattern.search(text):
                votes[label] += 1
    if not votes:
        return "TR", {}
    # A title reading "Technical Report and Preliminary Economic Assessment on
    # the ... Mineral Resource Estimate" names both; the economic study is the
    # report's grade, the resource estimate is its content.
    pool = [lab for lab in _ECONOMIC if lab in votes] or list(votes)
    top = max(votes[lab] for lab in pool)
    winners = [lab for lab in _TIE_ORDER if lab in pool and votes[lab] == top]
    return winners[0], dict(votes)


def metadata_for(doc: dict, key: str) -> dict:
    """Vintage + study type for one pipeline result."""
    title_text, front_text = _front_matter(doc)
    name = doc.get("filename") or key
    eff, filed = effective_year(front_text), filing_year(name)
    year = eff or filed
    pages = [p.get("text") or "" for p in (doc.get("page_text") or [])][:FRONT_PAGES]
    study_type, votes = classify_study_type(title_text, front_text, name, pages)
    if not front_text.strip():
        # No page text was extracted for this PDF — say so rather than
        # defaulting it into the generic "Technical Report" bucket.
        study_type, votes = "UNKNOWN", {}
    return {
        "year": year,
        "year_source": "effective_date" if eff else ("filing" if filed else None),
        "filing_year": filed,
        "study_type": study_type,
        "study_type_votes": votes,
        "pages_scanned": min(FRONT_PAGES, len(doc.get("page_text") or [])),
    }


def build(limit: int | None = None, workers: int = 32) -> dict:
    """Scan every pipeline result in S3 and return {doc_id: metadata}."""
    import boto3

    s3 = boto3.client("s3")
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=RESULTS_BUCKET, Prefix=RESULTS_PREFIX):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json"):
                keys.append(obj["Key"])
    keys.sort()
    if limit:
        keys = keys[:limit]
    print(f"Scanning {len(keys)} pipeline results ...", file=sys.stderr)

    def _one(key):
        try:
            body = s3.get_object(Bucket=RESULTS_BUCKET, Key=key)["Body"].read()
            doc = json.loads(body)
        except Exception as exc:  # keep the sweep going; report at the end
            print(f"  skip {key}: {exc}", file=sys.stderr)
            return None
        doc_id = doc.get("document_id")
        return (doc_id, metadata_for(doc, key)) if doc_id else None

    out = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, item in enumerate(pool.map(_one, keys), start=1):
            if item:
                out[item[0]] = item[1]
            if i % 200 == 0:
                print(f"  {i}/{len(keys)} ...", file=sys.stderr)
    return out


def summarize(meta: dict) -> None:
    types = Counter(m["study_type"] for m in meta.values())
    years = [m["year"] for m in meta.values() if m["year"]]
    sources = Counter(m["year_source"] for m in meta.values())
    print(f"  {len(meta)} documents", file=sys.stderr)
    print("  study types: " + ", ".join(
        f"{k}={v}" for k, v in types.most_common()), file=sys.stderr)
    if years:
        print(f"  years: {min(years)}–{max(years)}, median "
              f"{sorted(years)[len(years) // 2]} "
              f"({sources['effective_date']} from effective date, "
              f"{sources['filing']} from filing)", file=sys.stderr)


def load(path: str | Path = OUTPUT_PATH) -> dict:
    """Read the frozen map; empty dict if it hasn't been built yet."""
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUTPUT_PATH))
    ap.add_argument("--limit", type=int, default=None,
                    help="Only scan the first N results (smoke test).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the summary without writing the map.")
    args = ap.parse_args()

    meta = build(limit=args.limit)
    summarize(meta)
    if args.dry_run:
        for doc_id, m in list(meta.items())[:10]:
            print(f"  {doc_id[:12]}  {m['year']} ({m['year_source']})  "
                  f"{m['study_type']:<4} {m['study_type_votes']}")
        return
    Path(args.out).write_text(json.dumps(meta, indent=2))
    print(f"Wrote {len(meta)} entries -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
