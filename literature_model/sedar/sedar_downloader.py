"""
SEDAR+ Document Downloader

Playwright-based tool to search and download NI 43-101 technical reports
(and other document types) from SEDAR+ filing profile pages.

Usage:
    python sedar_downloader.py --url <SEDAR_PROFILE_URL> [options]

First run should use --headed so you can complete any bot challenge manually.
Session cookies are saved so subsequent runs reuse the solved challenge.
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ---------------------------------------------------------------------------
# DOM Selectors — update these when SEDAR+ changes its markup
# ---------------------------------------------------------------------------

SEL_RESULTS_ROWS = 'table tbody tr'
SEL_DOWNLOAD_LINK = 'a[href*="resource.html"]'
SEL_NEXT_PAGE = (
    'a:has-text("Next"), button:has-text("Next"), '
    'li.next a, a[aria-label="Next"]'
)
SEL_SEARCH_BUTTON = 'button[type="submit"], button:has-text("Search"), input[type="submit"]'

# Known column indices in the results table
COL_COMPANY = 1
COL_DOC_NAME = 2
COL_DATE = 3
COL_SIZE = 5

# Session file for persisting cookies between runs (suffix set by --instance)
SESSION_DIR = Path(__file__).parent


def _session_file(instance: str = "") -> Path:
    """Return session file path for the given instance name."""
    suffix = f"_{instance}" if instance else ""
    return SESSION_DIR / f".sedar_session{suffix}.json"


# ---------------------------------------------------------------------------
# Stealth & session persistence
# ---------------------------------------------------------------------------

def _apply_stealth(page):
    """Inject JS to hide automation signals from bot detectors."""
    page.add_init_script("""
        // Hide webdriver flag
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

        // Fake plugins array (real Chrome has plugins)
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });

        // Fake languages
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
        });

        // Remove Chrome DevTools detection
        window.chrome = { runtime: {} };

        // Spoof permissions query
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) =>
            parameters.name === 'notifications'
                ? Promise.resolve({ state: Notification.permission })
                : originalQuery(parameters);
    """)


def _save_session(context, instance=""):
    """Save cookies and storage state to disk."""
    sf = _session_file(instance)
    try:
        state = context.storage_state()
        sf.write_text(json.dumps(state, indent=2))
        print(f"  Session saved to {sf.name}")
    except Exception as e:
        print(f"  [warn] Could not save session: {e}")


def _load_session(browser, args):
    """Create context with saved session if available, else fresh context."""
    sf = _session_file(args.instance)
    context_args = {
        "accept_downloads": True,
        "viewport": {"width": 1280, "height": 900},
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
    }

    if sf.exists() and not args.fresh:
        try:
            state = json.loads(sf.read_text())
            context_args["storage_state"] = state
            print(f"  Loaded saved session from {sf.name}")
        except Exception:
            print("  [warn] Could not load saved session, starting fresh")

    return browser.new_context(**context_args)


def _human_delay(min_s=1, max_s=4):
    """Sleep a random duration to look human."""
    time.sleep(random.uniform(min_s, max_s))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_for_user(page, message, fallback_ms=15000):
    """Inject a 'START DOWNLOADING' button in the page; wait for the user to click it."""
    print(message)
    print(f"  Click the red 'START DOWNLOADING' button in the browser (up to {fallback_ms // 1000}s).")

    inject_js = """
        (() => {
            const add = () => {
                if (document.getElementById('claude-go-btn')) return;
                if (!document.body) return;
                const btn = document.createElement('button');
                btn.id = 'claude-go-btn';
                btn.textContent = 'START DOWNLOADING';
                btn.style.cssText = 'position:fixed;top:12px;right:12px;z-index:2147483647;' +
                    'padding:14px 22px;background:#d9534f;color:#fff;font-size:15px;' +
                    'font-weight:bold;border:2px solid #000;border-radius:6px;cursor:pointer;' +
                    'box-shadow:0 4px 10px rgba(0,0,0,0.35);font-family:sans-serif;';
                btn.onclick = () => {
                    window.__claudeReady = true;
                    btn.textContent = 'CONTINUING...';
                    btn.disabled = true;
                    btn.style.background = '#5cb85c';
                };
                document.body.appendChild(btn);
            };
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', add);
            } else {
                add();
            }
        })();
    """

    page.add_init_script(inject_js)
    try:
        page.evaluate(inject_js)
    except Exception:
        pass

    def _reinject(_frame):
        try:
            page.evaluate(inject_js)
        except Exception:
            pass

    page.on("framenavigated", _reinject)
    try:
        page.wait_for_function(
            "() => window.__claudeReady === true",
            timeout=fallback_ms,
        )
        print("  User clicked — continuing.")
        page.evaluate("() => { window.__claudeReady = false; }")
    except PlaywrightTimeout:
        print(f"  [warn] Button not clicked within {fallback_ms // 1000}s — continuing anyway")
    finally:
        try:
            page.remove_listener("framenavigated", _reinject)
        except Exception:
            pass


def parse_size_to_kb(text: str) -> int | None:
    """Convert '7,234 KB' or '12.3 MB' to integer KB. Returns None on failure."""
    if not text:
        return None
    text = text.strip().upper().replace(",", "")
    match = re.match(r"([\d.]+)\s*(KB|MB|GB|B)", text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    multipliers = {"B": 1/1024, "KB": 1, "MB": 1024, "GB": 1024*1024}
    return int(value * multipliers[unit])


def safe_filename(company: str, doc_name: str, date_text: str = "") -> str:
    """Build a safe filename from company + document name + date."""
    company_short = company.split("/")[0].strip()
    # Strip .pdf from doc_name if present so we don't get double .pdf
    name = re.sub(r'\.pdf$', '', doc_name, flags=re.IGNORECASE)
    date_stripped = re.sub(r'\s*(EST|CST|MST|PST|EDT|CDT|MDT|PDT)\s*$', '', date_text.strip(), flags=re.IGNORECASE)
    date_clean = re.sub(r'[<>:"/\\|?*]', '_', date_stripped) if date_stripped else ""
    raw = f"{company_short} - {name} - {date_clean}" if date_clean else f"{company_short} - {name}"
    safe = re.sub(r'[<>:"/\\|?*]', '_', raw)
    safe = re.sub(r'\s*-\s*', '-', safe)
    safe = safe.replace('.', '')
    safe = safe.replace(' ', '_').lower()
    return safe + ".pdf"


# ---------------------------------------------------------------------------
# Core: process one page of results (click each link, download, go back)
# ---------------------------------------------------------------------------

def _close_extra_pages(page):
    """Close all browser tabs except the main results page."""
    for p in page.context.pages:
        if p != page:
            try:
                p.close()
            except Exception:
                pass


def _try_download(page, doc, filepath, label):
    """Attempt to download a single document. Returns True on success.

    Uses expect_download on the main page. If the page navigates away
    on failure, restores it via go_back.
    """
    print(f"  {label} ...", end=" ", flush=True)
    url_before = page.url
    try:
        rows = page.locator(SEL_RESULTS_ROWS)
        row = rows.nth(doc["row_index"])
        link = row.locator(SEL_DOWNLOAD_LINK).first

        with page.expect_download(timeout=180000) as download_info:
            link.click()
        download = download_info.value
        download.save_as(str(filepath))
        print(f"OK ({filepath.stat().st_size // 1024} KB)")
        # If the page navigated during download, go back
        if page.url != url_before:
            page.go_back(wait_until="networkidle", timeout=15000)
        return True

    except PlaywrightTimeout:
        print("FAILED (timeout)")
        if page.url != url_before:
            try:
                page.go_back(wait_until="networkidle", timeout=15000)
            except Exception:
                pass
        _close_extra_pages(page)

    except Exception as e:
        print(f"ERROR: {e}")
        if page.url != url_before:
            try:
                page.go_back(wait_until="networkidle", timeout=15000)
            except Exception:
                pass
        _close_extra_pages(page)

    return False


def process_results_page(page, output_path: Path, min_size_kb: int,
                         dry_run: bool, page_num: int, running_total: int):
    """Process all document rows on the current results page.

    For each qualifying row: click the document link, wait for the PDF
    to load, save it, then navigate back to the results page.
    Failed downloads are retried once at the end of the page.

    Returns (downloaded_files, skipped_count, total_rows_on_page).
    """
    rows = page.locator(SEL_RESULTS_ROWS)
    count = rows.count()
    downloaded = []
    skipped = 0

    # First pass: identify document rows (those with a resource.html link)
    doc_rows = []
    for i in range(count):
        row = rows.nth(i)
        cells = row.locator("td")
        if cells.count() < 7:
            continue
        link = row.locator(SEL_DOWNLOAD_LINK).first
        try:
            href = link.get_attribute("href", timeout=1000)
        except Exception:
            href = None
        if not href:
            continue

        cell_texts = [cells.nth(j).inner_text().strip() for j in range(cells.count())]
        company = cell_texts[COL_COMPANY]
        doc_name = cell_texts[COL_DOC_NAME]
        date_text = cell_texts[COL_DATE].split("\n")[0]
        size_text = cell_texts[COL_SIZE]
        size_kb = parse_size_to_kb(size_text)

        doc_rows.append({
            "row_index": i,
            "company": company,
            "name": doc_name,
            "date": date_text,
            "size_text": size_text,
            "size_kb": size_kb,
        })

    print(f"\n  Page {page_num}: {len(doc_rows)} documents found")

    failed = []
    consecutive_fails = 0

    for idx, doc in enumerate(doc_rows):
        n = running_total + idx + 1
        filename = safe_filename(doc["company"], doc["name"], doc["date"])
        filepath = output_path / filename
        company_short = doc["company"].split("/")[0].strip()

        # Size filter
        if min_size_kb > 0 and doc["size_kb"] is not None and doc["size_kb"] < min_size_kb:
            print(f"  [{n}] [small] {doc['size_text']} | {doc['name']}")
            skipped += 1
            continue

        # Already downloaded
        if filepath.exists():
            print(f"  [{n}] [skip] {filename}")
            skipped += 1
            continue

        if dry_run:
            print(f"  [{n}] [dry-run] {company_short} | {doc['name']} | {doc['size_text']}")
            continue

        label = f"[{n}] {company_short} | {doc['size_text']}"
        if _try_download(page, doc, filepath, label):
            downloaded.append(str(filepath))
            consecutive_fails = 0
        else:
            failed.append((n, doc, filepath))
            consecutive_fails += 1
            # Backoff: if 3+ consecutive failures, likely throttled
            if consecutive_fails >= 3:
                wait = 30 * consecutive_fails
                print(f"  [backoff] {consecutive_fails} consecutive failures — waiting {wait}s...")
                time.sleep(wait)

        _human_delay(3, 7)

    # Retry failed downloads once
    if failed:
        print(f"\n  Retrying {len(failed)} failed download(s) on page {page_num}...")
        _human_delay(2, 10)
        still_failed = []
        for n, doc, filepath in failed:
            company_short = doc["company"].split("/")[0].strip()
            label = f"[{n}] [retry] {company_short} | {doc['size_text']}"
            if _try_download(page, doc, filepath, label):
                downloaded.append(str(filepath))
            else:
                still_failed.append(safe_filename(doc["company"], doc["name"], doc["date"]))
            _human_delay(2, 10)

        if still_failed:
            print(f"\n  {len(still_failed)} file(s) still failed after retry on page {page_num}:")
            for name in still_failed:
                print(f"    - {name}")

    return downloaded, skipped, len(doc_rows)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def _get_first_row_text(page):
    """Get text from the first result row to detect if the page actually changed."""
    try:
        first_row = page.locator(SEL_RESULTS_ROWS).first
        return first_row.inner_text(timeout=3000).strip()
    except Exception:
        return ""


def _wait_for_page_change(page, before_text, timeout_s=90):
    """Poll until the first row text changes, indicating new page content loaded."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        after = _get_first_row_text(page)
        if after and after != before_text:
            return True
        page.wait_for_timeout(1500)
    return False


def _debug_pagination(page):
    """Print what pagination elements exist on the page."""
    print("\n  [debug] Pagination elements found:")
    try:
        # Dump the pagination HTML
        pagination_html = page.evaluate("""() => {
            // Look for common pagination containers
            const selectors = [
                'nav[aria-label*="pag"]', '.pagination', '[class*="pager"]',
                '[class*="pagina"]', 'ul.pagination', 'nav.pagination',
                '[role="navigation"]'
            ];
            for (const sel of selectors) {
                const el = document.querySelector(sel);
                if (el) return `Selector: ${sel}\\n${el.outerHTML.substring(0, 1000)}`;
            }
            // Fallback: find any element containing page numbers
            const links = document.querySelectorAll('a');
            const pageLinks = [];
            for (const a of links) {
                const text = a.textContent.trim();
                if (/^\\d+$/.test(text) || /next/i.test(text) || /prev/i.test(text)) {
                    pageLinks.push(`<a class="${a.className}" href="${a.href}">${text}</a>`);
                }
            }
            if (pageLinks.length) return `Page-like links:\\n${pageLinks.join('\\n')}`;
            return 'No pagination elements found';
        }""")
        print(f"  {pagination_html}")
    except Exception as e:
        print(f"  [debug] Error inspecting pagination: {e}")


def _try_prev_page(page):
    """Attempt to navigate to the previous results page.

    Returns True if navigation succeeded and results changed, False otherwise.
    """
    before = _get_first_row_text(page)

    # Primary: find the active page number and click the previous one
    try:
        active = page.locator(
            'li.active a, a[aria-current="page"], '
            '.pagination .active, span.current'
        ).first
        if active.is_visible(timeout=2000):
            current_text = active.inner_text().strip()
            if current_text.isdigit():
                prev_num = int(current_text) - 1
                if prev_num < 1:
                    return False
                prev_link = page.locator(f'a:text-is("{prev_num}")').first
                if prev_link.is_visible(timeout=2000):
                    prev_link.scroll_into_view_if_needed()
                    _human_delay(1, 3)
                    prev_link.click()
                    if _wait_for_page_change(page, before):
                        return True
                    print(f"\n  [warn] Page content didn't change after clicking page {prev_num}")
    except (PlaywrightTimeout, Exception) as e:
        print(f"\n  [debug] Prev page number approach failed: {e}")

    # Fallback: try Previous button selectors
    prev_btn = page.locator(
        'a:has-text("Previous"), button:has-text("Previous"), '
        'li.previous a, a[aria-label="Previous"]'
    ).first
    try:
        if prev_btn.is_visible(timeout=3000):
            prev_btn.scroll_into_view_if_needed()
            _human_delay(1, 3)
            prev_btn.click()
            if _wait_for_page_change(page, before):
                return True
            print("\n  [warn] Page content didn't change after clicking Previous")
    except (PlaywrightTimeout, Exception):
        pass

    _debug_pagination(page)
    return False


def _try_next_page(page):
    """Attempt to navigate to the next results page.

    Returns True if navigation succeeded and results changed, False otherwise.
    """
    before = _get_first_row_text(page)

    # Primary: find the active page number and click the next one
    try:
        active = page.locator(
            'li.active a, a[aria-current="page"], '
            '.pagination .active, span.current'
        ).first
        if active.is_visible(timeout=2000):
            current_text = active.inner_text().strip()
            if current_text.isdigit():
                next_num = int(current_text) + 1
                next_link = page.locator(f'a:text-is("{next_num}")').first
                if next_link.is_visible(timeout=2000):
                    next_link.scroll_into_view_if_needed()
                    _human_delay(1, 3)
                    next_link.click()
                    if _wait_for_page_change(page, before):
                        return True
                    print(f"\n  [warn] Page content didn't change after clicking page {next_num}")
        else:
            print("\n  [debug] No active page number element found")
    except (PlaywrightTimeout, Exception) as e:
        print(f"\n  [debug] Page number approach failed: {e}")

    # Fallback: try Next button selectors
    next_btn = page.locator(
        'a:has-text("Next"), button:has-text("Next"), '
        'li.next a, a[aria-label="Next"]'
    ).first
    try:
        if next_btn.is_visible(timeout=3000):
            next_btn.scroll_into_view_if_needed()
            _human_delay(1, 3)
            next_btn.click()
            if _wait_for_page_change(page, before):
                return True
            print("\n  [warn] Page content didn't change after clicking Next")
    except (PlaywrightTimeout, Exception):
        pass

    # Both approaches failed — dump pagination HTML for debugging
    _debug_pagination(page)
    return False


def _click_page_number(page, num):
    """Click a specific page number link in the pagination. Returns True on success."""
    try:
        before = _get_first_row_text(page)
        link = page.locator(f'a:text-is("{num}")').first
        if link.is_visible(timeout=3000):
            link.scroll_into_view_if_needed()
            _human_delay(1, 2)
            link.click()
            if not _wait_for_page_change(page, before):
                return False
            _human_delay(1, 2)
            return True
    except (PlaywrightTimeout, Exception):
        pass
    return False


def _skip_to_page(page, target_page):
    """Jump to target page by clicking visible page number links (range +/- 5)."""
    print(f"  Jumping to page {target_page}...")
    current = 1
    while current < target_page:
        # Jump as far as possible — pick the highest visible page number
        # that doesn't overshoot the target
        jump_to = min(current + 5, target_page)
        # Try from the highest possible jump down to current+1
        jumped = False
        for num in range(jump_to, current, -1):
            print(f"    Clicking page {num}...", end=" ", flush=True)
            if _click_page_number(page, num):
                print("OK")
                current = num
                jumped = True
                break
            else:
                print("not visible")
        if not jumped:
            print(f"    FAILED — stuck at page {current}, could not advance")
            return current
        _human_delay(1, 3)
    return target_page


def _skip_to_last_page(page):
    """Click the last page number link (always visible in pagination).

    Returns the last page number reached, or 1 if it couldn't find one.
    """
    try:
        # Find all numeric page links and click the highest one
        page_links = page.locator('a').all()
        max_num = 0
        for link in page_links:
            try:
                text = link.inner_text(timeout=500).strip()
                if text.isdigit():
                    max_num = max(max_num, int(text))
            except Exception:
                continue

        if max_num > 1:
            print(f"  Last page detected: {max_num}")
            if _click_page_number(page, max_num):
                print(f"  Jumped to page {max_num}")
                return max_num
            else:
                print(f"  [warn] Could not click page {max_num}")
    except Exception as e:
        print(f"  [warn] Could not find last page: {e}")
    return 1


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_downloader(args):
    """Launch browser, navigate, apply filters, click-download each document."""
    instance_label = f" [{args.instance}]" if args.instance else ""
    print(f"SEDAR+ Downloader{instance_label}")
    print(f"  URL: {args.url}")
    print(f"  Min size: {args.min_size_kb} KB")
    print(f"  Output: {args.output_dir}")
    print(f"  Max pages: {args.max_pages}")
    if args.start_page > 1:
        print(f"  Starting from page: {args.start_page}")
    if args.reverse:
        print(f"  Direction: REVERSE (descending page numbers)")
    if args.dry_run:
        print(f"  Mode: DRY RUN")
    print()

    output_path = Path(args.output_dir)
    if args.subfolder:
        output_path = output_path / args.subfolder
    output_path.mkdir(parents=True, exist_ok=True)

    all_downloaded = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=not args.headed,
            slow_mo=args.slow_mo or 0,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        context = _load_session(browser, args)
        page = context.new_page()
        _apply_stealth(page)

        # Navigate to the SEDAR+ page
        print("Navigating to SEDAR+ page...")
        page.goto(args.url, wait_until="domcontentloaded", timeout=30000)

        # Bot challenge — headed mode (skip in --manual; user drives everything)
        if args.headed and not args.manual:
            print("\n" + "=" * 60)
            print("Complete any bot challenge in the browser, then press Enter.")
            print("=" * 60)
            _wait_for_user(page, "\nPress Enter when the page is ready...", 600000)

        if not args.manual:
            page.wait_for_load_state("networkidle", timeout=30000)
            print("Page loaded.\n")

        # Apply filters (if any)
        if args.manual:
            _wait_for_user(page,
                "\nNavigate to SEDAR search -> Documents, set filters, run search, "
                "then click START DOWNLOADING.", 600000)
        else:
            print("Applying filters...")
            apply_filters(page, args)
            if args.headed:
                _wait_for_user(page, "\nFilters applied. Press Enter to start downloading...", 600000)

        # Save session after bot challenge is solved
        _save_session(context, args.instance)

        # Skip to start page
        if args.reverse and args.manual:
            # User has already navigated to the desired last page manually.
            # Use max_pages as the starting counter so the loop iterates that many times.
            last_page = args.max_pages
            actual_start = args.max_pages
        elif args.reverse:
            # Reverse mode: jump to last page first, then walk back if needed
            last_page = _skip_to_last_page(page)
            if args.start_page > 1 and args.start_page < last_page:
                # User wants to start from a specific page, walk back to it
                print(f"  Walking back from page {last_page} to {args.start_page}...")
                actual_start = last_page
                while actual_start > args.start_page:
                    if not _try_prev_page(page):
                        break
                    actual_start -= 1
                    print(f"    Reached page {actual_start}")
                    _human_delay(2, 7)
            else:
                actual_start = last_page
        elif args.start_page > 1:
            actual_start = _skip_to_page(page, args.start_page)
            if actual_start < args.start_page:
                print(f"  [warn] Could only reach page {actual_start}")
        else:
            actual_start = 1

        # Process results page by page
        running_total = (actual_start - 1) * 30  # approximate docs per page
        consecutive_all_skipped = 0

        if args.reverse:
            # Reverse: count down from start_page
            end_page = max(actual_start - args.max_pages + 1, 1)
            page_range = range(actual_start, end_page - 1, -1)
        else:
            page_range = range(actual_start, actual_start + args.max_pages)

        for page_num in page_range:
            downloaded, skipped, row_count = process_results_page(
                page, output_path, args.min_size_kb, args.dry_run,
                page_num, running_total,
            )
            all_downloaded.extend(downloaded)
            running_total += row_count

            if row_count == 0:
                print("  No documents found on page — stopping.")
                break

            # Track if we're only skipping (all already downloaded)
            if len(downloaded) == 0 and skipped == row_count:
                consecutive_all_skipped += 1
            else:
                consecutive_all_skipped = 0

            # If all docs on this page were already downloaded, we've
            # overlapped with the other instance — stop early (unless disabled)
            if not args.ignore_overlap and consecutive_all_skipped >= 3:
                print(f"\n  3 consecutive pages fully skipped — likely met the other instance. Stopping.")
                break

            # Save session periodically (every 5 pages)
            if page_num % 5 == 0:
                _save_session(context, args.instance)

            # Navigate to next/previous page
            if args.reverse:
                nav_ok = _try_prev_page(page)
                if not nav_ok:
                    print("\n  [warn] Prev navigation failed — waiting 15s and retrying once...")
                    time.sleep(15)
                    nav_ok = _try_prev_page(page)
                if not nav_ok:
                    print("\n  No more pages (pagination ended).")
                    break
            else:
                if not _try_next_page(page):
                    print("\n  No more pages (pagination ended).")
                    break

            # Extra delay between pages
            _human_delay(2, 7)

        # Final session save
        _save_session(context, args.instance)
        browser.close()

    # Cleanup small files (in case actual size differs from reported)
    if all_downloaded and args.min_size_kb > 0 and not args.dry_run:
        print(f"\nCleaning up files under {args.min_size_kb} KB...")
        removed = 0
        for f in output_path.glob("*.pdf"):
            size_kb = f.stat().st_size / 1024
            if size_kb < args.min_size_kb:
                print(f"  Removing {f.name} ({size_kb:.0f} KB)")
                f.unlink()
                removed += 1
        if removed:
            print(f"  Removed {removed} small files")

    print(f"\nDone. {len(all_downloaded)} files downloaded to {output_path}")


def apply_filters(page, args):
    """Fill in search form fields and submit (best-effort, warns on failures)."""
    # Submit search (the main action — even without specific filters)
    try:
        btn = page.locator(SEL_SEARCH_BUTTON).first
        btn.wait_for(timeout=5000)
        btn.click()
        print("  Submitted search form")
        page.wait_for_load_state("networkidle", timeout=15000)
    except PlaywrightTimeout:
        print("  [warn] Search button not found or page did not settle")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download documents from SEDAR+ filing profile pages"
    )
    parser.add_argument("--url", required=True, help="SEDAR+ profile page URL")
    parser.add_argument("--min-size-kb", type=int, default=7000,
                        help="Minimum file size in KB to download (default: 7000)")
    parser.add_argument("--output-dir", default="../data/sedar_downloads",
                        help="Download output directory (default: ../data/sedar_downloads)")
    parser.add_argument("--subfolder", default=None,
                        help="Optional subfolder within output-dir")
    parser.add_argument("--headed", action="store_true",
                        help="Show browser (needed for bot challenge)")
    parser.add_argument("--slow-mo", type=int, default=None,
                        help="Delay between Playwright actions in ms")
    parser.add_argument("--dry-run", action="store_true",
                        help="List matching documents without downloading")
    parser.add_argument("--start-page", type=int, default=1,
                        help="Page number to start from (default: 1)")
    parser.add_argument("--max-pages", type=int, default=200,
                        help="Maximum pages to process (default: 200)")
    parser.add_argument("--reverse", action="store_true",
                        help="Process pages in reverse (descending). Use with --start-page to work backward.")
    parser.add_argument("--instance", default="",
                        help="Instance name for separate session files (e.g. 'fwd', 'rev'). "
                             "Allows running multiple browsers simultaneously.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore saved session and start with fresh cookies")
    parser.add_argument("--manual", action="store_true",
                        help="Skip auto-filter submission; user drives navigation/filtering in browser")
    parser.add_argument("--ignore-overlap", action="store_true",
                        help="Do not stop after 3 consecutive fully-skipped pages (walk the full range)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_downloader(args)
