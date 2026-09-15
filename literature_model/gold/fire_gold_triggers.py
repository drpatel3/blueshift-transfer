"""Fire gold PDF Lambda triggers via copy-in-place + fresh uploads.

Two batches per run:
  - copy-in-place re-trigger for stems in gold_need_processing.txt (already
    in s3://.../pdfs/{stem}.pdf, just need a fresh PutObject event)
  - fresh upload for stems in gold_missing_upload.txt, sourced from
    literature_platform/data/sedar_downloads/{stem}.pdf

Lambda env has SKIP_LLM=true, so each fired doc gets images/tables/text
extracted but no LLM stages. Run extract_pfs_llm.py afterwards to fill in
the LLM data block.
"""
import boto3
import concurrent.futures
import time
from pathlib import Path

BUCKET = "mineral-pipeline-pipeline"
HERE = Path(__file__).resolve().parent
# literature_platform/data/sedar_downloads — three levels up from dev/gold/
PDF_SOURCE_DIR = HERE.resolve().parents[2] / "data" / "sedar_downloads"


def fire_copy(stem):
    s3 = boto3.client("s3")  # fresh client per thread
    key = f"pdfs/{stem}.pdf"
    try:
        s3.copy_object(
            Bucket=BUCKET,
            Key=key,
            CopySource={"Bucket": BUCKET, "Key": key},
            MetadataDirective="REPLACE",
        )
        return stem, "ok", None
    except Exception as e:
        return stem, "err", str(e)


def fire_upload(stem):
    s3 = boto3.client("s3")
    local = PDF_SOURCE_DIR / f"{stem}.pdf"
    if not local.exists():
        return stem, "err", f"local file missing at {local}"
    try:
        s3.upload_file(str(local), BUCKET, f"pdfs/{stem}.pdf")
        return stem, "ok", None
    except Exception as e:
        return stem, "err", str(e)


def main():
    reprocess = (HERE / "gold_need_processing.txt").read_text().splitlines()
    upload = (HERE / "gold_missing_upload.txt").read_text().splitlines()
    reprocess = [s for s in reprocess if s]
    upload = [s for s in upload if s]

    print(f"Re-trigger (copy-in-place): {len(reprocess)}")
    print(f"Upload fresh:               {len(upload)}")
    print(f"Total:                      {len(reprocess) + len(upload)}")
    print()

    t0 = time.time()
    oks = errs = 0
    errors_sample = []

    print("Firing copy-in-place triggers (8 threads)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(fire_copy, s) for s in reprocess]
        for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
            stem, status, err = f.result()
            if status == "ok":
                oks += 1
            else:
                errs += 1
                if len(errors_sample) < 5:
                    errors_sample.append((stem, err))
            if i % 50 == 0:
                print(f"  {i}/{len(reprocess)} fired ({time.time()-t0:.1f}s)")
    print(f"Copy-triggers: {oks} ok, {errs} err, {time.time()-t0:.1f}s")
    for s, e in errors_sample:
        print(f"  err sample: {s}: {e}")

    print()
    print("Uploading missing PDFs...")
    for stem in upload:
        s, status, err = fire_upload(stem)
        marker = "ok" if status == "ok" else f"ERR: {err}"
        print(f"  {stem}: {marker}")

    print()
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print("Lambdas running in parallel. Watch s3://.../results/")


if __name__ == "__main__":
    main()
