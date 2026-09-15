"""Monitor gold extraction + quota approval. Emits only on state changes."""
import sys
import time

import boto3

REQ_ID = "6182972d41eb42a6a228cc79dbf56671kZ6RJCD2"
TARGET = 1548
BUCKET = "mineral-pipeline-pipeline"
REGION = "us-east-1"

s3 = boto3.client("s3")
sq = boto3.client("service-quotas", region_name=REGION)


def get_count():
    try:
        paginator = s3.get_paginator("list_objects_v2")
        n = 0
        for page in paginator.paginate(Bucket=BUCKET, Prefix="results/"):
            n += len(page.get("Contents", []))
        return n
    except Exception as e:
        print(f"POLL_ERROR count: {e}", flush=True)
        return None


def get_status():
    try:
        r = sq.get_requested_service_quota_change(RequestId=REQ_ID)
        return r["RequestedQuota"]["Status"]
    except Exception as e:
        print(f"POLL_ERROR status: {e}", flush=True)
        return None


def main():
    baseline = get_count()
    status = get_status()
    if baseline is None or status is None:
        print("WATCH_START failed to get initial state — retrying", flush=True)
        time.sleep(10)
        baseline = get_count() or 0
        status = get_status() or "UNKNOWN"

    prev_count = baseline
    prev_status = status
    print(
        f"WATCH_START baseline={baseline} target={TARGET} "
        f"remaining={TARGET - baseline} status={prev_status}",
        flush=True,
    )

    while True:
        time.sleep(180)
        count = get_count()
        status = get_status()

        if status and status != prev_status:
            print(f"QUOTA: {prev_status} -> {status}", flush=True)
            prev_status = status

        if count is not None:
            delta = count - prev_count
            if delta >= 50:
                print(f"PROGRESS: {count} / {TARGET} (+{delta})", flush=True)
                prev_count = count
            if count >= TARGET:
                print(f"COMPLETE: {count} / {TARGET}", flush=True)
                return


if __name__ == "__main__":
    main()
