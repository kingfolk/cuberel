#!/usr/bin/env python3
"""Run isolated F8 or bounded F1 SQL acceptance cases.

Build the release sql_acceptance example first. Each case starts a fresh process.
'first' is process-cold, not a claim of a cold driver/disk cache. Comparisons use
matched SQL fixtures; processes run sequentially and cache state is not cleared.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys


def cases(quick=False):
    if quick:
        return [("project", 65, 50, 20, 3, 2), ("aggregate", 65, 100, 100, 3, 2),
                ("project", 0, 100, 0, 1, 1), ("mixed", 65, 50, 20, 1, 1)]
    result = []
    for w in ("project", "aggregate"):
        for n in (0, 1024, 65536, 1048576, 2097152):
            result.append((w, n, 50, 20, 1, 1))
        for p in (0, 1, 100):
            result.append((w, 65536, p, 20, 1, 1))
        for null in (0, 100):
            result.append((w, 65536, 50, null, 1, 1))
        for parts in (1, 2):
            result.append((w, 65536, 50, 20, 4, parts))
    result += [("mixed", 65536, 50, 20, 1, 1), ("mixed", 65536, 50, 20, 4, 2)]
    return result


def batch_study_cases():
    return [("aggregate", rows, 50, 20, 1, 1) for rows in (65536, 1048576, 2097152)]


def positive_int(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return str(error)


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--warm", type=positive_int, default=5)
    parser.add_argument("--repeats", type=positive_int, default=1,
                        help="independent fresh processes per case/backend/batch size")
    parser.add_argument("--batch-sizes", nargs="+", type=positive_int, default=[8192],
                        help="SQL DataFusion session batch sizes")
    parser.add_argument("--batch-study", action="store_true",
                        help="run aggregate 64K/1M/2M, one source batch, one partition, pass 50, NULL 20")
    parser.add_argument("--study-rows", nargs="+", type=positive_int,
                        help="optional total-row subset for --batch-study")
    parser.add_argument("--backends", nargs="+", default=["datafusion", "cpu", "metal"],
                        choices=["datafusion", "cpu", "metal"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("--batch-sizes must not contain duplicates")
    if len(set(args.backends)) != len(args.backends):
        parser.error("--backends must not contain duplicates")
    if args.study_rows is not None and (not args.batch_study or len(set(args.study_rows)) != len(args.study_rows)):
        parser.error("--study-rows requires --batch-study and unique row counts")
    binary = args.binary.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": sys.argv, "platform": platform.platform(),
        "hardware": command_output(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"]),
        "os": command_output(["sw_vers"]), "rust": command_output(["rustc", "-Vv"]),
        "cargo": command_output(["cargo", "-V"]),
        "base_commit": command_output(["git", "rev-parse", "HEAD"]),
        "working_diff": command_output(["git", "diff", "--stat"]),
        "working_status": command_output(["git", "status", "--short"]),
        "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          for name in ("examples/sql_acceptance.rs", "examples/support/sql_acceptance.rs",
                                       "scripts/sql_acceptance.py", "Cargo.toml", "Cargo.lock")},
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "binary": str(binary), "profile": "release (build command recorded in report)",
        "environment": {k: v for k, v in os.environ.items()
                        if k in ("RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS", "CUBECL_LOG_LEVEL")},
        "warm_samples": args.warm, "repeats": args.repeats,
        "batch_sizes": args.batch_sizes, "batch_study": args.batch_study,
        "study_rows": args.study_rows,
        "note": "sequential processes; OS/driver caches not cleared; no concurrent queries",
    }
    (args.out / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    all_rows = []
    summaries = []
    jobs = []
    selected_cases = batch_study_cases() if args.batch_study else cases(args.quick)
    if args.study_rows is not None:
        selected_cases = [("aggregate", rows, 50, 20, 1, 1) for rows in args.study_rows]
    for case in selected_cases:
        dsl_case = case[0] != "mixed" and (args.quick or case[2:] == (50, 20, 1, 1))
        for repeat in range(1, args.repeats + 1):
            offset = (repeat - 1) % len(args.backends)
            backend_order = args.backends[offset:] + args.backends[:offset]
            # Batch study deliberately excludes DSL, since its execution path
            # cannot accept the SQL SessionConfig batch-size override.
            if args.batch_study:
                for batch_size in args.batch_sizes:
                    for backend in backend_order:
                        jobs.append((backend, "sql", case, batch_size, repeat))
                continue
            for batch_size in args.batch_sizes:
                for backend in backend_order:
                    jobs.append((backend, "sql", case, batch_size, repeat))
            # Keep the original DSL diagnostic once per original fixture/repeat,
            # independent of how many SQL batch sizes were requested.
            if dsl_case:
                for backend in backend_order:
                    if backend != "datafusion":
                        jobs.append((backend, "dsl", case, None, repeat))

    for i, (backend, path, case, batch_size, repeat) in enumerate(jobs):
        command = [str(binary), backend, path, *map(str, case), str(args.warm)]
        if path == "sql":
            command.append(str(batch_size))
        parts = [backend, path, *map(str, case), f"repeat{repeat}"]
        if batch_size is not None:
            parts.append(f"batch{batch_size}")
        name = "-".join(map(str, parts))
        print(f"[{i+1}/{len(jobs)}] {name}", flush=True)
        completed = subprocess.run(command, capture_output=True, text=True)
        (args.out / f"{name}.log").write_text(
            "command=" + json.dumps(command) + "\n" + completed.stderr)
        (args.out / f"{name}.csv").write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError(f"case failed ({completed.returncode}): {name}; see log")
        rows = list(csv.DictReader(completed.stdout.splitlines()))
        if len(rows) != args.warm + 1 or rows[0]["phase"] != "first":
            raise RuntimeError(f"incomplete samples: {name}")
        combined_rows = [dict(row, repeat=repeat) for row in rows]
        all_rows.extend(combined_rows)
        first, warm = rows[0], rows[1:]
        summary = {k: first[k] for k in ("backend", "path", "workload", "rows", "pass_percent",
                                        "null_percent", "batches", "partitions", "session_batch_size")}
        summary["repeat"] = repeat
        summary["warm_samples"] = len(warm)
        for key in ("init_ms", "prepare_ms", "execute_ms", "total_ms"):
            summary["first_" + key] = first[key]
        for key in ("prepare_ms", "execute_ms", "total_ms"):
            values = [float(row[key]) for row in warm]
            summary["warm_p50_" + key] = f"{percentile(values, .5):.6f}"
            summary["warm_p95_" + key] = f"{percentile(values, .95):.6f}"
        for key in ("output_rows", "input_batches", "completed_requests",
                    "actual_fused_requests", "actual_unfused_requests",
                    "upload_calls", "upload_bytes", "readback_calls", "readback_bytes",
                    "allocation_calls", "allocation_bytes", "launches", "count_reads",
                    "define_calls", "queue_wait_ns", "adapter_prepare_ns", "execute_export_ns", "worker_ns",
                    "arrow_adapt_ns", "resource_preflight_ns", "upload_submit_ns", "complete_export_ns",
                    "peak_device_bytes", "peak_host_bytes", "max_request_device_bytes"):
            summary["warm_last_" + key] = warm[-1][key]
        summaries.append(summary)
        # Checkpoint combined outputs and summaries after every successful process.
        for filename, records in (("samples.csv", all_rows), ("summary.csv", summaries)):
            with (args.out / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    (args.out / "COMPLETE").write_text(
        f"{len(jobs)} processes, {len(all_rows)} verified samples\n")


if __name__ == "__main__":
    main()
