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
import re
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


CASE_KEYS = {"id", "workload", "rows", "pass_percent", "null_percent", "batches",
             "partitions", "batch_sizes", "readback_modes", "shared_host_bytes",
             "shared_device_bytes"}
U64_MAX = 2**64 - 1
I32_MAX = 2**31 - 1
CSV_IDENTITY = ("backend", "path", "workload", "rows", "pass_percent", "null_percent",
                "batches", "partitions", "session_batch_size")
TIMING_COLUMNS = ("init_ms", "prepare_ms", "execute_ms", "total_ms")


def _exact_int(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} through {high}")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_manifest(path):
    """Read and fully validate the frozen SQL-only v1 JSON manifest."""
    raw = Path(path).read_bytes()
    try:
        document = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid manifest JSON: {error}") from error
    if not isinstance(document, dict) or set(document) != {"version", "description", "cases"}:
        raise ValueError("manifest root must contain exactly version, description, cases")
    if type(document["version"]) is not int or document["version"] != 1:
        raise ValueError("manifest version must be integer 1")
    if not isinstance(document["description"], str) or not document["description"].strip():
        raise ValueError("manifest description must be a nonempty string")
    if not isinstance(document["cases"], list) or not document["cases"]:
        raise ValueError("manifest cases must be a nonempty list")
    seen = set()
    validated = []
    for index, case in enumerate(document["cases"]):
        label = f"cases[{index}]"
        if not isinstance(case, dict) or set(case) != CASE_KEYS:
            raise ValueError(f"{label} must contain exactly: {', '.join(sorted(CASE_KEYS))}")
        case_id = case["id"]
        if not isinstance(case_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", case_id) is None:
            raise ValueError(f"{label}.id is invalid")
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        if case["workload"] not in ("project", "aggregate", "mixed"):
            raise ValueError(f"{label}.workload is invalid")
        out = dict(case)
        for field in ("rows",):
            _exact_int(case[field], f"{label}.{field}", 0, I32_MAX)
        for field in ("pass_percent", "null_percent"):
            _exact_int(case[field], f"{label}.{field}", 0, 100)
        for field in ("batches", "partitions"):
            _exact_int(case[field], f"{label}.{field}", 1, I32_MAX)
        sizes = case["batch_sizes"]
        if not isinstance(sizes, list) or not sizes:
            raise ValueError(f"{label}.batch_sizes must be a nonempty list")
        for value in sizes:
            _exact_int(value, f"{label}.batch_sizes item", 1, I32_MAX)
        if len(set(sizes)) != len(sizes):
            raise ValueError(f"{label}.batch_sizes must be unique")
        modes = case["readback_modes"]
        if not isinstance(modes, list) or not modes or any(m not in ("scalar", "batched") for m in modes):
            raise ValueError(f"{label}.readback_modes must be a nonempty subset of scalar, batched")
        if len(set(modes)) != len(modes):
            raise ValueError(f"{label}.readback_modes must be unique")
        for field in ("shared_host_bytes", "shared_device_bytes"):
            _exact_int(case[field], f"{label}.{field}", 1, U64_MAX)
        validated.append(out)
    return raw, {"version": 1, "description": document["description"], "cases": validated}


def manifest_jobs(manifest, backends, repeats):
    """Expand a manifest with batch-study ordering and no DSL jobs."""
    jobs = []
    for case in manifest["cases"]:
        case_tuple = (case["workload"], case["rows"], case["pass_percent"], case["null_percent"],
                      case["batches"], case["partitions"])
        for repeat in range(1, repeats + 1):
            offset = (repeat - 1) % len(backends)
            backend_order = backends[offset:] + backends[:offset]
            mode_offset = (repeat - 1) % len(case["readback_modes"])
            mode_order = case["readback_modes"][mode_offset:] + case["readback_modes"][:mode_offset]
            for batch_size in case["batch_sizes"]:
                for backend in backend_order:
                    modes = (mode_order if backend in ("cpu", "metal") else (None,))
                    for mode in modes:
                        jobs.append({"backend": backend, "path": "sql", "case": case_tuple,
                                     "batch_size": batch_size, "repeat": repeat,
                                     "readback_mode": mode, "case_id": case["id"],
                                     "shared_host_bytes": case["shared_host_bytes"],
                                     "shared_device_bytes": case["shared_device_bytes"]})
    return jobs


def build_command(binary, job, warm_samples):
    command = [str(binary), job["backend"], job["path"], *map(str, job["case"]), str(warm_samples)]
    if job["path"] == "sql":
        command.append(str(job["batch_size"]))
    if job["readback_mode"] is not None:
        command.append(job["readback_mode"])
    if job.get("manifest"):
        command.extend(["--shared-host-bytes", str(job["shared_host_bytes"]),
                        "--shared-device-bytes", str(job["shared_device_bytes"])])
    return command


def validate_output(rows, job, warm_samples):
    """Validate CSV contents against the requested job before accepting samples."""
    name = job["name"]
    if not rows or len(rows) != warm_samples + 1:
        raise RuntimeError(f"incomplete samples: {name}")
    expected = dict(zip(CSV_IDENTITY, (job["backend"], job["path"], *job["case"],
                                      job["batch_size"] if job["path"] == "sql" else "")))
    # Exact string comparisons catch reordered/mislabeled outputs and blank SQL sizes.
    for index, row in enumerate(rows):
        for field, value in expected.items():
            if row.get(field) != str(value):
                raise RuntimeError(f"unexpected {field} at sample {index}: {name}: {row.get(field)!r}")
        if row.get("phase") != ("first" if index == 0 else "warm") or row.get("sample") != str(index):
            raise RuntimeError(f"unexpected phase/sample sequence: {name}")
        if job["backend"] == "datafusion":
            if row.get("aggregate_readback_mode") != "":
                raise RuntimeError(f"DataFusion readback mode must be blank: {name}")
        else:
            mode = job["readback_mode"] or "scalar"
            if row.get("aggregate_readback_mode") != mode:
                raise RuntimeError(f"unexpected aggregate readback mode: {name}")
        for field in TIMING_COLUMNS:
            try:
                value = float(row[field])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"invalid {field} at sample {index}: {name}") from error
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(f"invalid {field} at sample {index}: {name}")
            if field == "init_ms" and ((index == 0 and value < 0) or (index > 0 and value != 0)):
                raise RuntimeError(f"init_ms must be present only on first sample: {name}")
        if "case_id" in row and row["case_id"] != job["case_id"]:
            raise RuntimeError(f"unexpected case_id: {name}")
        if job.get("manifest"):
            for field, expected_value in (("shared_host_limit_bytes", job["shared_host_bytes"]),
                                          ("shared_device_limit_bytes", job["shared_device_bytes"])):
                if row.get(field) != str(expected_value):
                    raise RuntimeError(f"unexpected {field}: {name}")
    return rows


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
    parser.add_argument("--readback-modes", nargs="+", choices=["scalar", "batched"], default=["scalar"],
                        help="aggregate readback modes; nondefault modes require --batch-study")
    parser.add_argument("--study-rows", nargs="+", type=positive_int,
                        help="optional total-row subset for --batch-study")
    parser.add_argument("--backends", nargs="+", default=["datafusion", "cpu", "metal"],
                        choices=["datafusion", "cpu", "metal"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--manifest", type=Path,
                        help="run bounded SQL-only jobs from a frozen version 1 JSON manifest")
    args = parser.parse_args()
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("--batch-sizes must not contain duplicates")
    if len(set(args.backends)) != len(args.backends):
        parser.error("--backends must not contain duplicates")
    if len(set(args.readback_modes)) != len(args.readback_modes):
        parser.error("--readback-modes must not contain duplicates")
    if any(mode != "scalar" for mode in args.readback_modes) and not args.batch_study:
        parser.error("nondefault --readback-modes require --batch-study")
    if args.study_rows is not None and (not args.batch_study or len(set(args.study_rows)) != len(args.study_rows)):
        parser.error("--study-rows requires --batch-study and unique row counts")
    manifest_bytes = None
    manifest = None
    if args.manifest is not None:
        if args.quick or args.batch_study or args.study_rows is not None:
            parser.error("--manifest is incompatible with --quick, --batch-study, and --study-rows")
        if args.batch_sizes != [8192]:
            parser.error("--manifest is incompatible with nondefault --batch-sizes")
        if args.readback_modes != ["scalar"]:
            parser.error("--manifest is incompatible with nondefault --readback-modes")
        try:
            manifest_bytes, manifest = load_manifest(args.manifest)
        except (OSError, ValueError) as error:
            parser.error(str(error))
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
                                       "examples/support/sql_acceptance_limits.rs",
                                       "scripts/sql_acceptance.py", "Cargo.toml", "Cargo.lock")},
        "library_sha256": {str(name): hashlib.sha256(name.read_bytes()).hexdigest()
                           for name in sorted(Path("src").rglob("*.rs"))},
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "binary": str(binary), "profile": "release (build command recorded in report)",
        "environment": {k: v for k, v in os.environ.items()
                        if k in ("RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS", "CUBECL_LOG_LEVEL")},
        "warm_samples": args.warm, "repeats": args.repeats,
        "batch_sizes": args.batch_sizes, "batch_study": args.batch_study,
        "readback_modes": args.readback_modes,
        "study_rows": args.study_rows,
        "manifest_description": manifest["description"] if manifest else None,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes is not None else None,
        "note": "sequential processes; OS/driver caches not cleared; no concurrent queries",
    }
    (args.out / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if manifest_bytes is not None:
        (args.out / "workload-manifest.json").write_bytes(manifest_bytes)
    all_rows = []
    summaries = []
    jobs = []
    if manifest is not None:
        jobs = manifest_jobs(manifest, args.backends, args.repeats)
        for job in jobs:
            job["manifest"] = True
    else:
        selected_cases = batch_study_cases() if args.batch_study else cases(args.quick)
        if args.study_rows is not None:
            selected_cases = [("aggregate", rows, 50, 20, 1, 1) for rows in args.study_rows]
        for case in selected_cases:
            dsl_case = case[0] != "mixed" and (args.quick or case[2:] == (50, 20, 1, 1))
            for repeat in range(1, args.repeats + 1):
                offset = (repeat - 1) % len(args.backends)
                backend_order = args.backends[offset:] + args.backends[:offset]
                if args.batch_study:
                    for batch_size in args.batch_sizes:
                        for backend in backend_order:
                            if backend == "datafusion":
                                jobs.append({"backend": backend, "path": "sql", "case": case,
                                             "batch_size": batch_size, "repeat": repeat,
                                             "readback_mode": None, "case_id": ""})
                            else:
                                mode_offset = (repeat - 1) % len(args.readback_modes)
                                mode_order = args.readback_modes[mode_offset:] + args.readback_modes[:mode_offset]
                                for mode in mode_order:
                                    jobs.append({"backend": backend, "path": "sql", "case": case,
                                                 "batch_size": batch_size, "repeat": repeat,
                                                 "readback_mode": mode, "case_id": ""})
                    continue
                for batch_size in args.batch_sizes:
                    for backend in backend_order:
                        jobs.append({"backend": backend, "path": "sql", "case": case,
                                     "batch_size": batch_size, "repeat": repeat,
                                     "readback_mode": None, "case_id": ""})
                if dsl_case:
                    for backend in backend_order:
                        if backend != "datafusion":
                            jobs.append({"backend": backend, "path": "dsl", "case": case,
                                         "batch_size": None, "repeat": repeat,
                                         "readback_mode": None, "case_id": ""})

    for i, job in enumerate(jobs):
        backend, path, case = job["backend"], job["path"], job["case"]
        batch_size, repeat, readback_mode = job["batch_size"], job["repeat"], job["readback_mode"]
        command = build_command(binary, job, args.warm)
        parts = [backend, path, *map(str, case), f"repeat{repeat}"]
        if job["case_id"]:
            parts.insert(0, job["case_id"])
        if batch_size is not None:
            parts.append(f"batch{batch_size}")
        if readback_mode is not None:
            parts.append(readback_mode)
        name = "-".join(map(str, parts))
        print(f"[{i+1}/{len(jobs)}] {name}", flush=True)
        completed = subprocess.run(command, capture_output=True, text=True)
        (args.out / f"{name}.log").write_text(
            "command=" + json.dumps(command) + "\n" + completed.stderr)
        (args.out / f"{name}.csv").write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError(f"case failed ({completed.returncode}): {name}; see log")
        rows = list(csv.DictReader(completed.stdout.splitlines()))
        job["name"] = name
        validate_output(rows, job, args.warm)
        combined_rows = [dict(row, repeat=repeat, case_id=job["case_id"]) for row in rows]
        all_rows.extend(combined_rows)
        first, warm = rows[0], rows[1:]
        summary = {k: first[k] for k in ("backend", "path", "workload", "rows", "pass_percent",
                                        "null_percent", "batches", "partitions", "session_batch_size",
                                        "aggregate_readback_mode")}
        summary["repeat"] = repeat
        summary["case_id"] = job["case_id"]
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
                    "readback_buffers",
                    "peak_device_bytes", "peak_host_bytes", "max_request_device_bytes"):
            summary["warm_last_" + key] = warm[-1][key]
        for key in ("shared_host_limit_bytes", "shared_device_limit_bytes", "peak_requests"):
            if key in first:
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
