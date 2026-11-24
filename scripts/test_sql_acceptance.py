import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import sql_acceptance as runner


def valid_case(**changes):
    case = {
        "id": "small-case", "workload": "aggregate", "rows": 64,
        "pass_percent": 50, "null_percent": 20, "batches": 2,
        "partitions": 1, "batch_sizes": [8192, 16384],
        "readback_modes": ["scalar", "batched"],
        "shared_host_bytes": 1048576, "shared_device_bytes": 2097152,
    }
    case.update(changes)
    return case


def write_manifest(directory, value):
    path = Path(directory) / "manifest.json"
    path.write_text(json.dumps(value))
    return path


class ManifestTests(unittest.TestCase):
    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"version":1,"version":1,"description":"x","cases":[]}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                runner.load_manifest(path)

    def test_rejects_booleans_ranges_unknown_fields_and_duplicate_ids(self):
        invalid_cases = [
            valid_case(rows=True),
            valid_case(rows=2147483648),
            valid_case(shared_host_bytes=0),
            valid_case(unexpected=1),
        ]
        for case in invalid_cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as directory:
                    path = write_manifest(directory, {"version": 1, "description": "x", "cases": [case]})
                    with self.assertRaises(ValueError):
                        runner.load_manifest(path)
        with tempfile.TemporaryDirectory() as directory:
            path = write_manifest(directory, {"version": 1, "description": "x",
                                              "cases": [valid_case(), valid_case()]})
            with self.assertRaisesRegex(ValueError, "duplicate case id"):
                runner.load_manifest(path)

    def test_rejects_invalid_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_manifest(directory, {"version": 1, "description": "x",
                                              "cases": [valid_case(id="Bad ID")]})
            with self.assertRaisesRegex(ValueError, "id is invalid"):
                runner.load_manifest(path)

    def test_manifest_jobs_baseline_once_rotated_modes_and_budget_flags(self):
        manifest = {"version": 1, "description": "x", "cases": [valid_case()]}
        jobs = runner.manifest_jobs(manifest, ["datafusion", "cpu", "metal"], 2)
        self.assertEqual(len(jobs), 20)
        repeat1 = [job for job in jobs if job["repeat"] == 1 and job["batch_size"] == 8192]
        repeat2 = [job for job in jobs if job["repeat"] == 2 and job["batch_size"] == 8192]
        self.assertEqual([job["backend"] for job in repeat1],
                         ["datafusion", "cpu", "cpu", "metal", "metal"])
        self.assertEqual([job["backend"] for job in repeat2],
                         ["cpu", "cpu", "metal", "metal", "datafusion"])
        self.assertEqual([job["readback_mode"] for job in repeat1[1:]],
                         ["scalar", "batched", "scalar", "batched"])
        df = repeat1[0]
        command = runner.build_command("sql_acceptance", {**df, "manifest": True}, 5)
        self.assertEqual(command[1], "datafusion")
        self.assertNotIn("scalar", command)
        self.assertEqual(command[-4:], ["--shared-host-bytes", "1048576",
                                       "--shared-device-bytes", "2097152"])
        self.assertFalse(any(job["path"] == "dsl" for job in jobs))

    def test_manifest_profiling_cross_product_rotates_and_keeps_one_baseline(self):
        manifest = {"version": 1, "description": "x", "cases": [valid_case()]}
        jobs = runner.manifest_jobs(manifest, ["datafusion", "cpu"], 2,
                                    ["off", "host", "device"])
        self.assertEqual(len(jobs), 2 * 2 * (1 + 2 * 3))
        for repeat in (1, 2):
            group = [job for job in jobs if job["repeat"] == repeat and job["batch_size"] == 8192]
            self.assertEqual(sum(job["backend"] == "datafusion" for job in group), 1)
            baseline = next(job for job in group if job["backend"] == "datafusion")
            self.assertIsNone(baseline["profiling_mode"])
            device_jobs = [job for job in group if job["backend"] == "cpu"]
            self.assertEqual(len(device_jobs), 6)
            self.assertEqual([job["profiling_mode"] for job in device_jobs[:3]],
                             ["off", "host", "device"] if repeat == 1
                             else ["host", "device", "off"])
        cpu = next(job for job in jobs if job["backend"] == "cpu")
        command = runner.build_command("sql_acceptance", cpu, 3)
        self.assertEqual(command[-2:], ["--profiling", "off"])
        baseline_command = runner.build_command("sql_acceptance", jobs[0], 3)
        self.assertNotIn("--profiling", baseline_command)

    def test_legacy_fixture_counts(self):
        self.assertEqual(len(runner.cases()), 26)
        self.assertEqual(len(runner.cases(quick=True)), 4)
        self.assertEqual(len(runner.batch_study_cases()), 3)


class OutputValidationTests(unittest.TestCase):
    def _job(self, backend="cpu", mode="scalar"):
        return {"backend": backend, "path": "sql",
                "case": ("aggregate", 64, 50, 20, 2, 1), "batch_size": 8192,
                "readback_mode": mode, "case_id": "small-case", "name": "job"}

    def _rows(self, backend="cpu", mode="scalar"):
        header = {"backend": backend, "path": "sql", "workload": "aggregate",
                  "rows": "64", "pass_percent": "50", "null_percent": "20",
                  "batches": "2", "partitions": "1", "session_batch_size": "8192",
                  "aggregate_readback_mode": "" if backend == "datafusion" else mode,
                  "profiling_mode": "" if backend == "datafusion" else "off",
                  "init_ms": "1.0", "prepare_ms": "2.0", "execute_ms": "3.0",
                  "total_ms": "5.0", "case_id": "small-case"}
        header.update({field: "" for field in runner.PROFILE_COLUMNS})
        return [dict(header, phase="first", sample="0"),
                dict(header, phase="warm", sample="1", init_ms="0")]

    def test_valid_and_mismatched_csv_outputs(self):
        rows = self._rows()
        runner.validate_output(rows, self._job(), 1)
        for changed in (dict(rows[0], backend="metal"), dict(rows[1], sample="7"),
                        dict(rows[1], phase="first"), dict(rows[1], total_ms="nan"),
                        dict(rows[1], case_id="wrong"), dict(rows[1], init_ms="0.01")):
            with self.subTest(changed=changed):
                bad = [rows[0], changed]
                with self.assertRaises(RuntimeError):
                    runner.validate_output(bad, self._job(), 1)

    def test_datafusion_mode_must_be_blank(self):
        rows = self._rows(backend="datafusion", mode=None)
        rows[0]["aggregate_readback_mode"] = "scalar"
        with self.assertRaisesRegex(RuntimeError, "DataFusion readback mode"):
            runner.validate_output(rows, self._job(backend="datafusion", mode=None), 1)

    def test_profiling_mode_must_match_and_cpu_baseline_is_off(self):
        rows = self._rows()
        runner.validate_output(rows, self._job(), 1)
        rows[1]["profiling_mode"] = "host"
        with self.assertRaisesRegex(RuntimeError, "profiling mode"):
            runner.validate_output(rows, self._job(), 1)
        rows = self._rows(backend="datafusion", mode=None)
        runner.validate_output(rows, self._job(backend="datafusion", mode=None), 1)
        rows[0]["profiling_mode"] = "off"
        with self.assertRaisesRegex(RuntimeError, "profiling mode"):
            runner.validate_output(rows, self._job(backend="datafusion", mode=None), 1)

    def test_off_datafusion_and_dsl_profile_metrics_must_be_blank(self):
        for backend, path in (("cpu", "sql"), ("datafusion", "sql"), ("cpu", "dsl")):
            job = {**self._job(backend=backend), "path": path,
                   "batch_size": None if path == "dsl" else 8192,
                   "profiling_mode": None}
            rows = self._rows(backend=backend)
            for row in rows:
                row["path"] = path
                row["session_batch_size"] = "" if path == "dsl" else "8192"
                row["profiling_mode"] = "" if backend == "datafusion" or path == "dsl" else "off"
            runner.validate_output(rows, job, 1)
            rows[1]["profile_value_encode_ns"] = "0"
            with self.assertRaisesRegex(RuntimeError, "must be blank"):
                runner.validate_output(rows, job, 1)

    def test_host_metrics_must_be_complete_nonnegative_integers_without_gpu_time(self):
        job = {**self._job(), "profiling_mode": "host"}
        rows = self._rows()
        for row in rows:
            row["profiling_mode"] = "host"
            for field in runner.PROFILE_COLUMNS:
                row[field] = "" if field == "profile_device_window_ns" else "0"
        runner.validate_output(rows, job, 1)
        rows[1]["profile_arrow_export_ns"] = ""
        with self.assertRaisesRegex(RuntimeError, "invalid profile_arrow_export_ns"):
            runner.validate_output(rows, job, 1)
        rows[1]["profile_arrow_export_ns"] = "0"
        rows[1]["profile_device_window_ns"] = "1"
        with self.assertRaisesRegex(RuntimeError, "GPU time must be blank"):
            runner.validate_output(rows, job, 1)
        rows[1]["profile_device_window_ns"] = ""
        rows[1]["profile_device_unavailable"] = "1"
        with self.assertRaisesRegex(RuntimeError, "device counters must be zero"):
            runner.validate_output(rows, job, 1)

    def test_device_gpu_time_must_match_valid_windows(self):
        job = {**self._job(), "profiling_mode": "device"}
        rows = self._rows()
        for row in rows:
            row["profiling_mode"] = "device"
            for field in runner.PROFILE_COLUMNS:
                row[field] = "0"
            row["profile_device_windows"] = "1"
            row["profile_device_window_ns"] = "5"
        runner.validate_output(rows, job, 1)
        rows[1]["profile_device_window_ns"] = "0"
        with self.assertRaisesRegex(RuntimeError, "GPU time must be positive"):
            runner.validate_output(rows, job, 1)
        rows[1]["profile_device_window_ns"] = ""
        rows[1]["profile_device_failed"] = "1"
        runner.validate_output(rows, job, 1)
        rows[1]["profile_device_window_ns"] = "5"
        with self.assertRaisesRegex(RuntimeError, "GPU time must be blank"):
            runner.validate_output(rows, job, 1)

    def test_manifest_budget_caps_must_match(self):
        job = {**self._job(), "manifest": True, "shared_host_bytes": 100,
               "shared_device_bytes": 200}
        rows = self._rows()
        for row in rows:
            row["shared_host_limit_bytes"] = "100"
            row["shared_device_limit_bytes"] = "200"
        runner.validate_output(rows, job, 1)
        rows[1]["shared_device_limit_bytes"] = "201"
        with self.assertRaisesRegex(RuntimeError, "shared_device_limit_bytes"):
            runner.validate_output(rows, job, 1)


if __name__ == "__main__":
    unittest.main()
