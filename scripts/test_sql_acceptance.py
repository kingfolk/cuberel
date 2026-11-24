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
                  "init_ms": "1.0", "prepare_ms": "2.0", "execute_ms": "3.0",
                  "total_ms": "5.0", "case_id": "small-case"}
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
