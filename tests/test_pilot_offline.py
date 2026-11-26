#!/usr/bin/env python3
"""
Unit test for Phase 3 Part B offline pilot execution.
Tests pilot extraction, synthetic forward pass, log-prob shift, and McNemar test.
Run: python tests/test_pilot_offline.py
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(ROOT, "..", "partB_benchmark")
MANIFEST_DIR = os.path.join(BENCH_DIR, "manifests")


class TestPilotOffline(unittest.TestCase):
    def test_pilot_manifest_files(self):
        for name in ("pilot_real.json", "pilot_shuffled.json", "pilot_swapped.json"):
            path = os.path.join(MANIFEST_DIR, name)
            self.assertTrue(os.path.exists(path), f"Missing {path}")
            with open(path) as f:
                items = json.load(f)
            self.assertEqual(len(items), 150, f"Expected 150 items in {name}, got {len(items)}")

    def test_kaggle_pilot_smoke_run(self):
        with tempfile.TemporaryDirectory() as tmp_out:
            cmd = [
                sys.executable,
                os.path.join(BENCH_DIR, "kaggle_m3d_pilot.py"),
                "--smoke",
                "--manifest_dir", MANIFEST_DIR,
                "--output_dir", tmp_out
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Pilot smoke test failed with stderr: {res.stderr}")

            # Check outputs
            csv_path = os.path.join(tmp_out, "m3d_pilot_per_item.csv")
            sum_path = os.path.join(tmp_out, "m3d_pilot_summary.json")

            self.assertTrue(os.path.exists(csv_path), "Missing m3d_pilot_per_item.csv")
            self.assertTrue(os.path.exists(sum_path), "Missing m3d_pilot_summary.json")

            with open(sum_path) as f:
                summary = json.load(f)
            self.assertGreaterEqual(len(summary), 5, "Expected summary breakdown for overall and families")
            overall = next(item for item in summary if item["group"] == "overall")
            self.assertEqual(overall["n"], 150)


if __name__ == "__main__":
    unittest.main()
