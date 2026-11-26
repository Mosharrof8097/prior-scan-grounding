#!/usr/bin/env python3
"""
Unit test for Part B TotalSegmentator benchmark generation, answer balance, and counterfactual probes.
Run: python tests/test_partb_benchmark.py
"""

import json
import os
import sys
import unittest
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(ROOT, "..", "partB_benchmark")
sys.path.insert(0, BENCH_DIR)

import class_metadata
import build_benchmark


class TestPartBBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate manifests
        build_benchmark.generate_benchmark_suites()
        cls.manifest_dir = os.path.join(BENCH_DIR, "manifests")
        with open(os.path.join(cls.manifest_dir, "benchmark_real.json")) as f:
            cls.real_q = json.load(f)
        with open(os.path.join(cls.manifest_dir, "benchmark_shuffled.json")) as f:
            cls.shuffled_q = json.load(f)
        with open(os.path.join(cls.manifest_dir, "benchmark_swapped.json")) as f:
            cls.swapped_q = json.load(f)
        with open(os.path.join(cls.manifest_dir, "benchmark_summary.json")) as f:
            cls.summary = json.load(f)

    def test_class_count(self):
        self.assertEqual(len(class_metadata.CLASSES), 117, "TotalSegmentator v2 must have 117 classes")

    def test_question_count(self):
        self.assertGreaterEqual(len(self.real_q), 5000, "Should generate thousands of benchmark questions")
        self.assertEqual(len(self.real_q), len(self.shuffled_q))
        self.assertEqual(len(self.real_q), len(self.swapped_q))

    def test_per_family_answer_balance(self):
        df = pd.DataFrame(self.real_q)
        for fam, g in df.groupby("family"):
            freqs = g["answer_choice"].value_counts(normalize=True).to_dict()
            if fam == "existence":
                # Binary family: A and B should both be between 0.45 and 0.55
                for choice in ("A", "B"):
                    self.assertAlmostEqual(freqs[choice], 0.50, delta=0.05,
                                           msg=f"Binary existence {choice} frequency {freqs[choice]:.2f} not balanced!")
            else:
                # 4-choice MCQ family: each of A, B, C, D should be roughly 0.25 (between 0.18 and 0.32)
                for choice in ("A", "B", "C", "D"):
                    self.assertAlmostEqual(freqs.get(choice, 0), 0.25, delta=0.07,
                                           msg=f"Family {fam} choice {choice} frequency {freqs.get(choice, 0):.2f} not balanced!")

    def test_counterfactual_swapped_different_donor(self):
        for q in self.swapped_q:
            self.assertNotEqual(q["scan_id"], q["donor_scan_id"], "Swapped scan must come from a different donor patient")

    def test_shuffled_options_contain_gold(self):
        for orig, shuff in zip(self.real_q[:100], self.shuffled_q[:100]):
            self.assertEqual(orig["question_id"], shuff["question_id"])
            gold_text = orig["answer_text"]
            shuff_gold_letter = shuff["answer_choice"]
            self.assertEqual(shuff["choices"][shuff_gold_letter], gold_text, "Shuffled choice map must correctly point to the gold answer text")


if __name__ == "__main__":
    unittest.main()
