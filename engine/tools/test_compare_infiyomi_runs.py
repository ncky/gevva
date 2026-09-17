#!/usr/bin/env python3
"""CPU-only checks for the local benchmark comparison's paired statistics."""
from pathlib import Path
import unittest
from unittest.mock import patch

import compare_infiyomi_runs as compare


class ComparisonTest(unittest.TestCase):
    def compare_scores(self, baseline, candidate, candidate_config=None):
        data = {
            Path("baseline/quality/page_scores.json"): baseline,
            Path("candidate/quality/page_scores.json"): candidate,
            Path("baseline/quality/judge_model_config.json"): {"model": "fixed"},
            Path("candidate/quality/judge_model_config.json"):
                candidate_config or {"model": "fixed"},
        }
        with patch.object(Path, "exists", return_value=True), \
                patch.object(compare, "read", side_effect=data.__getitem__):
            return compare.paired_quality(Path("baseline"), Path("candidate"))

    def test_constant_paired_difference(self):
        baseline = [{"page_id": str(i), "scores": [{"id": "a", "average": 7}]}
                    for i in range(3)]
        candidate = [{"page_id": str(i), "scores": [{"id": "a", "average": 8}]}
                     for i in range(3)]
        result = self.compare_scores(baseline, candidate)
        self.assertEqual(result["candidate_minus_baseline_mean"], 1)
        self.assertEqual(result["page_cluster_bootstrap_95_percent_interval"], [1, 1])
        self.assertEqual(result["matched_scored_regions"], 3)
        self.assertTrue(result["identical_judge_config_files"])

    def test_missing_scores_are_not_silently_paired(self):
        baseline = [{"page_id": "p", "scores": [
            {"id": "a", "average": 7}, {"id": "b", "average": 3}]}]
        candidate = [{"page_id": "p", "scores": [{"id": "a", "average": 7}]}]
        result = self.compare_scores(baseline, candidate, {"model": "different"})
        self.assertEqual(result["matched_scored_regions"], 1)
        self.assertEqual(result["baseline_scored_regions"], 2)
        self.assertEqual(result["candidate_scored_regions"], 1)
        self.assertEqual(result["candidate_minus_baseline_mean"], 0)
        self.assertFalse(result["identical_judge_config_files"])

    def test_no_common_scores(self):
        result = self.compare_scores(
            [{"page_id": "p", "scores": [{"id": "a", "average": 7}]}],
            [{"page_id": "p", "scores": [{"id": "b", "average": 7}]}])
        self.assertFalse(result["available"])


if __name__ == "__main__":
    unittest.main()
