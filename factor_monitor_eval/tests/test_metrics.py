import unittest

from factor_monitor_eval.metrics import auroc, average_precision, score_decisions, score_rows


class MetricsTests(unittest.TestCase):
    def test_gmean2_and_invalid(self):
        result = score_decisions([
            {"label_id": 1, "decision": "applied"},
            {"label_id": 1, "decision": "not_applied"},
            {"label_id": 0, "decision": "not_applied"},
            {"label_id": 0, "decision": "garbage"},
        ])
        self.assertEqual((result["tp"], result["fn"], result["tn"], result["fp"]), (1, 1, 1, 1))
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["gmean2"], 0.25)

    def test_score_metrics_are_auxiliary_to_frozen_threshold(self):
        result = score_rows([
            {"label_id": 1, "score": 0.9},
            {"label_id": 0, "score": 0.1},
        ], 0.5)
        self.assertEqual(result["gmean2"], 1.0)
        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["auprc"], 1.0)
        self.assertEqual(auroc([{"label_id": 1, "score": 0.5}, {"label_id": 0, "score": 0.5}]), 0.5)
        self.assertEqual(average_precision([{"label_id": 1, "score": 0.5}, {"label_id": 0, "score": 0.5}]), 0.5)


if __name__ == "__main__":
    unittest.main()
