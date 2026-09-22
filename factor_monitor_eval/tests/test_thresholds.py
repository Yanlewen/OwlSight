import json
import tempfile
import unittest
from pathlib import Path

from factor_monitor_eval.thresholds import resolve_threshold


class ThresholdTests(unittest.TestCase):
    def test_resolves_declared_file_and_never_searches_eval_rows(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            root = Path(root)
            source = root / "validation_threshold.json"
            source.write_text(json.dumps({"threshold": 0.37}))
            mapping = root / "thresholds.json"
            mapping.write_text(json.dumps({"mask_qwen": 0.41}))
            entry = {"id": "mask_qwen"}
            result = resolve_threshold(entry, manifest_path=root / "manifest.yaml", thresholds_file=mapping)
            self.assertEqual(result["value"], 0.41)
            result = resolve_threshold({"id": "mask_qwen", "threshold_source": str(source)}, manifest_path=root / "manifest.yaml")
            self.assertEqual(result["value"], 0.37)

    def test_missing_threshold_is_explicit(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as root:
            result = resolve_threshold({"id": "mask_qwen"}, manifest_path=Path(root) / "manifest.yaml")
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
