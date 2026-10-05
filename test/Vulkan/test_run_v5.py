from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("run_v5.py")
SPEC = spec_from_file_location("run_v5", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
run_v5 = module_from_spec(SPEC)
SPEC.loader.exec_module(run_v5)


class CompareOutputsTest(unittest.TestCase):
    def test_rejects_non_finite_observed_values(self):
        with self.assertRaisesRegex(RuntimeError, "contains non-finite"):
            run_v5.compare_outputs(
                {"copy_exact": False}, [[1.0]], [[float("nan")]], "Vulkan"
            )

    def test_rejects_non_finite_reference_values(self):
        with self.assertRaisesRegex(RuntimeError, "reference output.*non-finite"):
            run_v5.compare_outputs(
                {"copy_exact": False}, [[float("inf")]], [[1.0]], "Vulkan"
            )


if __name__ == "__main__":
    unittest.main()
