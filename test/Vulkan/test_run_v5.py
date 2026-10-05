from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tempfile
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


class V52IndependentGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="v5-independent-golden-")
        cls.fixtures = run_v5.fixture_definitions(
            Path(cls.temporary.name), "v5.2"
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_v52_suite_contains_ranked_softmax_pool_and_concat_cases(self):
        self.assertEqual(
            set(self.fixtures),
            {
                "max_pool_tail",
                "global_avg_pool",
                "conv1x1_pad1",
                "softmax_rank1",
                "softmax_rank3_axis2",
                "concat_height_axis",
                "concat_width_axis",
            },
        )
        self.assertEqual(
            self.fixtures["softmax_rank1"]["inputs"][0][1], (1000, 1, 1)
        )
        self.assertEqual(self.fixtures["softmax_rank1"]["outputs"][0][1], (1000,))

    def test_max_pool_golden_ignores_negative_tail_padding(self):
        fixture = self.fixtures["max_pool_tail"]
        source = [-1.0] * run_v5.count(fixture["inputs"][0][1])
        result = run_v5.independent_golden(fixture, [source])[0]
        self.assertEqual(len(result), 28 * 28)
        self.assertEqual(result, [-1.0] * (28 * 28))

    def test_global_average_golden_preserves_each_channel(self):
        fixture = self.fixtures["global_avg_pool"]
        source = list(range(196)) + list(range(196, 392))
        result = run_v5.independent_golden(fixture, [source])[0]
        self.assertEqual(result, [97.5, 293.5])

    def test_conv1x1_padding_golden_materializes_zero_border(self):
        fixture = self.fixtures["conv1x1_pad1"]
        source = [float(value) for value in range(1, 19)]
        result = run_v5.independent_golden(fixture, [source])[0]
        self.assertEqual(len(result), 25)
        self.assertEqual(result[0], 0.125)
        self.assertEqual(result[6], 0.5 - 0.25 * 10 + 0.125)

    def test_softmax_golden_handles_rank_one_and_exact_axis_broadcast(self):
        rank1 = self.fixtures["softmax_rank1"]
        result = run_v5.independent_golden(rank1, [[-2.0, 0.0, 2.0] + [0.0] * 997])[0]
        self.assertEqual(len(result), 1000)
        self.assertAlmostEqual(sum(result), 1.0, places=14)

        rank3 = self.fixtures["softmax_rank3_axis2"]
        values = [float(index - 20) / 8.0 for index in range(7 * 11)]
        result = run_v5.independent_golden(rank3, [values])[0]
        for channel in range(7):
            self.assertAlmostEqual(
                sum(result[channel * 11:(channel + 1) * 11]), 1.0, places=14
            )

    def test_softmax_golden_handles_extreme_finite_logits(self):
        fixture = self.fixtures["softmax_rank1"]
        values = [-1000.0, 1000.0, 0.0, 999.0, -999.0] * 200
        result = run_v5.independent_golden(fixture, [values])[0]
        self.assertTrue(all(value >= 0.0 for value in result))
        self.assertAlmostEqual(sum(result), 1.0, places=12)
        self.assertTrue(all(run_v5.math.isfinite(value) for value in result))

    def test_concat_goldens_cover_spatial_axes(self):
        for name, axis in (("concat_height_axis", 1), ("concat_width_axis", 2)):
            fixture = self.fixtures[name]
            inputs = [
                [float(index) for index in range(run_v5.count(shape))]
                for _, shape in fixture["inputs"]
            ]
            result = run_v5.independent_golden(fixture, inputs)[0]
            self.assertEqual(len(result), run_v5.count(fixture["outputs"][0][1]))
            self.assertEqual(fixture["oracle"]["axis"], axis)


if __name__ == "__main__":
    unittest.main()
