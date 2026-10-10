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


class RuntimeLibraryPathTest(unittest.TestCase):
    def test_static_llvm_does_not_prepend_helper_library_directory(self):
        path = run_v5.configured_library_path(
            False,
            "/opt/static-llvm/lib",
            "/opt/sanitizer/lib",
            "/opt/application/lib",
        )
        self.assertEqual(path, "/opt/sanitizer/lib:/opt/application/lib")

    def test_shared_llvm_retains_its_runtime_library_directory(self):
        path = run_v5.configured_library_path(
            True,
            "/opt/shared-llvm/lib",
            "/opt/sanitizer/lib",
            "/opt/application/lib",
        )
        self.assertEqual(
            path,
            "/opt/shared-llvm/lib:/opt/sanitizer/lib:/opt/application/lib",
        )


class V53ModelDefinitionTest(unittest.TestCase):
    def test_uses_vendored_squeezenet_with_verified_topology(self):
        repository = MODULE_PATH.resolve().parents[2]
        model_directory = (
            repository / "test/third_party/ncnn/examples"
        )
        param = model_directory / "squeezenet_v1.1.param"
        binary = model_directory / "squeezenet_v1.1.bin"
        with tempfile.TemporaryDirectory(prefix="v53-model-definition-") as temp:
            fixtures = run_v5.fixture_definitions(
                Path(temp), "v5.3", param, binary
            )
        fixture = fixtures["squeezenet_v1_1"]
        self.assertEqual(fixture["inputs"], [("data", (3, 227, 227))])
        self.assertEqual(fixture["outputs"], [("prob", (1000,))])
        self.assertEqual(
            fixture["layer_counts"],
            {
                "Concat": 8,
                "Convolution": 26,
                "Dropout": 1,
                "Input": 1,
                "Pooling": 4,
                "ReLU": 26,
                "Softmax": 1,
                "Split": 8,
            },
        )
        self.assertEqual(fixture["expected_conv_kernels"], 26)
        self.assertEqual(fixture["expected_weight_resources"], 52)
        self.assertEqual(fixture["classification_top_k"], 5)
        self.assertEqual(fixture["seed_base"], 0x53515545)
        self.assertAlmostEqual(fixture["atol"], 1.0e-6)
        self.assertAlmostEqual(fixture["rtol"], 1.0e-4)


class WeightResourceDeduplicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="v51-weight-dedup-")
        cls.fixtures = run_v5.fixture_definitions(
            Path(cls.temporary.name), "v5.1"
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_split_concat_fixture_shares_identical_bias_resource(self):
        oracles = self.fixtures["split_concat"]["weight_oracles"]
        self.assertEqual(oracles[0]["bias_sha256"], oracles[1]["bias_sha256"])
        expected = {}
        roles = {}
        for oracle in oracles:
            run_v5.record_expected_weight(
                expected,
                roles,
                9,
                "bias",
                oracle["bias_sha256"],
                oracle["bias_size_bytes"],
            )
        self.assertEqual(len(expected), 1)
        self.assertEqual(roles, {9: "bias"})

    def test_rejects_conflicting_identity_or_role_for_shared_resource(self):
        expected = {}
        roles = {}
        run_v5.record_expected_weight(expected, roles, 9, "bias", "a" * 64, 4)
        with self.assertRaisesRegex(RuntimeError, "conflicting identities"):
            run_v5.record_expected_weight(expected, roles, 9, "bias", "b" * 64, 4)
        with self.assertRaisesRegex(RuntimeError, "incompatible roles"):
            run_v5.record_expected_weight(expected, roles, 9, "filter", "a" * 64, 4)


class ClassificationSummaryTest(unittest.TestCase):
    def test_top_k_is_stable_for_equal_probabilities(self):
        result = run_v5.classification_summary([0.25, 0.4, 0.4, 0.1], 3)
        self.assertEqual(result["top1_index"], 1)
        self.assertEqual(result["topk_indices"], [1, 2, 0])
        self.assertAlmostEqual(result["probability_sum"], 1.15)

    def test_rejects_non_finite_classification_probabilities(self):
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            run_v5.classification_summary([0.5, float("nan")], 1)


class VulkanDeviceSummaryTest(unittest.TestCase):
    def test_nvidia_record_is_gpu_not_optimus_layer(self):
        text = """\
VK_LAYER_NV_optimus NVIDIA Optimus layer 1.4.351
GPU0:
    apiVersion = 1.4.351
    driverVersion = 615.71.9.0
    vendorID = 0x10de
    deviceID = 0x2803
    deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
    deviceName = NVIDIA GeForce RTX 4060 Ti
    driverID = DRIVER_ID_NVIDIA_PROPRIETARY
    driverName = NVIDIA
    driverInfo = 615.71.09
"""
        device = run_v5.parse_vulkan_device_summary(text, "NVIDIA", "/icd/nvidia.json")
        self.assertEqual(device["device_name"], "NVIDIA GeForce RTX 4060 Ti")
        self.assertEqual(device["driver_name"], "NVIDIA")
        self.assertEqual(device["driver_info"], "615.71.09")
        self.assertEqual(device["vendor_id"], "0x10de")
        self.assertEqual(device["device_type"], "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU")

    def test_llvmpipe_record_captures_driver_identity(self):
        text = """\
GPU0:
    apiVersion = 1.4.305
    driverVersion = 0.0.1
    vendorID = 0x10005
    deviceID = 0x0000
    deviceType = PHYSICAL_DEVICE_TYPE_CPU
    deviceName = llvmpipe (LLVM 19.1.7, 256 bits)
    driverID = DRIVER_ID_MESA_LLVMPIPE
    driverName = llvmpipe
    driverInfo = Mesa 25.0.7 (LLVM 19.1.7)
"""
        device = run_v5.parse_vulkan_device_summary(text, "llvmpipe", "/icd/lvp.json")
        self.assertEqual(device["device_name"], "llvmpipe (LLVM 19.1.7, 256 bits)")
        self.assertEqual(device["driver_name"], "llvmpipe")
        self.assertEqual(device["vendor_id"], "0x10005")
        self.assertEqual(device["device_type"], "PHYSICAL_DEVICE_TYPE_CPU")
        self.assertEqual(len(device["vulkaninfo_summary_sha256"]), 64)

    def test_missing_device_type_is_not_hardware_evidence(self):
        text = """\
GPU0:
    apiVersion = 1.4.351
    driverVersion = 615.71.9.0
    vendorID = 0x10de
    deviceID = 0x2803
    deviceName = NVIDIA GeForce RTX 4060 Ti
"""
        with self.assertRaisesRegex(RuntimeError, "deviceType"):
            run_v5.parse_vulkan_device_summary(text, "NVIDIA", "/icd/nvidia.json")

    def test_layer_match_cannot_substitute_for_a_physical_device(self):
        text = "VK_LAYER_NV_optimus NVIDIA Optimus layer\nGPU0:\n    deviceName = llvmpipe\n"
        with self.assertRaisesRegex(RuntimeError, "enumerated devices"):
            run_v5.parse_vulkan_device_summary(text, "NVIDIA", "/icd/nvidia.json")


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
