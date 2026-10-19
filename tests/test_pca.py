import io
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.subspacead.config import get_args
from src.subspacead.core.pca import PCAModel
from src.subspacead.utils.common import generate_run_name


class StreamingPCATests(unittest.TestCase):
    def setUp(self):
        self.progress = patch(
            "src.subspacead.core.pca.tqdm", lambda iterable, **kwargs: iterable
        )
        self.progress.start()
        self.addCleanup(self.progress.stop)
        rng = np.random.default_rng(431)
        self.values = rng.normal(size=(137, 6)) * np.array([9, 5, 3, 1, .4, .1])

    def fit(self, values, algorithm="single_pass", sizes=(1, 13, 47, 76), **kwargs):
        batches = np.split(values, np.cumsum(sizes)[:-1])
        calls = []

        def features():
            calls.append(1)
            yield from batches

        model = PCAModel(algorithm=algorithm, **kwargs)
        model.device = torch.device("cpu")
        params = model.fit(features, values.shape[1], len(values), len(batches))
        return params, calls

    def assert_equivalent(self, a, b):
        self.assertEqual(a["k"], b["k"])
        self.assertEqual(set(a), set(b))
        np.testing.assert_allclose(a["mu"], b["mu"], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(a["eigvals"], b["eigvals"], atol=1e-10, rtol=1e-10)
        # Projectors are invariant to eigenvector sign and basis rotation.
        pa = a["components"] @ a["components"].T
        pb = b["components"] @ b["components"].T
        np.testing.assert_allclose(pa, pb, atol=1e-10, rtol=1e-10)
        test = np.random.default_rng(927).normal(size=(19, len(a["mu"])))
        ra = np.sum(((test - a["mu"]) @ (np.eye(len(pa)) - pa)) ** 2, axis=1)
        rb = np.sum(((test - b["mu"]) @ (np.eye(len(pb)) - pb)) ** 2, axis=1)
        np.testing.assert_allclose(ra, rb, atol=1e-9, rtol=1e-9)

    def test_fixed_rank_matches_two_pass(self):
        for k in (1, 3, 6):
            with self.subTest(k=k):
                a, _ = self.fit(self.values, k=k)
                b, _ = self.fit(self.values, algorithm="two_pass", k=k)
                self.assert_equivalent(a, b)

    def test_explained_variance_rank_matches_two_pass(self):
        for ev in (.8, .95, .99):
            with self.subTest(ev=ev):
                a, _ = self.fit(self.values, ev=ev)
                b, _ = self.fit(self.values, algorithm="two_pass", ev=ev)
                self.assert_equivalent(a, b)

    def test_matches_dense_numpy_pca(self):
        a, _ = self.fit(self.values, k=3)
        mu = self.values.mean(axis=0)
        centered = self.values - mu
        vals, vecs = np.linalg.eigh(centered.T @ centered / (len(centered) - 1))
        np.testing.assert_allclose(a["mu"], mu, atol=1e-12)
        np.testing.assert_allclose(a["eigvals"], vals[::-1][:3], rtol=1e-10)
        basis = vecs[:, ::-1][:, :3]
        np.testing.assert_allclose(
            a["components"] @ a["components"].T, basis @ basis.T, atol=1e-10
        )

    def test_single_pass_calls_generator_once(self):
        calls = []

        def features():
            calls.append(1)
            if len(calls) > 1:
                raise AssertionError("Feature stream must not be replayed")
            yield self.values[:37]
            yield self.values[37:]

        model = PCAModel(k=3, algorithm="single_pass")
        model.device = torch.device("cpu")
        model.fit(features, 6, len(self.values), 2)
        self.assertEqual(len(calls), 1)

    def test_default_preserves_two_generator_passes(self):
        calls = []

        def features():
            calls.append(1)
            yield self.values

        model = PCAModel(k=3)
        model.device = torch.device("cpu")
        result = model.fit(features, 6, len(self.values), 1)
        explicit, _ = self.fit(self.values, algorithm="two_pass", sizes=(137,), k=3)
        self.assertEqual(len(calls), 2)
        self.assert_equivalent(result, explicit)

    def test_partition_sizes_do_not_change_subspace(self):
        reference, _ = self.fit(self.values, sizes=(137,), k=3)
        for sizes in ((1,) * 137, (0, 3, 0, 91, 43), (68, 69)):
            with self.subTest(sizes=sizes):
                result, _ = self.fit(self.values, sizes=sizes, k=3)
                self.assert_equivalent(result, reference)

    def test_float32_inputs_match_two_pass(self):
        values = self.values.astype(np.float32)
        a, _ = self.fit(values, k=3)
        b, _ = self.fit(values, algorithm="two_pass", k=3)
        self.assert_equivalent(a, b)

    def test_dino_small_feature_dimension(self):
        values = np.random.default_rng(528).normal(size=(512, 384)).astype(np.float32)
        sizes = (17, 100, 300, 95)
        a, _ = self.fit(values, sizes=sizes, k=24)
        b, _ = self.fit(values, algorithm="two_pass", sizes=sizes, k=24)
        self.assert_equivalent(a, b)

    def test_large_offset_retains_small_covariance(self):
        values = 1e9 + self.values * .01
        result, _ = self.fit(values, k=6)
        centered = values - values.mean(axis=0)
        reference = centered.T @ centered / (len(values) - 1)
        restored = (result["components"] * result["eigvals"]) @ result["components"].T
        np.testing.assert_allclose(restored, reference, atol=1e-9, rtol=1e-4)

    def test_rank_deficient_stream(self):
        values = np.column_stack((self.values[:, :2], self.values[:, 0] * 2))
        a, _ = self.fit(values, k=2)
        b, _ = self.fit(values, algorithm="two_pass", k=2)
        self.assert_equivalent(a, b)

    def test_constant_features_with_fixed_rank(self):
        result, _ = self.fit(np.full((137, 6), 4.), k=2)
        np.testing.assert_array_equal(result["mu"], np.full(6, 4.))
        np.testing.assert_array_equal(result["eigvals"], np.zeros(2))
        self.assertTrue(np.isfinite(result["cov_Z_inv"]).all())

    def test_initial_mean_does_not_require_squaring_absolute_values(self):
        values = np.full((2, 6), 1e200)
        result, _ = self.fit(values, sizes=(1, 1), k=2)
        np.testing.assert_array_equal(result["mu"], np.full(6, 1e200))
        np.testing.assert_array_equal(result["eigvals"], np.zeros(2))

    def test_whitening_metadata_matches_two_pass(self):
        a, _ = self.fit(self.values, k=3, whiten=True)
        b, _ = self.fit(self.values, algorithm="two_pass", k=3, whiten=True)
        self.assertTrue(a["whiten"])
        self.assert_equivalent(a, b)
        np.testing.assert_allclose(a["sqrt_eig"], b["sqrt_eig"], rtol=1e-10)
        np.testing.assert_allclose(a["cov_Z_inv"], b["cov_Z_inv"], rtol=1e-10)

    def test_uses_observed_count_after_token_filtering(self):
        model = PCAModel(k=3, algorithm="single_pass")
        model.device = torch.device("cpu")
        result = model.fit(lambda: iter([self.values]), 6, 10000, 1)
        reference, _ = self.fit(self.values, sizes=(137,), k=3)
        self.assert_equivalent(result, reference)

    def test_input_arrays_are_not_modified(self):
        original = self.values.copy()
        self.fit(self.values, k=3)
        np.testing.assert_array_equal(original, self.values)

    def test_too_few_observed_tokens_are_rejected(self):
        for batches in ([], [np.empty((0, 6))], [np.ones((1, 6))]):
            with self.subTest(lengths=[len(x) for x in batches]):
                model = PCAModel(algorithm="single_pass")
                model.device = torch.device("cpu")
                with self.assertRaisesRegex(ValueError, "at least two"):
                    model.fit(lambda: iter(batches), 6, 100, len(batches))

    def test_invalid_shapes_are_rejected(self):
        for batch in (np.ones(6), np.ones((3, 7))):
            model = PCAModel(algorithm="single_pass")
            model.device = torch.device("cpu")
            with self.subTest(shape=batch.shape):
                with self.assertRaisesRegex(ValueError, "shape"):
                    model.fit(lambda: iter([batch]), 6, len(batch), 1)

    def test_nonfinite_values_are_rejected(self):
        for invalid in (np.nan, np.inf, -np.inf):
            values = self.values.copy()
            values[0, 0] = invalid
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "finite"):
                    self.fit(values)

    def test_invalid_algorithm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "algorithm"):
            PCAModel(algorithm="unknown")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_single_pass_matches_cpu_two_pass(self):
        model = PCAModel(k=3, algorithm="single_pass")
        model.device = torch.device("cuda")
        result = model.fit(lambda: iter([self.values[:37], self.values[37:]]), 6, 137, 2)
        reference, _ = self.fit(self.values, algorithm="two_pass", k=3)
        self.assert_equivalent(result, reference)


class PCAArgumentTests(unittest.TestCase):
    def parse(self, *extra):
        argv = [
            "main.py", "--dataset_name", "mvtec_ad", "--dataset_path", "unused", *extra
        ]
        with patch.object(sys, "argv", argv):
            return get_args()

    def test_default_algorithm(self):
        self.assertEqual(self.parse().pca_algorithm, "two_pass")

    def test_single_pass_is_selectable(self):
        self.assertEqual(self.parse("--pca_algorithm", "single_pass").pca_algorithm, "single_pass")

    def test_single_pass_output_name_is_distinct(self):
        default = generate_run_name(self.parse())
        single = generate_run_name(self.parse("--pca_algorithm", "single_pass"))
        self.assertNotIn("_single-pass", default)
        self.assertEqual(single.replace("_single-pass", ""), default)

    def test_kernel_pca_name_is_unaffected(self):
        default = generate_run_name(self.parse("--use_kernel_pca"))
        single = generate_run_name(
            self.parse("--use_kernel_pca", "--pca_algorithm", "single_pass")
        )
        self.assertEqual(default, single)

    def test_legacy_argument_namespace_keeps_default_name(self):
        args = self.parse()
        expected = generate_run_name(args)
        del args.pca_algorithm
        self.assertEqual(generate_run_name(args), expected)

    def test_invalid_algorithm_is_rejected(self):
        with patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as error:
                self.parse("--pca_algorithm", "unknown")
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
