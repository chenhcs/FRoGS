"""Tests for the --LR_model and --demo functionality of gene_vec_model.py.

The heavy training pipeline in gene_vec_model.py lives inside ``main()`` and is
guarded by ``if __name__ == '__main__'``, so importing the module does not
trigger any training. These tests exercise the small, self-contained helpers
that back the --LR_model and --demo command line arguments.

These tests use only the standard-library ``unittest`` framework so they can be
run without pytest:

    # from the src/ directory
    python3 -m unittest test_gene_vec_model -v

(pytest, if installed, can also collect and run this file.)

The logistic-regression tests require scikit-learn; they are skipped
automatically when it is not installed.
"""
import importlib
import os
import pickle
import sys
import tempfile
import unittest

import numpy as np

# Make sure the module under test (and its ``utils`` package) is importable
# regardless of the directory the tests are launched from.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

gvm = importlib.import_module("gene_vec_model")

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:  # scikit-learn is an optional dependency for these tests
    HAS_SKLEARN = False


def _separable_dataset():
    rng = np.random.RandomState(0)
    X_pos = rng.normal(loc=2.0, scale=0.2, size=(40, 3))
    X_neg = rng.normal(loc=-2.0, scale=0.2, size=(40, 3))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 40 + [0] * 40)
    return X, y


class ParseArgsTest(unittest.TestCase):
    def test_accepts_lr_model_and_demo(self):
        argv = [
            "gene_vec_model.py",
            "--datatype", "data/",
            "--association_file", "results/",
            "--outfile", "saved_model/",
            "--LR_model", "LR_model/",
            "--demo", "demo/",
        ]
        old_argv = sys.argv
        try:
            sys.argv = argv
            args = gvm.parse_args()
        finally:
            sys.argv = old_argv
        self.assertEqual(args.LR_model, "LR_model/")
        self.assertEqual(args.demo, "demo/")

    def test_lr_model_and_demo_default_to_none(self):
        old_argv = sys.argv
        try:
            sys.argv = ["gene_vec_model.py"]
            args = gvm.parse_args()
        finally:
            sys.argv = old_argv
        self.assertIsNone(args.LR_model)
        self.assertIsNone(args.demo)


class BuildPairFeaturesTest(unittest.TestCase):
    def test_is_elementwise_product(self):
        encoded_genes = np.array([[1.0, 2.0],
                                  [3.0, 4.0],
                                  [5.0, 6.0]])
        feats = gvm.build_pair_features(encoded_genes, [0, 2], [1, 1])
        expected = np.array([[1.0 * 3.0, 2.0 * 4.0],
                             [5.0 * 3.0, 6.0 * 4.0]])
        self.assertEqual(feats.shape, (2, 2))
        np.testing.assert_allclose(feats, expected)

    def test_casts_float_indices(self):
        # Indices arrive as floats after np.squeeze on sampled data.
        encoded_genes = np.arange(6.0).reshape(3, 2)
        feats = gvm.build_pair_features(
            encoded_genes, np.array([0.0, 1.0]), np.array([2.0, 2.0]))
        expected = np.vstack([encoded_genes[0] * encoded_genes[2],
                              encoded_genes[1] * encoded_genes[2]])
        np.testing.assert_allclose(feats, expected)


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn is required for the LR model tests")
class TrainOrLoadLRModelTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_saves_model_when_absent(self):
        X, y = _separable_dataset()
        lr_dir = os.path.join(self.tmp, "LR_model")

        model, path = gvm.train_or_load_lr_model(X, y, lr_dir)

        self.assertEqual(path, os.path.join(lr_dir, "gene_pair_lr.pkl"))
        self.assertTrue(os.path.exists(path))
        self.assertIsInstance(model, LogisticRegression)
        self.assertAlmostEqual(model.score(X, y), 1.0)
        with open(path, "rb") as f:
            reloaded = pickle.load(f)
        np.testing.assert_allclose(reloaded.coef_, model.coef_)

    def test_creates_directory_if_missing(self):
        X, y = _separable_dataset()
        lr_dir = os.path.join(self.tmp, "nested", "LR_model")
        self.assertFalse(os.path.exists(lr_dir))

        gvm.train_or_load_lr_model(X, y, lr_dir)

        self.assertTrue(os.path.isdir(lr_dir))
        self.assertTrue(os.path.exists(os.path.join(lr_dir, "gene_pair_lr.pkl")))

    def test_loads_existing_model_without_retraining(self):
        lr_dir = os.path.join(self.tmp, "LR_model")
        os.makedirs(lr_dir)
        path = os.path.join(lr_dir, "gene_pair_lr.pkl")

        # Pre-train and persist a model on one dataset.
        X_a, y_a = _separable_dataset()
        original = LogisticRegression(max_iter=1000).fit(X_a, y_a)
        with open(path, "wb") as f:
            pickle.dump(original, f)

        # Call with a *different* dataset; because a model already exists it
        # must be loaded as-is rather than retrained on the new data.
        X_b = X_a + 100.0
        y_b = 1 - y_a
        model, returned_path = gvm.train_or_load_lr_model(X_b, y_b, lr_dir)

        self.assertEqual(returned_path, path)
        np.testing.assert_allclose(model.coef_, original.coef_)
        np.testing.assert_allclose(model.intercept_, original.intercept_)

    def test_custom_filename_is_respected(self):
        X, y = _separable_dataset()
        lr_dir = os.path.join(self.tmp, "LR_model")
        model, path = gvm.train_or_load_lr_model(X, y, lr_dir, filename="custom.pkl")
        self.assertEqual(path, os.path.join(lr_dir, "custom.pkl"))
        self.assertTrue(os.path.exists(path))


class ExportEmbeddingsForDemoTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _write_embeddings(self, path):
        content = "1234,0.1,0.2\n5678,0.3,0.4\n"
        with open(path, "w") as f:
            f.write(content)
        return content

    def test_copies_into_data_subdir_when_present(self):
        outfile = os.path.join(self.tmp, "gene_vec_go.csv")
        content = self._write_embeddings(outfile)
        demo_dir = os.path.join(self.tmp, "demo")
        os.makedirs(os.path.join(demo_dir, "data"))

        dest = gvm.export_embeddings_for_demo(outfile, demo_dir)

        self.assertEqual(dest, os.path.join(demo_dir, "data", "gene_vec_go.csv"))
        self.assertTrue(os.path.exists(dest))
        with open(dest) as f:
            self.assertEqual(f.read(), content)

    def test_copies_into_demo_dir_when_no_data_subdir(self):
        outfile = os.path.join(self.tmp, "gene_vec_go.csv")
        content = self._write_embeddings(outfile)
        demo_dir = os.path.join(self.tmp, "demo")  # no data/ subdirectory

        dest = gvm.export_embeddings_for_demo(outfile, demo_dir)

        self.assertEqual(dest, os.path.join(demo_dir, "gene_vec_go.csv"))
        self.assertTrue(os.path.exists(dest))
        with open(dest) as f:
            self.assertEqual(f.read(), content)

    def test_preserves_outfile_basename(self):
        outfile = os.path.join(self.tmp, "my_embeddings.csv")
        self._write_embeddings(outfile)
        demo_dir = os.path.join(self.tmp, "demo")
        dest = gvm.export_embeddings_for_demo(outfile, demo_dir)
        self.assertEqual(os.path.basename(dest), "my_embeddings.csv")


if __name__ == "__main__":
    unittest.main()
