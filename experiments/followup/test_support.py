import unittest
import numpy as np
from audit_support import features, support
from knn_control import weights


class SupportTests(unittest.TestCase):
    def test_similarity_coordinates_remove_common_length_scale(self):
        raw = np.array([[.2, 1.4, .02, 4.], [.4, 1.4, .04, 4.]])
        np.testing.assert_allclose(features(raw)[0], features(raw)[1])

    def test_coordinate_box_does_not_imply_joint_support(self):
        train = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        out, fit = support(train, np.array([[.8, .8, .8], [.1, .1, .1], [1.1, 0, 0]]))
        np.testing.assert_array_equal(out["within_training_coordinate_bounds"], [True, True, False])
        np.testing.assert_array_equal(out["inside_training_convex_hull"], [False, True, False])
        self.assertEqual(fit["training_max"], [1., 1., 1.])

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            features([[0, 1, .1, 4]])

    def test_exact_neighbors_do_not_divide_by_zero_or_mix_distant_fields(self):
        np.testing.assert_array_equal(weights([0., 1., 0.]), [.5, 0., .5])
        np.testing.assert_allclose(weights([1., 2.]), [2 / 3, 1 / 3])


if __name__ == "__main__":
    unittest.main()
