import unittest
import numpy as np
from metrics import Measurer


class TestMeasurer(unittest.TestCase):
    def setUp(self):
        self.m = Measurer()

    def test_disjunct(self):
        a = np.array([[1., 0.], [0., 1.]])
        b = np.array([[0., 0.], [1., 0.]])
        self.assertEqual(self.m._union_area(a, b), 3)
        self.assertEqual(self.m._intersection_area(a, b), 0)
        self.assertAlmostEqual(self.m._volumetric_overlap(a, b), 0.00, places=2)
        self.assertAlmostEqual(self.m._relative_absolute_volume_difference(a, b), 1.00, places=2)

    def test_conjunct(self):
        a = np.array([[1., 1.], [0., 1.]])
        b = np.array([[1., 0.], [1., 0.]])
        self.assertEqual(self.m._union_area(a, b), 4)
        self.assertEqual(self.m._intersection_area(a, b), 1)
        self.assertAlmostEqual(self.m._volumetric_overlap(a, b), 0.25, places=2)
        self.assertAlmostEqual(self.m._relative_absolute_volume_difference(a, b), 0.50, places=2)

    def test_contained(self):
        a = np.array([[1., 1.], [0., 1.]])
        b = np.array([[1., 0.], [0., 1.]])
        self.assertEqual(self.m._union_area(a, b), 3)
        self.assertEqual(self.m._intersection_area(a, b), 2)
        self.assertAlmostEqual(self.m._volumetric_overlap(a, b), 0.67, places=2)
        self.assertAlmostEqual(self.m._relative_absolute_volume_difference(a, b), 0.50, places=2)

    def test_identical(self):
        a = np.array([[1., 1.], [0., 1.]])
        b = np.array([[1., 1.], [0., 1.]])
        self.assertEqual(self.m._union_area(a, b), 3)
        self.assertEqual(self.m._intersection_area(a, b), 3)
        self.assertAlmostEqual(self.m._volumetric_overlap(a, b), 1.00, places=2)
        self.assertAlmostEqual(self.m._relative_absolute_volume_difference(a, b), 0.00, places=2)

    def test_empty(self):
        a = np.array([[0., 0.], [0., 0.]])
        b = np.array([[0., 0.], [0., 0.]])
        self.assertEqual(self.m._union_area(a, b), 0)
        self.assertEqual(self.m._intersection_area(a, b), 0)
        with self.assertWarns(RuntimeWarning):
            self.m._volumetric_overlap(a, b)
        with self.assertWarns(RuntimeWarning):
            self.m._relative_absolute_volume_difference(a, b)

    def test_border(self):
        a = np.array([[0., 0., 1., 1., 1.],
                      [1., 1., 1., 1., 1.],
                      [0., 1., 1., 1., 1.],
                      [0., 1., 1., 1., 1.],
                      [1., 0., 1., 1., 1.]])
        b = np.array([[0., 0., 1., 1., 1.],
                      [1., 1., 1., 0., 1.],
                      [0., 1., 0., 0., 1.],
                      [0., 1., 1., 0., 1.],
                      [1., 0., 1., 1., 1.]])
        np.testing.assert_array_equal(self.m._get_border(a), b > 0.5)

    def test_distances_between_borders(self):
        a = np.array([[0., 1., 1., 1., 1.],
                      [1., 0., 0., 0., 1.],
                      [0., 1., 0., 0., 1.],
                      [0., 1., 0., 0., 1.],
                      [0., 0., 1., 1., 1.]])
        b = np.array([[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 1., 1., 1.],
                      [0., 0., 1., 0., 1.],
                      [0., 0., 1., 1., 1.]])
        distances_of_a_to_b = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
        np.testing.assert_array_equal(self.m._min_distances_between_borders(b, a), distances_of_a_to_b)
        distances_of_b_to_a = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(self.m._min_distances_between_borders(a, b), distances_of_b_to_a)
        self.assertAlmostEqual(self.m._average_symmetric_surface_distance(a, b), 0.86, places=2)
        self.assertAlmostEqual(self.m._rms_symmetric_surface_distance(a, b), 1.31, places=2)
        self.assertEqual(self.m._max_symmetric_surface_distance(a, b), 3)

    def test_empty_borders(self):
        a = np.array([[0., 0.], [0., 0.]])
        b = np.array([[1., 0.], [1., 1.]])
        self.assertEqual(self.m._min_distances_between_borders(a, b), np.array([np.inf]))


if __name__ == '__main__':
    unittest.main()
