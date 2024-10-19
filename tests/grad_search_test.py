import unittest

import numpy as np

from scripts.grad_search import GradSearch


class TestGradSearch(unittest.TestCase):

    def setUp(self):
        self.grad_search = GradSearch()
        self.x = np.array([1, 2, 3, 4, 5])
        self.y = np.array([2.5, 3.5, 4.5, 5.5, 6.5])

    def test_initial_parameters(self):
        self.assertEqual(self.grad_search.a, 1.0)
        self.assertEqual(self.grad_search.b, 0.0)
        self.assertEqual(self.grad_search.x0, 1.0)
        self.assertEqual(self.grad_search.l, 1e-2)

    def test_loss_function(self):
        loss = self.grad_search.f(self.x, self.y, self.grad_search.a,
                                  self.grad_search.b, self.grad_search.x0)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

    def test_partial_derivatives(self):
        dfda = self.grad_search.dfda(self.x, self.y, self.grad_search.a,
                                     self.grad_search.b, self.grad_search.x0)
        dfdb = self.grad_search.dfdb(self.x, self.y, self.grad_search.a,
                                     self.grad_search.b, self.grad_search.x0)
        dfdx0 = self.grad_search.dfdx0(self.x, self.y, self.grad_search.a,
                                       self.grad_search.b, self.grad_search.x0)
        self.assertIsInstance(dfda, float)
        self.assertIsInstance(dfdb, float)
        self.assertIsInstance(dfdx0, float)

    def test_fit(self):
        initial_loss = self.grad_search.f(self.x, self.y, self.grad_search.a,
                                          self.grad_search.b, self.grad_search.x0)
        self.grad_search.fit(self.x, self.y, epochs=10)
        final_loss = self.grad_search.loss[-1]
        self.assertLess(final_loss, initial_loss)

    def test_get_a_b_x0(self):
        a, b, x0 = self.grad_search.get_a_b_x0()
        self.assertEqual(a, self.grad_search.a)
        self.assertEqual(b, self.grad_search.b)
        self.assertEqual(x0, self.grad_search.x0)


if __name__ == '__main__':
    unittest.main()
