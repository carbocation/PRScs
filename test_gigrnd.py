#!/usr/bin/env python3

"""Tests for the fused generalized inverse Gaussian update."""


import unittest

import numpy as np

import gigrnd


class GigVectorTests(unittest.TestCase):
    def setUp(self):
        self.p = 100
        self.delta = np.linspace(0.1, 2.0, self.p)
        self.beta = np.linspace(0.0001, 0.01, self.p)

    def test_seeded_vector_update_is_reproducible(self):
        first = np.empty(self.p)
        second = np.empty(self.p)
        gigrnd.seed_rng(123)
        gigrnd.gig_rvs_vec(
            first, 0.5, self.delta, self.beta, 1.0, 200000
        )
        gigrnd.seed_rng(123)
        gigrnd.gig_rvs_vec(
            second, 0.5, self.delta, self.beta, 1.0, 200000
        )
        np.testing.assert_array_equal(first, second)

    def test_vector_update_matches_scalar_draw_sequence(self):
        vector = np.empty(self.p)
        scalar = np.empty(self.p)
        gigrnd.seed_rng(456)
        gigrnd.gig_rvs_vec(
            vector, 0.5, self.delta, self.beta, 1.0, 200000
        )
        gigrnd.seed_rng(456)
        for jj in range(self.p):
            scalar[jj] = gigrnd.gigrnd(
                0.5,
                2.0 * self.delta[jj],
                200000 * self.beta[jj] * self.beta[jj],
            )
        np.testing.assert_allclose(vector, scalar, rtol=1e-15, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
