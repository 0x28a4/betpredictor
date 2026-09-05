import unittest

from tests import _bootstrap  # noqa: F401  (sets import path)

from betpredictor.models.poisson import DixonColesModel, _poisson_pmf


class TestPoisson(unittest.TestCase):
    def test_pmf_sums_to_one(self):
        total = sum(_poisson_pmf(k, 1.7) for k in range(30))
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_matrix_is_normalised(self):
        m = DixonColesModel().score_matrix(1.6, 1.1)
        total = sum(sum(row) for row in m)
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_matrix_non_negative(self):
        m = DixonColesModel(rho=-0.12).score_matrix(1.4, 1.3)
        self.assertTrue(all(v >= 0 for row in m for v in row))

    def test_expected_goals_recovered(self):
        # With rho=0 (independent Poisson) the matrix means equal the inputs.
        m = DixonColesModel(rho=0.0, max_goals=15).score_matrix(1.8, 1.2)
        eh, ea = DixonColesModel.expected_goals_from_matrix(m)
        self.assertAlmostEqual(eh, 1.8, places=2)
        self.assertAlmostEqual(ea, 1.2, places=2)

    def test_stronger_home_wins_more(self):
        m = DixonColesModel().score_matrix(2.2, 0.7)
        n = len(m)
        p_home = sum(m[i][j] for i in range(n) for j in range(n) if i > j)
        p_away = sum(m[i][j] for i in range(n) for j in range(n) if i < j)
        self.assertGreater(p_home, p_away)


if __name__ == "__main__":
    unittest.main()
