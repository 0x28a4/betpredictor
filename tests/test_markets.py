import unittest

from tests import _bootstrap  # noqa: F401

from betpredictor.models.poisson import DixonColesModel
from betpredictor.markets import derive_markets, draw_or_over_2_5, prob_over


class TestMarkets(unittest.TestCase):
    def setUp(self):
        self.m = DixonColesModel().score_matrix(1.7, 1.3)

    def test_1x2_sums_to_one(self):
        mk = derive_markets(self.m)
        self.assertAlmostEqual(mk.p_home + mk.p_draw + mk.p_away, 1.0, places=6)

    def test_over_under_complement(self):
        mk = derive_markets(self.m)
        self.assertAlmostEqual(mk.p_over_2_5 + mk.p_under_2_5, 1.0, places=6)

    def test_combo_equals_inclusion_exclusion(self):
        """P(draw or over2.5) must equal P(draw)+P(over)-P(draw AND over)."""
        n = len(self.m)
        p_draw = sum(self.m[i][i] for i in range(n))
        p_over = prob_over(self.m, 2.5)
        # draw AND over2.5 -> scorelines 2-2, 3-3, ... (total >= 3, even, equal)
        p_draw_and_over = sum(self.m[i][i] for i in range(2, n))
        expected = p_draw + p_over - p_draw_and_over
        self.assertAlmostEqual(draw_or_over_2_5(self.m), expected, places=9)

    def test_combo_equals_one_minus_four_losing_cells(self):
        losing = self.m[1][0] + self.m[2][0] + self.m[0][1] + self.m[0][2]
        self.assertAlmostEqual(draw_or_over_2_5(self.m), 1.0 - losing, places=9)

    def test_combo_in_unit_interval(self):
        for hx, ax in [(0.5, 0.5), (3.0, 2.5), (0.2, 4.0), (1.4, 1.4)]:
            m = DixonColesModel().score_matrix(hx, ax)
            p = draw_or_over_2_5(m)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_high_scoring_game_high_combo(self):
        m = DixonColesModel().score_matrix(2.4, 2.2)
        self.assertGreater(draw_or_over_2_5(m), 0.85)


if __name__ == "__main__":
    unittest.main()
