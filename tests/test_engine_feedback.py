import tempfile
import unittest

from tests import _bootstrap  # noqa: F401

from betpredictor.engine import PredictionEngine
from betpredictor.feedback import FeedbackLoop, combo_hit
from betpredictor.models.ml import LogisticBlender
from betpredictor.storage import Storage


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.eng = PredictionEngine()

    def test_predict_fields_consistent(self):
        p = self.eng.predict("premier-league", "Manchester City", "Coventry City")
        self.assertAlmostEqual(p.p_home + p.p_draw + p.p_away, 1.0, places=5)
        self.assertGreaterEqual(p.p_draw_or_over_2_5, 0.0)
        self.assertLessEqual(p.p_draw_or_over_2_5, 1.0)

    def test_home_advantage_effect(self):
        home = self.eng.expected_goals("championship", "Burnley", "Bristol City")[0]
        # Same pairing reversed: Burnley away should have lower xG than at home.
        away_when_visiting = self.eng.expected_goals("championship", "Bristol City", "Burnley")[1]
        self.assertGreater(home, away_when_visiting)

    def test_unknown_team_uses_defaults(self):
        p = self.eng.predict("league-one", "Some New FC", "Another New FC")
        self.assertTrue(0.0 <= p.p_draw_or_over_2_5 <= 1.0)

    def test_value_edge_computed(self):
        p = self.eng.predict("premier-league", "Manchester City", "Coventry City", odds=2.0)
        self.assertIsNotNone(p.edge)
        self.assertAlmostEqual(p.edge, p.p_draw_or_over_2_5 * 2.0 - 1.0, places=6)

    def test_screen_sorted_and_filtered(self):
        fx = [
            ("premier-league", "Manchester City", "Coventry City"),
            ("championship", "Preston North End", "Blackburn Rovers"),
        ]
        picks = self.eng.screen(fx, threshold=0.5)
        probs = [p.p_draw_or_over_2_5 for p in picks]
        self.assertEqual(probs, sorted(probs, reverse=True))


class TestCombo(unittest.TestCase):
    def test_combo_hit_rules(self):
        self.assertTrue(combo_hit(1, 1))    # draw, 2 goals
        self.assertTrue(combo_hit(0, 0))    # draw, 0 goals
        self.assertTrue(combo_hit(2, 1))    # 3 goals
        self.assertTrue(combo_hit(3, 0))    # 3 goals decisive
        self.assertFalse(combo_hit(1, 0))   # losing scoreline
        self.assertFalse(combo_hit(2, 0))
        self.assertFalse(combo_hit(0, 2))


class TestML(unittest.TestCase):
    def test_blender_learns_separable_signal(self):
        # High-signal rows are labelled 1, low-signal rows 0. After fitting, a
        # high row must score above a low row.
        high = [0.9, 3.2, 10, 2.1, 1.9, 1.6, 1.5]
        low = [0.4, 2.1, 300, 1.8, 2.1, 1.0, 1.0]
        X = [high if i % 2 else low for i in range(60)]
        y = [1 if i % 2 else 0 for i in range(60)]
        b = LogisticBlender(epochs=300).fit(X, y)
        self.assertGreater(b.predict_proba(high), b.predict_proba(low))

    def test_unfit_blender_is_noop(self):
        b = LogisticBlender()
        self.assertEqual(b.predict_proba([0.77, 3, 1, 2, 2, 1, 1]), 0.77)

    def test_roundtrip_json(self):
        X = [[0.8, 3, 10, 2, 2, 1.5, 1.5], [0.3, 2, 200, 1.8, 2.2, 1, 1]]
        y = [1, 0]
        b = LogisticBlender(epochs=50).fit(X, y)
        b2 = LogisticBlender.from_json(b.to_json())
        self.assertAlmostEqual(b.predict_proba(X[0]), b2.predict_proba(X[0]), places=9)


class TestFeedbackLoop(unittest.TestCase):
    def _loop(self):
        db = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
        return FeedbackLoop(PredictionEngine(), storage=Storage(db_path=db))

    def test_record_and_score(self):
        loop = self._loop()
        p = loop.engine.predict("premier-league", "Manchester City", "Coventry City")
        loop.record_predictions("2026-09-05", [p])
        hit = loop.record_result("2026-09-05", "Manchester City", "Coventry City", 3, 1)
        self.assertTrue(hit)
        perf = loop.performance(avg_odds=1.3)
        self.assertEqual(perf.n, 1)
        self.assertEqual(perf.hit_rate, 1.0)

    def test_retrain_needs_min_samples(self):
        loop = self._loop()
        out = loop.retrain(min_samples=30)
        self.assertFalse(out["retrained"])

    def test_user_feedback_persists(self):
        loop = self._loop()
        fid = loop.record_user_feedback(rating=4, comment="looks right")
        self.assertIsInstance(fid, int)
        self.assertEqual(len(loop.storage.all_feedback()), 1)


if __name__ == "__main__":
    unittest.main()
