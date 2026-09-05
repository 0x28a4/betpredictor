import unittest

from tests import _bootstrap  # noqa: F401

from betpredictor.data.sources import _parse_espn_scoreboard, Fixture
from betpredictor.data.calibrate import estimate_strengths
from betpredictor.data.sample_data import normalize_team_name

# A trimmed but structurally faithful ESPN scoreboard payload: one scheduled
# game and one finished game.
ESPN_PAYLOAD = {
    "events": [
        {
            "id": "1",
            "date": "2026-09-05T14:00Z",
            "name": "Coventry City at Manchester City",
            "status": {"type": {"state": "pre", "completed": False}},
            "competitions": [
                {
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Manchester City", "name": "Man City"}, "score": ""},
                        {"homeAway": "away", "team": {"displayName": "Coventry City", "name": "Coventry"}, "score": ""},
                    ]
                }
            ],
        },
        {
            "id": "2",
            "date": "2026-09-05T11:30Z",
            "name": "Bristol City at Burnley",
            "status": {"type": {"state": "post", "completed": True}},
            "competitions": [
                {
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Burnley"}, "score": "2"},
                        {"homeAway": "away", "team": {"displayName": "Bristol City"}, "score": "1"},
                    ]
                }
            ],
        },
    ]
}


class TestEspnParser(unittest.TestCase):
    def setUp(self):
        self.fx = _parse_espn_scoreboard(ESPN_PAYLOAD, "premier-league")

    def test_parses_both_events(self):
        self.assertEqual(len(self.fx), 2)

    def test_home_away_orientation(self):
        game = self.fx[0]
        self.assertEqual(game.home, "Manchester City")
        self.assertEqual(game.away, "Coventry City")

    def test_scheduled_has_no_score(self):
        game = self.fx[0]
        self.assertEqual(game.status, "SCHEDULED")
        self.assertIsNone(game.home_goals)
        self.assertIsNone(game.away_goals)

    def test_finished_has_score(self):
        game = self.fx[1]
        self.assertEqual(game.status, "FINISHED")
        self.assertEqual(game.home_goals, 2)
        self.assertEqual(game.away_goals, 1)

    def test_source_tagged(self):
        self.assertTrue(all(g.source == "espn" for g in self.fx))

    def test_skips_incomplete_events(self):
        bad = {"events": [{"competitions": [{"competitors": [{"homeAway": "home", "team": {"displayName": "X"}}]}]}]}
        self.assertEqual(_parse_espn_scoreboard(bad, "premier-league"), [])


class TestNormalize(unittest.TestCase):
    def test_alias_and_fc_suffix(self):
        self.assertEqual(normalize_team_name("Bournemouth"), "AFC Bournemouth")
        self.assertEqual(normalize_team_name("Spurs"), "Tottenham Hotspur")
        self.assertEqual(normalize_team_name("Bristol City FC"), "Bristol City")
        self.assertEqual(normalize_team_name("Manchester City"), "Manchester City")


class TestCalibrate(unittest.TestCase):
    def _finished(self):
        # Team H scores freely and concedes little; team L is the opposite.
        games = []
        for _ in range(8):
            games.append(Fixture("championship", "Attackers", "Minnows",
                                  home_goals=3, away_goals=0, status="FINISHED"))
            games.append(Fixture("championship", "Minnows", "Attackers",
                                  home_goals=0, away_goals=2, status="FINISHED"))
        return games

    def test_strength_ordering(self):
        s = estimate_strengths(self._finished())
        self.assertIn("Attackers", s)
        self.assertGreater(s["Attackers"]["attack"], s["Minnows"]["attack"])
        self.assertLess(s["Attackers"]["defense"], s["Minnows"]["defense"])

    def test_ignores_unfinished(self):
        games = [Fixture("league-one", "A", "B", home_goals=None, away_goals=None)]
        self.assertEqual(estimate_strengths(games), {})

    def test_shrinkage_pulls_small_samples_toward_one(self):
        one = [Fixture("league-one", "A", "B", home_goals=5, away_goals=0, status="FINISHED")]
        s = estimate_strengths(one)
        # A scored 5 (huge) but on one game shrinkage keeps attack well below 5x.
        self.assertLess(s["A"]["attack"], 2.5)
        self.assertGreater(s["A"]["attack"], 1.0)


if __name__ == "__main__":
    unittest.main()
