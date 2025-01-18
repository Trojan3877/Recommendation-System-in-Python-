import unittest
from src.utils import load_interaction_data
from src.recommendation import RecommendationSystem

class TestRecommendationSystem(unittest.TestCase):
    def setUp(self):
        self.data = load_interaction_data('data/interactions.csv')
        self.recommender = RecommendationSystem(self.data)

    def test_recommendations(self):
        recommendations = self.recommender.get_recommendations(target_user=1, top_n=2)
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

if __name__ == '__main__':
    unittest.main()
