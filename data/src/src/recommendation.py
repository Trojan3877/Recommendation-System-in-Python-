import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationSystem:
    def __init__(self, interaction_data):
        """
        Initialize the recommendation system.
        :param interaction_data: DataFrame containing user-item interactions.
        """
        self.data = interaction_data
        self.user_item_matrix = self._create_user_item_matrix()

    def _create_user_item_matrix(self):
        """
        Create a user-item matrix.
        :return: Pivot table with users as rows and items as columns.
        """
        return self.data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

    def get_recommendations(self, target_user, top_n=5):
        """
        Generate recommendations for a target user.
        :param target_user: User ID to generate recommendations for.
        :param top_n: Number of recommendations to return.
        :return: List of recommended item IDs.
        """
        user_matrix = self.user_item_matrix.values
        user_similarities = cosine_similarity(user_matrix)
        target_index = self.user_item_matrix.index.get_loc(target_user)

        similarity_scores = user_similarities[target_index]
        weighted_ratings = np.dot(similarity_scores, user_matrix)
        item_scores = pd.Series(weighted_ratings, index=self.user_item_matrix.columns)

        rated_items = self.user_item_matrix.loc[target_user][self.user_item_matrix.loc[target_user] > 0].index
        recommendations = item_scores.drop(rated_items).nlargest(top_n).index.tolist()
        return recommendations
