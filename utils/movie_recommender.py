from pathlib import Path
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
from src.movie_model import MatrixFactorization
from src.movie_dataset import MovieRatingsDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DF_PATH = PROJECT_ROOT / "data" / "raw" / "Movies.csv"


class MovieRecommender:
    def __init__(self):
        self.model_path = (
            PROJECT_ROOT / "models" / "movie_matrix_factorization_checkpoint.pth"
        )
        self.movie_encoder = joblib.load(
            PROJECT_ROOT / "models" / "movie_encoder.joblib"
        )
        self.user_encoder = joblib.load(
            PROJECT_ROOT / "models" / "movie_user_encoder.joblib"
        )
        self.dataset = MovieRatingsDataset()
        self.model = None
        self.item_embeddings = None
        self.movies_df = None
        self.knn = None

        self.valid_movie_ids = set(self.movie_encoder.classes_)
        self.load_resources()

    def load_resources(self):
        self.model = MatrixFactorization(
            num_users=len(self.user_encoder.classes_),
            num_items=len(self.movie_encoder.classes_),
        )
        self.checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()
        self.item_embeddings = self.model.item_embedding.weight.data.numpy()
        self.knn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
        self.knn.fit(self.item_embeddings)
        self.movies_df = pd.read_csv(DF_PATH, low_memory=False)

    def is_valid_movie_id(self, movie_id: int) -> bool:
        return movie_id in self.valid_movie_ids

    def recommend(self, movie_id: int):
        """
            Function to get movie recommendations based on user input.

        Args:
            movie_id (int): Id of the movie for which recommendations are sought.

        Output:
            list: A list containing recommended movie titles and their details.
        """

        if not self.is_valid_movie_id(movie_id):
            return []

        encoded_id = self.movie_encoder.transform([movie_id])[0]
        distances, indices = self.knn.kneighbors(
            [self.item_embeddings[encoded_id]], n_neighbors=11
        )
        query_title = str(
            self.movies_df[self.movies_df["movieId"] == movie_id]["title"].values[0]
        )
        recommendations = []

        for i in range(1, len(indices[0])):
            rec_idx = indices[0][i]
            rec_movie_id = self.movie_encoder.inverse_transform([rec_idx])[0]
            rec_movie = self.movies_df[self.movies_df["movieId"] == rec_movie_id]

            if not rec_movie.empty:
                movie_info = rec_movie.iloc[0]
                recommendations.append(
                    {
                        "movieId": str(movie_info["movieId"]),
                        "title": str(movie_info["title"]),
                        "genres": str(movie_info["genres"]),
                    }
                )

        return query_title, recommendations
