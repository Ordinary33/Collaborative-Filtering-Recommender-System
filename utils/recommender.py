from pathlib import Path
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
from src.model import MatrixFactorization
from src.dataset import RatingsDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DF_PATH = PROJECT_ROOT / "data" / "raw" / "Books.csv"


class Recommender:
    def __init__(self):
        self.model_path = PROJECT_ROOT / "models" / "matrix_factorization.pth"
        self.book_encoder = joblib.load(PROJECT_ROOT / "models" / "book_encoder.joblib")
        self.user_encoder = joblib.load(PROJECT_ROOT / "models" / "user_encoder.joblib")
        self.dataset = RatingsDataset()
        self.model = None
        self.item_embeddings = None
        self.books_df = None
        self.knn = None

        self.load_resources()

    def load_resources(self):
        self.model = MatrixFactorization(
            num_users=len(self.user_encoder.classes_),
            num_items=len(self.book_encoder.classes_),
        )
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=torch.device("cpu"))
        )
        self.model.eval()
        self.item_embeddings = self.model.item_embedding.weight.data.numpy()
        self.knn = NearestNeighbors(n_neighbors=10, metric="cosine", algorithm="brute")
        self.knn.fit(self.item_embeddings)
        self.books_df = pd.read_csv(DF_PATH, low_memory=False)

    def recommend(self, isbn: str):
        """
            Function to get book recommendations based on user input.
        Args:
            input_data (int): The ISBN of the book for which recommendations are sought.

        Returns:
            list: A list containing recommended book titles and their details.
        """
        try:
            encoded_id = self.book_encoder.transform([isbn])[0]
        except ValueError:
            print(f"ISBN {isbn} not found in the encoder.")
            return []

        distances, indices = self.knn.kneighbors(
            [self.item_embeddings[encoded_id]], n_neighbors=11
        )

        recommendations = []
        for i in range(1, len(indices[0])):
            books_id = indices[0][i]
            recommended_isbns = self.book_encoder.inverse_transform([books_id])[0]
            book_matches = self.books_df[self.books_df["ISBN"] == recommended_isbns]

            if not book_matches.empty:
                book_info = book_matches.iloc[0]
                try:
                    year = int(book_info["Year-Of-Publication"])
                except ValueError:
                    year = 0
                recommendations.append(
                    {
                        "Title": book_info["Book-Title"],
                        "Author": book_info["Book-Author"],
                        "Year": year,
                        "Image": book_info["Image-URL-L"],
                        "ISBN": str(book_info["ISBN"]),
                    }
                )

        return recommendations


if __name__ == "__main__":
    recommender = Recommender()
    sample_isbn = "0395177111"
    recs = recommender.recommend(sample_isbn)
    for rec in recs:
        print(rec)
