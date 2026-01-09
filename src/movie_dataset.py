import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
from src.logger_config import setup_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DF_PATH = PROJECT_ROOT / "data"
MODELS_PATH = PROJECT_ROOT / "models"
MODELS_PATH.mkdir(parents=True, exist_ok=True)

logger = setup_logger(__name__)


class MovieRatingsDataset(Dataset):
    def __init__(self, df_path=DF_PATH, models_path=MODELS_PATH):
        """
        Reads the CSV, encodes User/Movie IDs to 0..N integers,
        and saves the encoders for later use.
        """
        movie_ratings_df = pd.read_csv(
            df_path / "raw" / "Movie_Ratings.csv", low_memory=False
        )
        movie_df = pd.read_csv(df_path / "raw" / "Movies.csv", low_memory=False)

        movie_ratings_df = movie_ratings_df.drop("timestamp", axis=1)

        valid_id = set(movie_df["movieId"].unique())

        self.df = movie_ratings_df[movie_ratings_df["movieId"].isin(valid_id)].copy()

        self.movie_user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()

        logger.info("Fitting label encoders for users and movies.")
        self.df["user_idx"] = self.movie_user_encoder.fit_transform(self.df["userId"])
        self.df["movie_idx"] = self.movie_encoder.fit_transform(self.df["movieId"])

        self.user_ids = self.df["user_idx"].values
        self.movie_ids = self.df["movie_idx"].values
        self.ratings = self.df["rating"].values

        uencoder_path = models_path / "movie_user_encoder.joblib"
        mencoder_path = models_path / "movie_encoder.joblib"
        joblib.dump(self.movie_user_encoder, uencoder_path)
        joblib.dump(self.movie_encoder, mencoder_path)

        logger.info(f"Saved user encoder to {uencoder_path}")
        logger.info(f"Saved movie encoder to {mencoder_path}")

        processed_path = df_path / "processed"
        processed_path.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(processed_path / "Movie_Ratings_encoded.csv", index=False)
        logger.info("Saved processed ratings to Movie_Ratings_encoded.csv")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = torch.tensor(self.user_ids[idx], dtype=torch.long)
        item = torch.tensor(self.movie_ids[idx], dtype=torch.long)
        rating = torch.tensor(self.ratings[idx], dtype=torch.float32)

        return user, item, rating
