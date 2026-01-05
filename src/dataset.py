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


class RatingsDataset(Dataset):
    def __init__(self, df_path=DF_PATH, models_path=MODELS_PATH):
        """
        Reads the CSV, encodes User/Book IDs to 0..N integers,
        and saves the encoders for later use.
        """
        self.df = pd.read_csv(df_path / "raw" / "Ratings.csv")
        self.user_encoder = LabelEncoder()
        self.book_encoder = LabelEncoder()

        logger.info("Fitting label encoders for users and books.")
        self.df["user_idx"] = self.user_encoder.fit_transform(self.df["User-ID"])
        self.df["book_idx"] = self.book_encoder.fit_transform(self.df["ISBN"])

        self.user_ids = self.df["user_idx"].values
        self.book_ids = self.df["book_idx"].values
        self.ratings = self.df["Book-Rating"].values

        uencoder_path = models_path / "user_encoder.joblib"
        bencoder_path = models_path / "book_encoder.joblib"
        joblib.dump(self.user_encoder, uencoder_path)
        joblib.dump(self.book_encoder, bencoder_path)

        logger.info(f"Saved user encoder to {uencoder_path}")
        logger.info(f"Saved book encoder to {bencoder_path}")

        self.df.to_csv(df_path / "processed" / "Ratings_encoded.csv", index=False)
        logger.info("Saved processed ratings to Ratings_encoded.csv")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = torch.tensor(self.user_ids[idx], dtype=torch.long)
        item = torch.tensor(self.book_ids[idx], dtype=torch.long)
        rating = torch.tensor(self.ratings[idx], dtype=torch.float32)

        return user, item, rating
