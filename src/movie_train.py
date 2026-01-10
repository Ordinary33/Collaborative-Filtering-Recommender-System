from src.movie_dataset import MovieRatingsDataset
from src.movie_model import MatrixFactorization
from src.logger_config import setup_logger
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = PROJECT_ROOT / "models"
MODELS_PATH.mkdir(parents=True, exist_ok=True)

logger = setup_logger("movie-train-val")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 256
LR = 0.001


def train(num_epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LR):
    dataset = MovieRatingsDataset(limit=100000)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_df, val_df = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_df, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_df, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    num_users = len(dataset.movie_user_encoder.classes_)
    num_items = len(dataset.movie_encoder.classes_)

    model = MatrixFactorization(num_users, num_items)
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for users, items, ratings in train_loader:
            users = users.to(DEVICE)
            items = items.to(DEVICE)
            ratings = ratings.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(users, items)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}"
        )

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for users, items, ratings in val_loader:
                users = users.to(DEVICE)
                items = items.to(DEVICE)
                ratings = ratings.to(DEVICE)

                outputs = model(users, items)
                loss = criterion(outputs, ratings)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            models_path = MODELS_PATH / "movie_matrix_factorization.pth"
            torch.save(model.state_dict(), models_path)
            logger.info(
                "New best model found! Saved to models/movie_matrix_factorization.pth"
            )

    logger.info("Training Complete")

    return train_losses, val_losses


if __name__ == "__main__":
    train()
