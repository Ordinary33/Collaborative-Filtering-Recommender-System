from fastapi import FastAPI
from utils.recommender import Recommender
from src.logger_config import setup_logger

app = FastAPI()
recommender = Recommender()
logger = setup_logger("api")


@app.get("/")
def home():
    return {"message": "Book Recommender API is running."}


@app.get("/recommend/{isbn}")
def recommend(isbn: str):
    recommendations = recommender.recommend(isbn)

    response = {
        "input_isbn": isbn,
        "results": len(recommendations),
        "data": recommendations,
    }
    return response
