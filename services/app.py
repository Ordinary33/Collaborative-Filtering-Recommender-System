from fastapi import FastAPI, HTTPException
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
    if not recommender.is_valid_isbn(isbn):
        raise HTTPException(status_code=404, detail="ISBN not found in the database.")

    title, recommendations = recommender.recommend(isbn)

    response = {
        "input_isbn": isbn,
        "title": title,
        "results": len(recommendations),
        "data": recommendations,
    }
    return response
