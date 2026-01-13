from fastapi import FastAPI, HTTPException
from utils.book_recommender import BookRecommender
from utils.movie_recommender import MovieRecommender
from src.logger_config import setup_logger

app = FastAPI()
book_recommender = BookRecommender()
movie_recommender = MovieRecommender()
logger = setup_logger("api")


@app.get("/")
def home():
    return {"message": "Recommender API is running."}


@app.get("/recommend/book/{isbn}")
def recommend(isbn: str):
    if not book_recommender.is_valid_isbn(isbn):
        raise HTTPException(status_code=404, detail="ISBN not found in the database.")

    title, recommendations = book_recommender.recommend(isbn)

    response = {
        "input_isbn": isbn,
        "title": title,
        "results": len(recommendations),
        "data": recommendations,
    }
    return response


@app.get("/recommend/movie/{movie_id}")
def recommend_movie(movie_id: int):
    if not movie_recommender.is_valid_movie_id(movie_id):
        raise HTTPException(
            status_code=404, detail="Movie ID not found in the database."
        )

    title, genre, recommendations = movie_recommender.recommend(movie_id)

    response = {
        "input_movie_id": movie_id,
        "title": title,
        "genre": genre,
        "results": len(recommendations),
        "data": recommendations,
    }
    return response
