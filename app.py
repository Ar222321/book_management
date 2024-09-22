from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import List, Optional
from book_manager import BookManager, LLaMAQuick  # Import from book_manager.py
from book_recommendation import BookRecommendation  # Import from book_recommendation.py

app = FastAPI()

# SQLAlchemy setup
DATABASE_URL = "postgresql://postgres:Queen%4009876@localhost:5432/books_management"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Pydantic Models
class BookDetails(BaseModel):
    ID: int
    Title: str
    Author: str
    Genre: str
    Year_Published: int
    Summary: Optional[str] = None

class ReviewDetails(BaseModel):
    ID: int
    Book_ID: int
    User_ID: int
    Review_Text: str
    Rating: float

class RecommendationRequest(BaseModel):
    genre: str
    min_rating: float

class SummaryRequest(BaseModel):
    content: str

# Initialize LLaMA Model for text generation
model_path = "C:\\Users\\PE586UG\\OneDrive - EY\\Documents\\Gen AI\\jk\\Sheared-LLaMA-1.3B"
llama_model = LLaMAQuick(model_path)

# Initialize Book Manager and Recommendation System
book_manager = BookManager(SessionLocal(), llama_model)
recommendation_engine = BookRecommendation(DATABASE_URL)

# Routes

## Books

# POST /books: Add a new book
@app.post("/books/")
async def add_book(book: BookDetails):
    try:
        book_manager.add_new_book(book.dict())
        return {"message": "Book added successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# GET /books: Retrieve all books
@app.get("/books/")
async def get_all_books():
    books = book_manager.get_all_books()
    if books:
        return books
    raise HTTPException(status_code=404, detail="No books found")

# GET /books/{id}: Retrieve a specific book by its ID
@app.get("/books/{id}")
async def get_book(id: int = Path(..., description="The ID of the book to retrieve")):
    book = book_manager.get_book_by_id(id)
    if book:
        return book
    raise HTTPException(status_code=404, detail="Book not found")

# PUT /books/{id}: Update a book's information by its ID
@app.put("/books/{id}")
async def update_book(id: int, book: BookDetails):
    updated_book = book_manager.update_book(id, book.dict())
    if updated_book:
        return updated_book
    raise HTTPException(status_code=404, detail="Book not found")

# DELETE /books/{id}: Delete a book by its ID
@app.delete("/books/{id}")
async def delete_book(id: int):
    deleted = book_manager.delete_book(id)
    if deleted:
        return {"message": f"Book with ID {id} deleted successfully"}
    raise HTTPException(status_code=404, detail="Book not found")

## Reviews

# POST /books/{id}/reviews: Add a review for a book
@app.post("/books/{id}/reviews/")
async def add_review(id: int, review: ReviewDetails):
    try:
        review.Book_ID = id
        book_manager.add_review(review.dict())
        return {"message": "Review added successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# GET /books/{id}/reviews: Retrieve all reviews for a book
@app.get("/books/{id}/reviews/")
async def get_reviews(id: int):
    reviews = book_manager.get_reviews_for_book(id)
    if reviews:
        return reviews
    raise HTTPException(status_code=404, detail="No reviews found for this book")

## Book Summary and Ratings

# GET /books/{id}/summary: Get a summary and aggregated rating for a book
@app.get("/books/{id}/summary/")
async def get_book_summary(id: int):
    try:
        summary = book_manager.get_book_summary(id)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# POST /generate-summary: Generate a summary for a given book content
@app.post("/generate-summary/")
async def generate_summary(request: SummaryRequest):
    try:
        summary = llama_model.generate_text(request.content)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

## Book Recommendations

# GET /recommendations: Get book recommendations based on user preferences
@app.post("/recommendations/")
async def get_recommendations(request: RecommendationRequest):
    try:
        recommendations = recommendation_engine.recommend_books(request.genre, request.min_rating)
        return recommendations.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

