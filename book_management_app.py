import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from asyn_book_manager import BookManager, LLaMAQuick
from jwt_utils import create_access_token, verify_token
from asyn_book_recommendation import BookRecommendation  # Adjust the path as necessary

app = FastAPI()

# Database setup
DATABASE_URL = "postgresql+asyncpg://postgres:Queen%4009876@localhost:5432/books_management"
async_engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)
book_recommendation = BookRecommendation(DATABASE_URL)

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

class UserPreferences(BaseModel):
    genre: str
    min_rating: float

# Initialize LLaMA Model for text generation
model_path = r"C:\Users\PE586UG\OneDrive - EY\Documents\Gen AI\jk\Sheared-LLaMA-1.3B"
llama_model = LLaMAQuick(model_path)

# Dependency for getting the database session
async def get_db() -> AsyncSession:
    async with SessionLocal() as session:
        yield session

# Token generation and verification
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Token Generation Endpoint
@app.post("/token")
async def generate_token(username: str):
    access_token = create_access_token({"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

# Verify token
async def get_current_user(token: str = Security(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = verify_token(token)
    if payload is None:
        raise credentials_exception
    return payload

# Routes
@app.post("/books/")
async def add_book(book: BookDetails, db: AsyncSession = Depends(get_db), current_user: dict = Depends(get_current_user)):
    book_manager = BookManager(db, llama_model)
    await book_manager.add_new_book(book.dict())
    return {"message": "Book added successfully"}

@app.get("/books/")
async def get_all_books(db: AsyncSession = Depends(get_db), current_user: dict = Depends(get_current_user)):
    book_manager = BookManager(db, llama_model)
    books = await book_manager.get_all_books()
    if books:
        return books
    raise HTTPException(status_code=404, detail="No books found")

@app.get("/books/{id}")
async def get_book(id: int, db: AsyncSession = Depends(get_db), current_user: dict = Depends(get_current_user)):
    book_manager = BookManager(db, llama_model)
    book = await book_manager.get_book_by_id(id)
    if book:
        return book
    raise HTTPException(status_code=404, detail="Book not found")

@app.put("/books/{id}")
async def update_book(id: int, book: BookDetails, db: AsyncSession = Depends(get_db), current_user: dict = Depends(get_current_user)):
    book_manager = BookManager(db, llama_model)
    updated_book = await book_manager.update_book(id, book.dict())
    if updated_book:
        return updated_book
    raise HTTPException(status_code=404, detail="Book not found")

@app.delete("/books/{id}")
async def delete_book(id: int, db: AsyncSession = Depends(get_db), current_user: dict = Depends(get_current_user)):
    book_manager = BookManager(db, llama_model)
    deleted = await book_manager.delete_book(id)
    if deleted:
        return {"message": f"Book with ID {id} deleted successfully"}
    raise HTTPException(status_code=404, detail="Book not found")

@app.post("/books/{id}/reviews/")
async def add_review(id: int, review: ReviewDetails, db: AsyncSession = Depends(get_db), current_user: dict = Depends(get_current_user)):
    book_manager = BookManager(db, llama_model)
    review.Book_ID = id
    await book_manager.add_review(review.dict())
    return {"message": "Review added successfully"}

@app.get("/books/{id}/reviews/")
async def get_reviews(id: int, db: AsyncSession = Depends(get_db), current_user: dict = Depends(get_current_user)):
    book_manager = BookManager(db, llama_model)
    reviews = await book_manager.get_reviews_for_book(id)
    if reviews:
        return reviews
    raise HTTPException(status_code=404, detail="No reviews found for this book")

@app.post("/recommendations/")
async def get_book_recommendations(user_preferences: UserPreferences, db: AsyncSession = Depends(get_db)):
    await book_recommendation.load_data()  # Load the data if not already loaded
    await book_recommendation.train_model()  # Train the model if not already trained
    recommendations = await book_recommendation.recommend_books(user_preferences.genre, user_preferences.min_rating)
    
    # Check if the response is a dictionary (for messages)
    if isinstance(recommendations, dict):
        return recommendations  # Return the message directly
    
    # If recommendations is a DataFrame, convert to a list of dictionaries
    return recommendations.to_dict(orient='records')  # Return the recommendations as a list of dictionaries
