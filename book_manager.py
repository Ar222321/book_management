from sqlalchemy import Column, Integer, String, ForeignKey, Float, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Base for SQLAlchemy ORM Models
Base = declarative_base()

# Book table schema
class Book(Base):
    __tablename__ = 'books'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    author = Column(String)
    genre = Column(String)
    year_published = Column(Integer)
    summary = Column(String)

# Review table schema
class Review(Base):
    __tablename__ = 'reviews'
    id = Column(Integer, primary_key=True)
    book_id = Column(Integer, ForeignKey('books.id'))
    user_id = Column(Integer)
    review_text = Column(String)
    rating = Column(Float)

# LLaMA Model Integration Class
class LLaMAQuick:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_text(self, prompt, max_length=150, num_beams=2):
        inputs = self.tokenizer(prompt, return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            output = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_p=0.9
            )
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

# Book Manager class to handle DB operations and LLaMA summary generation
class BookManager:
    def __init__(self, db_session: AsyncSession, llama_model: LLaMAQuick):
        self.db_session = db_session
        self.llama_model = llama_model

    async def add_new_book(self, book_details):
        book_id = book_details['ID']
        title = book_details['Title']
        author = book_details['Author']
        genre = book_details['Genre']
        year_published = book_details['Year_Published']
        user_provided_summary = book_details.get('Summary', '')

        if not user_provided_summary.strip():
            description = f"The book {title} by {author} is a {genre} published in {year_published}."
            summary = self.llama_model.generate_text(description)
        else:
            summary = user_provided_summary

        new_book = Book(
            id=book_id,
            title=title,
            author=author,
            genre=genre,
            year_published=year_published,
            summary=summary
        )
        self.db_session.add(new_book)
        await self.db_session.commit()
        print(f"Book '{title}' added with summary: {summary}")

    async def get_all_books(self):
        result = await self.db_session.execute(select(Book))
        books = result.scalars().all()
        return books

    async def add_review(self, review_details):
        review_id = review_details['ID']
        book_id = review_details['Book_ID']
        user_id = review_details['User_ID']
        review_text = review_details['Review_Text']
        rating = review_details['Rating']

        new_review = Review(
            id=review_id,
            book_id=book_id,
            user_id=user_id,
            review_text=review_text,
            rating=rating
        )
        self.db_session.add(new_review)
        await self.db_session.commit()
        print(f"Review added for book ID {book_id}")

    async def get_reviews_for_book(self, book_id):
        result = await self.db_session.execute(select(Review).filter_by(book_id=book_id))
        reviews = result.scalars().all()
        return reviews

    async def get_book_by_id(self, book_id):
        result = await self.db_session.execute(select(Book).filter_by(id=book_id))
        book = result.scalar_one_or_none()
        return book

    async def update_book(self, book_id, book_details):
        book = await self.get_book_by_id(book_id)
        if not book:
            return None
        
        for key, value in book_details.items():
            setattr(book, key.lower(), value)
        await self.db_session.commit()
        return book

    async def delete_book(self, book_id):
        book = await self.get_book_by_id(book_id)
        if not book:
            return None
        
        await self.db_session.delete(book)
        await self.db_session.commit()
        return True


# Main function to run the application
async def main():
    DATABASE_URL = "postgresql+asyncpg://postgres:Queen%4009876@localhost:5432/books_management"
    async_engine = create_async_engine(DATABASE_URL)
    SessionLocal = sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)
    
    # Initialize the database session
    async with SessionLocal() as session:
        # Initialize model
        model_path = "C:\\Users\\PE586UG\\OneDrive - EY\\Documents\\Gen AI\\jk\\Sheared-LLaMA-1.3B"
        llama_model = LLaMAQuick(model_path)

        # Initialize BookManager
        book_manager = BookManager(session, llama_model)

        while True:
            print("\nMenu:")
            print("1. Add New Book")
            print("2. Add New Review")
            print("3. Get All Books")
            print("4. Exit")

            choice = input("Enter your choice (1-4): ")

            if choice == '1':
                book_details = {
                    'ID': int(input("Enter Book ID: ")),
                    'Title': input("Enter Book Title: "),
                    'Author': input("Enter Author Name: "),
                    'Genre': input("Enter Genre: "),
                    'Year_Published': int(input("Enter Year Published: ")),
                    'Summary': input("Enter Summary (leave blank to generate): ")
                }
                await book_manager.add_new_book(book_details)

            elif choice == '2':
                review_details = {
                    'ID': int(input("Enter Review ID: ")),
                    'Book_ID': int(input("Enter Book ID for the review: ")),
                    'User_ID': int(input("Enter User ID: ")),
                    'Review_Text': input("Enter Review Text: "),
                    'Rating': float(input("Enter Rating (1-5): "))
                }
                await book_manager.add_review(review_details)

            elif choice == '3':
                books = await book_manager.get_all_books()
                print("Books:")
                for book in books:
                    print(f"{book.id}: {book.title} by {book.author}")

            elif choice == '4':
                print("Exiting the program.")
                break

            else:
                print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    asyncio.run(main())
