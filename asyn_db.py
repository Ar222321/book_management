from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql+asyncpg://postgres:password@localhost:5432/books_management"

# Create asynchronous engine
async_engine = create_async_engine(DATABASE_URL, echo=True)

# Create asynchronous session
async_session = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)
