import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from sqlalchemy import text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class BookRecommendation:
    def __init__(self, database_url):
        self.engine = create_async_engine(database_url, echo=True)
        self.session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        self.df = None
        self.model = None

    async def load_data(self):
        async with self.session() as session:
            async with session.begin():
                result = await session.execute(text("SELECT * FROM book_rating"))
                data = result.fetchall()
        
        self.df = pd.DataFrame(data, columns=result.keys())
        rating_columns = ['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5']
        for col in rating_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        self.df.dropna(subset=['Rating'] + rating_columns, inplace=True)
        self.df['AverageRating'] = self.df[rating_columns].mean(axis=1)
        self.df['Genre'] = self.df['Genre'].str.lower()

    async def train_model(self):
        if self.df is None:
            await self.load_data()

        X = self.df[['Genre']]
        y = self.df['AverageRating']

        label_encoder = LabelEncoder()
        X['GenreEncoded'] = label_encoder.fit_transform(X['Genre'])

        X_train, X_test, y_train, y_test = train_test_split(X[['GenreEncoded']], y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    async def recommend_books(self, genre, min_rating):
        if self.df is None:
            await self.load_data()

        await self.train_model()  # Ensure model is trained before making recommendations

        genre_lower = genre.lower()
        possible_genres = self.df[self.df['Genre'].str.contains(genre_lower, na=False)]

        # Check if any books are found for the specified genre
        if possible_genres.empty:
            return {"message": "No books found for the specified genre. Please try a different genre."}

        recommendations = possible_genres[possible_genres['AverageRating'] >= min_rating]

        # Check if any recommendations are found
        if recommendations.empty:
            return {"message": "No recommendations found. Consider lowering the minimum rating or trying another genre."}
        
        return recommendations[['Name']]
