import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

class BookRecommendation:
    def __init__(self, database_url):
        # Create a SQLAlchemy engine
        self.engine = create_engine(database_url)
        self.df = self.load_data()
        self.model = self.train_model()

    def load_data(self):
        # Load data from the book_rating table
        df = pd.read_sql_table('book_rating', self.engine)

        # Convert rating distributions to numeric
        rating_columns = ['RatingDist1', 'RatingDist2', 'RatingDist3', 'RatingDist4', 'RatingDist5']
        for col in rating_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN values
        df.dropna(subset=['Rating'] + rating_columns, inplace=True)

        # Combine rating distributions to create an average rating
        df['AverageRating'] = df[rating_columns].mean(axis=1)

        # Encode the genre
        df['Genre'] = df['Genre'].str.lower()  # Normalize genres
        return df

    def train_model(self):
        # Prepare data for training
        X = self.df[['Genre']]
        y = self.df['AverageRating']

        # Encode the genre
        label_encoder = LabelEncoder()
        X['GenreEncoded'] = label_encoder.fit_transform(X['Genre'])

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X[['GenreEncoded']], y, test_size=0.2, random_state=42)

        # Initialize and fit the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        return model

    def recommend_books(self, genre, min_rating):
        genre_lower = genre.lower()  # Normalize genre input
        possible_genres = self.df[self.df['Genre'].str.contains(genre_lower, na=False)]
        recommendations = possible_genres[possible_genres['AverageRating'] >= min_rating]
        return recommendations[['Name']]

# Example usage
if __name__ == "__main__":
    DATABASE_URL = "postgresql://postgres:Queen%4009876@localhost:5432/books_management"
    recommender = BookRecommendation(DATABASE_URL)

    user_genre = input("Enter the genre you are interested in: ")
    user_rating = input("Enter the minimum average rating you are looking for: ")

    try:
        user_rating = float(user_rating)
    except ValueError:
        print("Invalid rating input. Please enter a number.")
        exit()

    recommended_books = recommender.recommend_books(user_genre, user_rating)

    if recommended_books.empty:
        print("No recommendations found for the specified genre or rating.")
    else:
        print("Recommended books:")
        print(recommended_books)
