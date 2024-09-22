import pandas as pd
from sqlalchemy import create_engine
import pandas as pd

try:
    df = pd.read_csv(r'C:\Users\PE586UG\OneDrive - EY\Documents\Gen AI\jk\book_reviews_data.csv', encoding='ISO-8859-1')
except UnicodeDecodeError:
    # If 'ISO-8859-1' fails, try without specifying encoding (pandas will guess the encoding)
    df = pd.read_csv(r'C:\Users\PE586UG\OneDrive - EY\Documents\Gen AI\jk\book_reviews_data.csv', engine='python')

# Display the first few rows to inspect the data
print("Original DataFrame:")
print(df.head())

# Clean the 'Name' column and any other necessary columns
#df['Name'] = df['Name'].str.replace('Harry Potter', 'Harry Potter', regex=False)  # Adjust as needed

# Display the cleaned DataFrame
print("Cleaned DataFrame:")
print(df.head())

# Optionally, you can check for missing values
print("Missing values in each column:")
print(df.isnull().sum())


# Load the dataset
final_dataset = df
# Create a SQLAlchemy engine
DATABASE_URL = "postgresql://postgres:Queen%4009876@localhost:5432/books_management"
engine = create_engine(DATABASE_URL)

# Write the DataFrame to a new SQL table
final_dataset.to_sql('book_rating', engine, if_exists='replace', index=False)
