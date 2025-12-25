# data_script.py
import pandas as pd

def load_movie_data():
    """Load and return the movie dataset as a DataFrame."""
    df = pd.read_csv("movie_data_2.csv")
    print("✅ Dataset Loaded Successfully!")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df

if __name__ == "__main__":
    data = load_movie_data()
    print(data.head())