# predictor_script.py
import joblib
import pandas as pd

def predict_movie_success(input_data):
    print("🎥 Loading model...")
    model = joblib.load("movie_hit_flop_model.pkl")

    print("🔍 Preparing input data...")
    df = pd.DataFrame([input_data])

    # Ensure consistent feature columns
    expected_features = [
        'year', 'genre', 'director', 'production_size', 'language', 'country',
        'budget', 'marketing', 'runtime', 'release_month', 'sequel', 'franchise',
        'director_popularity', 'cast_popularity', 'critic_score', 'audience_score',
        'screen_count', 'opening_weekend', 'total_gross'
    ]
    missing = [col for col in expected_features if col not in df.columns]
    if missing:
        print(f"⚠️ Missing columns: {missing}. Filling with 0.")
        for col in missing:
            df[col] = 0

    # Reorder columns to match training
    df = df[expected_features]

    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][prediction]
    result = "🎬 HIT" if prediction == 1 else "💤 FLOP"
    print(f"\nPrediction: {result}")
    print(f"Probability: {prob:.2f}")
    return result, prob

if __name__ == "__main__":
    new_movie = {
        "year": 2025,
        "genre": 2,
        "director": 5,
        "production_size": 1,
        "language": 3,
        "country": 1,
        "budget": 80,
        "marketing": 30,
        "runtime": 140,
        "release_month": 6,
        "sequel": 0,
        "franchise": 1,
        "director_popularity": 8.5,
        "cast_popularity": 9.1,
        "critic_score": 7.9,
        "audience_score": 8.3,
        "screen_count": 2500,
        "opening_weekend": 20,
        "total_gross": 150
    }

    predict_movie_success(new_movie)