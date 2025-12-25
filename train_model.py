import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

np.random.seed(42)

print("🎬 Generating balanced movie dataset...")

# Generate 500 FLOPs and 500 HITs
n_samples = 500

# FLOP movies
flops = pd.DataFrame({
    'year': np.random.randint(2015, 2025, n_samples),
    'genre': np.random.randint(0, 11, n_samples),
    'director': np.random.randint(0, 11, n_samples),
    'production_size': np.random.randint(0, 3, n_samples),
    'language': np.random.randint(0, 11, n_samples),
    'country': np.random.randint(0, 11, n_samples),
    'budget': np.random.uniform(5, 50, n_samples),
    'marketing': np.random.uniform(2, 20, n_samples),
    'runtime': np.random.uniform(80, 130, n_samples),
    'release_month': np.random.randint(1, 13, n_samples),
    'sequel': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    'franchise': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    'director_popularity': np.random.uniform(2.0, 6.5, n_samples),
    'cast_popularity': np.random.uniform(2.5, 6.8, n_samples),
    'critic_score': np.random.uniform(3.0, 6.5, n_samples),
    'audience_score': np.random.uniform(3.5, 6.8, n_samples),
    'screen_count': np.random.randint(200, 1500, n_samples),
    'hit_or_flop': 0
})

# HIT movies
hits = pd.DataFrame({
    'year': np.random.randint(2015, 2025, n_samples),
    'genre': np.random.randint(0, 11, n_samples),
    'director': np.random.randint(0, 11, n_samples),
    'production_size': np.random.randint(2, 6, n_samples),
    'language': np.random.randint(0, 11, n_samples),
    'country': np.random.randint(0, 11, n_samples),
    'budget': np.random.uniform(50, 250, n_samples),
    'marketing': np.random.uniform(25, 150, n_samples),
    'runtime': np.random.uniform(110, 180, n_samples),
    'release_month': np.random.randint(1, 13, n_samples),
    'sequel': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
    'franchise': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    'director_popularity': np.random.uniform(6.5, 10.0, n_samples),
    'cast_popularity': np.random.uniform(7.0, 10.0, n_samples),
    'critic_score': np.random.uniform(6.5, 9.5, n_samples),
    'audience_score': np.random.uniform(7.0, 9.8, n_samples),
    'screen_count': np.random.randint(2000, 6000, n_samples),
    'hit_or_flop': 1
})

# Combine and shuffle
df = pd.concat([flops, hits], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✓ Dataset: {len(df)} movies")
print(f"  FLOPs: {sum(df['hit_or_flop']==0)}, HITs: {sum(df['hit_or_flop']==1)}")

# Train model
features = ["year", "genre", "director", "production_size", "language",
            "country", "budget", "marketing", "runtime", "release_month",
            "sequel", "franchise", "director_popularity", "cast_popularity",
            "critic_score", "audience_score", "screen_count"]

X = df[features]
y = df['hit_or_flop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

print(f"\n✓ Model trained!")
print(f"  Training accuracy: {model.score(X_train, y_train):.1%}")
print(f"  Test accuracy: {model.score(X_test, y_test):.1%}")

# Test predictions
test_pred = model.predict(X_test)
print(f"\n✓ Test predictions: {sum(test_pred==0)} FLOPs, {sum(test_pred==1)} HITs")

# Save
joblib.dump(model, "movie_hit_flop_model.pkl")
print("\n✅ Model saved: movie_hit_flop_model.pkl")
print("Now run: streamlit run app.py")