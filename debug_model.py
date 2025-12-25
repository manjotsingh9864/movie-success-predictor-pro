import joblib
import pandas as pd
import numpy as np

# Load the model
model = joblib.load("movie_hit_flop_model.pkl")

print("=" * 60)
print("MODEL DIAGNOSTICS")
print("=" * 60)

# 1. Check model type
print(f"\n1. Model Type: {type(model).__name__}")

# 2. Check if it's a pipeline or raw model
if hasattr(model, 'named_steps'):
    print("   → Model is a Pipeline")
    print(f"   → Steps: {list(model.named_steps.keys())}")
else:
    print("   → Model is a standalone classifier")

# 3. Check feature names (if available)
try:
    if hasattr(model, 'feature_names_in_'):
        print(f"\n2. Expected Features ({len(model.feature_names_in_)}):")
        for i, feat in enumerate(model.feature_names_in_, 1):
            print(f"   {i}. {feat}")
    elif hasattr(model, 'named_steps') and hasattr(model[:-1], 'get_feature_names_out'):
        features = model[:-1].get_feature_names_out()
        print(f"\n2. Expected Features ({len(features)}):")
        for i, feat in enumerate(features, 1):
            print(f"   {i}. {feat}")
    else:
        print("\n2. Feature names not available in model")
        if hasattr(model, 'n_features_in_'):
            print(f"   → Model expects {model.n_features_in_} features")
except Exception as e:
    print(f"\n2. Error getting features: {e}")

# 4. Check classes
if hasattr(model, 'classes_'):
    print(f"\n3. Model Classes: {model.classes_}")
    print(f"   → 0 = FLOP, 1 = HIT")

# 5. Test with sample data
print("\n" + "=" * 60)
print("TESTING WITH SAMPLE DATA")
print("=" * 60)

# Create test data matching your app
test_cases = [
    {
        "name": "Low Budget Indie",
        "data": {
            "year": 2025, "genre": 2, "director": 3,
            "production_size": 0, "language": 3, "country": 1,
            "budget": 5, "marketing": 2, "runtime": 90,
            "release_month": 6, "sequel": 0, "franchise": 0,
            "director_popularity": 4.0, "cast_popularity": 5.0,
            "critic_score": 6.0, "audience_score": 6.5,
            "screen_count": 500, "opening_weekend": 1,
            "total_gross": 10
        }
    },
    {
        "name": "Big Budget Blockbuster",
        "data": {
            "year": 2025, "genre": 0, "director": 9,
            "production_size": 5, "language": 3, "country": 1,
            "budget": 200, "marketing": 100, "runtime": 150,
            "release_month": 5, "sequel": 1, "franchise": 1,
            "director_popularity": 9.5, "cast_popularity": 9.8,
            "critic_score": 8.5, "audience_score": 9.0,
            "screen_count": 5000, "opening_weekend": 150,
            "total_gross": 800
        }
    },
    {
        "name": "Medium Budget Drama",
        "data": {
            "year": 2025, "genre": 2, "director": 7,
            "production_size": 2, "language": 3, "country": 1,
            "budget": 50, "marketing": 20, "runtime": 120,
            "release_month": 9, "sequel": 0, "franchise": 0,
            "director_popularity": 7.5, "cast_popularity": 7.0,
            "critic_score": 8.0, "audience_score": 7.5,
            "screen_count": 2000, "opening_weekend": 25,
            "total_gross": 150
        }
    }
]

for test_case in test_cases:
    print(f"\n{test_case['name']}:")
    df = pd.DataFrame([test_case['data']])
    
    print(f"  Input shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    try:
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        
        print(f"  → Prediction: {'HIT' if prediction == 1 else 'FLOP'}")
        print(f"  → Probability [FLOP, HIT]: [{proba[0]:.3f}, {proba[1]:.3f}]")
        print(f"  → Confidence: {max(proba):.1%}")
    except Exception as e:
        print(f"  → ERROR: {e}")

# 6. Check for common issues
print("\n" + "=" * 60)
print("DIAGNOSTIC CHECKS")
print("=" * 60)

# Check model parameters
if hasattr(model, 'get_params'):
    params = model.get_params()
    print(f"\n4. Model Parameters:")
    for key, value in list(params.items())[:10]:  # Show first 10
        print(f"   {key}: {value}")

# Check if model is fitted
if hasattr(model, 'n_features_in_'):
    print(f"\n5. Model is fitted: ✓")
    print(f"   Number of features: {model.n_features_in_}")
else:
    print(f"\n5. Model fit status: Unknown")

# 7. Recommendations
print("\n" + "=" * 60)
print("POSSIBLE ISSUES & SOLUTIONS")
print("=" * 60)
print("""
If all predictions are the same:

1. IMBALANCED TRAINING DATA
   → Your training data might have mostly FLOPs
   → Solution: Balance your dataset or use class_weight='balanced'

2. FEATURE MISMATCH
   → Column names or order don't match training data
   → Solution: Check feature names above match your training

3. POOR MODEL TRAINING
   → Model didn't learn meaningful patterns
   → Solution: Retrain with more diverse data

4. SCALING ISSUES
   → Features need to be scaled the same way as training
   → Solution: Ensure preprocessing matches training

5. DATA LEAKAGE IN TRAINING
   → Model might have learned a shortcut
   → Solution: Remove features like 'total_gross' before training

6. WRONG MODEL FILE
   → The .pkl file might be from an early/bad training run
   → Solution: Retrain and save a new model

NEXT STEPS:
- Check the output above for feature names
- Verify your training data had both HITs and FLOPs
- Try retraining the model with balanced data
- Remove target-leaking features like 'total_gross' from training
""")