import pandas as pd

# Load your training data - CHANGE THIS to your actual file name
df = pd.read_csv("your_training_data.csv")

# Check the target column distribution
print("Class distribution:")
print(df['hit_or_flop'].value_counts())  # Change 'hit_or_flop' to your actual column name
print("\n0 = FLOP, 1 = HIT")
print(f"\nTotal rows: {len(df)}")
print(f"\nColumn names:")
print(df.columns.tolist())