from pathlib import Path	
import pandas as pd

df= pd.read_csv("../data/processed/2018_combined.csv")

df = df.sort_values("Timestamp").reset_index(drop=True)

# 2. Split chronologically
train = df[df["Timestamp"] < "2018-10-01"].copy()

val = df[
    (df["Timestamp"] >= "2018-10-01")].copy()

# 3. Separate features and target
target = "Total_Leakage"  # Use your actual column name

feature_cols = [
    c for c in df.columns
    if c not in ["Timestamp", target]
]

X_train = train[feature_cols]
y_train = train[target]

X_val = val[feature_cols]
y_val = val[target]
# 4. Verify dimensions
for name, X, y in [
    ("Train", X_train, y_train),
    ("Validation", X_val, y_val),
]:
    print(f"{name}: X={X.shape}, y={y.shape}")

# 5. Save datasets

# Define output directory and ensure it exists
output_dir = Path("../data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Save files
X_train.to_csv(output_dir / "X_train.csv", index=False)
y_train.to_csv(output_dir / "y_train.csv", index=False)
X_val.to_csv(output_dir / "X_val.csv", index=False)
y_val.to_csv(output_dir / "y_val.csv", index=False)
