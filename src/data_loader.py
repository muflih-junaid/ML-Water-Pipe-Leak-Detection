
import pandas as pd
from pathlib import Path

# 1. Define paths
DATA_DIR = Path("../data/raw/2018")
OUTPUT_DIR = Path("../data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load datasets
pressure = pd.read_csv(DATA_DIR / "2018_SCADA_Pressures.csv", sep=";")
flow = pd.read_csv(DATA_DIR / "2018_SCADA_Flows.csv", sep=";")
demand = pd.read_csv(DATA_DIR / "2018_SCADA_Demands.csv", sep=";")
leakage = pd.read_csv(DATA_DIR / "2018_Leakages.csv", sep=";")
level = pd.read_csv(DATA_DIR / "2018_SCADA_Levels.csv", sep=";")


# 3. Convert timestamps and sensor values
def clean_dataset(data):

    data = data.copy()

    data["Timestamp"] = pd.to_datetime(
        data["Timestamp"],
        errors="coerce"
    )

    sensor_columns = data.columns.drop("Timestamp")

    data[sensor_columns] = data[sensor_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    return data


pressure = clean_dataset(pressure)
flow = clean_dataset(flow)
demand = clean_dataset(demand)
leakage = clean_dataset(leakage)
level = clean_dataset(level)


# 4. Check duplicate timestamps
datasets = {
    "pressure": pressure,
    "flow": flow,
    "demand": demand,
    "leakage": leakage,
    "level": level
}

for name, data in datasets.items():

    print(f"\n{name.upper()}")
    print("Rows:", len(data))
    print("Duplicate timestamps:",
          data["Timestamp"].duplicated().sum())
    print("Missing timestamps:",
          data["Timestamp"].isna().sum())


# 5. Rename sensor columns
def rename_sensors(data, prefix):

    return data.rename(
        columns=lambda col:
        f"{prefix}_{col}" if col != "Timestamp" else col
    )


pressure = rename_sensors(pressure, "pressure")
flow = rename_sensors(flow, "flow")
demand = rename_sensors(demand, "demand")
level = rename_sensors(level, "level")


# 6. Merge SCADA measurements
combined = (
    pressure
    .merge(flow, on="Timestamp", how="inner",
           validate="one_to_one")
    .merge(demand, on="Timestamp", how="inner",
           validate="one_to_one")
    .merge(level, on="Timestamp", how="inner",
           validate="one_to_one")
)


# 7. Calculate total leakage
pipe_columns = leakage.columns.drop("Timestamp")

leakage["Total_Leakage"] = leakage[pipe_columns].sum(
    axis=1,
    min_count=1
)

leakage_target = leakage[
    ["Timestamp", "Total_Leakage"]
]


# 8. Merge target with measurements
df = combined.merge(
    leakage_target,
    on="Timestamp",
    how="inner",
    validate="one_to_one"
)


# 9. Final dataset validation
print("\nFINAL DATASET")
print("Shape:", df.shape)
print("Duplicate timestamps:",
      df["Timestamp"].duplicated().sum())
print("Missing values:", df.isna().sum().sum())

print(df.head())


# 10. Save processed dataset
df.to_csv(
    OUTPUT_DIR / "2018_combined.csv",
    index=False
)

print("\nDataset successfully saved!")
