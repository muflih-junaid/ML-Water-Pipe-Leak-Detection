import pandas as pd
from pathlib import Path

DATA_DIR = Path("../data/raw/2018")

pressure = pd.read_csv(DATA_DIR / "2018_SCADA_Pressures.csv")
flow = pd.read_csv(DATA_DIR / "2018_SCADA_Flows.csv")
demand = pd.read_csv(DATA_DIR / "2018_SCADA_Demands.csv")
leakage = pd.read_csv(DATA_DIR / "2018_Leakages.csv")
level = pd.read_csv(DATA_DIR / "2018_SCADA_Levels.csv")

datasets = {
    "pressure": pressure,
    "flow": flow,
    "demand": demand,
    "leakage": leakage,
    "level" : level
}

# In[16]:


pressure = pressure.rename(
    columns=lambda col: f"pressure_{col}" if col != "Timestamp" else col
)

flow = flow.rename(
    columns=lambda col: f"flow_{col}" if col != "Timestamp" else col
)

demand = demand.rename(
    columns=lambda col: f"demand_{col}" if col != "Timestamp" else col
)

level = level.rename(
    columns=lambda col: f"level_{col}" if col != "Timestamp" else col
)

# Merge all measurements
combined = (
    pressure
    .merge(flow, on="Timestamp", how="inner", validate="one_to_one")
    .merge(demand, on="Timestamp", how="inner", validate="one_to_one")
    .merge(level, on="Timestamp", how="inner", validate="one_to_one")
)



# In[17]:


duplicate_rows = combined[combined.duplicated()]
print(duplicate_rows)


# In[18]:


pipe_columns = leakage.columns.drop("Timestamp")

leakage["Total_Leakage"] = leakage[pipe_columns].sum(
    axis=1,
    min_count=1
)
leakage_target = leakage[["Timestamp", "Total_Leakage"]]



# In[19]:


df = (
    combined
    .merge(leakage_target, on="Timestamp", how="inner", validate="one_to_one")
)

# In[20]:


output = Path("../data/processed")
output.mkdir(parents=True, exist_ok=True)

df.to_csv(
    output / "2018_combined.csv",
    index=False
)

