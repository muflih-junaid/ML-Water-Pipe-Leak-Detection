#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Load datasets
X_train = pd.read_csv("../data/processed/X_train.csv")
X_val = pd.read_csv("../data/processed/X_val.csv")

y_train = pd.read_csv("../data/processed/y_train.csv")
y_val = pd.read_csv("../data/processed/y_val.csv")

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)


X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)



X_val_scaled = pd.DataFrame(
    X_val_scaled,
    columns=X_val.columns
)
# Export preprocessed datasets
X_train_scaled.to_csv(
    "../data/processed/X_train_processed.csv", index=False
)

X_val_scaled.to_csv(
    "../data/processed/X_val_processed.csv", index=False
)

y_train.to_csv(
    "../data/processed/y_train_processed.csv", index=False
)

y_val.to_csv(
    "../data/processed/y_val_processed.csv", index=False
)

print("All preprocessed datasets exported successfully!")

