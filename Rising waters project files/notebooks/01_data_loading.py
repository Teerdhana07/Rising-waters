# %% [markdown]
# # 01 - Data Loading & Inspection
# Rising Waters: Flood Prediction System
#
# This notebook loads the raw flood dataset and performs initial
# inspection to understand its structure, completeness, and distributions.

# %%
import numpy as np
import pandas as pd

# %%
# Load dataset
df = pd.read_csv('../dataset/flood_data.csv')
print("Dataset loaded successfully!\n")

# %% [markdown]
# ## Dataset Shape

# %%
print(f"Shape: {df.shape}")
print(f"  → {df.shape[0]} rows, {df.shape[1]} columns\n")

# %% [markdown]
# ## First 10 Rows

# %%
print("=" * 80)
print("HEAD (10 rows)")
print("=" * 80)
df.head(10)

# %% [markdown]
# ## Last 5 Rows

# %%
print("=" * 80)
print("TAIL (5 rows)")
print("=" * 80)
df.tail(5)

# %% [markdown]
# ## Dataset Info

# %%
print("=" * 80)
print("DATASET INFO")
print("=" * 80)
df.info()

# %% [markdown]
# ## Data Types

# %%
print("=" * 80)
print("DATA TYPES")
print("=" * 80)
print(df.dtypes)

# %% [markdown]
# ## Null Value Check

# %%
print("=" * 80)
print("NULL VALUES PER COLUMN")
print("=" * 80)
null_counts = df.isnull().sum()
print(null_counts)
print(f"\nTotal null values: {null_counts.sum()}")

# %% [markdown]
# ## Target Variable Distribution

# %%
print("=" * 80)
print("FLOODS - Value Counts")
print("=" * 80)
flood_counts = df['FLOODS'].value_counts()
print(flood_counts)
print(f"\nFlood rate: {df['FLOODS'].mean():.2%}")

# %% [markdown]
# ## Descriptive Statistics

# %%
print("=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)
df.describe()
