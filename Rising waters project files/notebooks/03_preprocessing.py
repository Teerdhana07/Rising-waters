# %% [markdown]
# # 03 - Data Preprocessing
# Rising Waters: Flood Prediction System
#
# Missing value handling, outlier capping, encoding, scaling,
# and train-test splitting.

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# %%
# Load dataset
df = pd.read_csv('../dataset/flood_data.csv')
print(f"Original shape: {df.shape}\n")

# %% [markdown]
# ## 1. Handle Missing Values

# %%
print("=" * 60)
print("MISSING VALUES — Before Imputation")
print("=" * 60)
print(df.isnull().sum())
print(f"\nTotal nulls: {df.isnull().sum().sum()}")

# Fill numeric columns with median, categorical with mode
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled {col} with median: {median_val:.2f}")

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  Filled {col} with mode: {mode_val}")

print(f"\nTotal nulls after imputation: {df.isnull().sum().sum()}")

# %% [markdown]
# ## 2. Handle Outliers — IQR Capping

# %%
print("\n" + "=" * 60)
print("OUTLIER CAPPING (IQR Method)")
print("=" * 60)

# Columns to cap (exclude target and categorical)
cap_cols = [c for c in numeric_cols if c not in ['FLOODS', 'YEAR']]

for col in cap_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    before_lower = (df[col] < lower).sum()
    before_upper = (df[col] > upper).sum()

    df[col] = df[col].clip(lower=lower, upper=upper)

    if before_lower + before_upper > 0:
        print(f"  {col}: capped {before_lower} low, {before_upper} high "
              f"[{lower:.1f}, {upper:.1f}]")

print("Outlier capping complete.")

# %% [markdown]
# ## 3. Encode Categorical — LabelEncoder for SUBDIVISION

# %%
print("\n" + "=" * 60)
print("LABEL ENCODING")
print("=" * 60)

le = LabelEncoder()
df['SUBDIVISION_ENCODED'] = le.fit_transform(df['SUBDIVISION'])
print(f"SUBDIVISION encoded into {df['SUBDIVISION_ENCODED'].nunique()} classes")
print(f"Sample mapping:")
for label, encoded in zip(le.classes_[:5], le.transform(le.classes_[:5])):
    print(f"  {label} → {encoded}")
print(f"  ... ({len(le.classes_) - 5} more)")

# %% [markdown]
# ## 4. Feature / Target Split
#
# **Design Decision**: Drop `YEAR` (not predictive) and original
# `SUBDIVISION` (replaced by encoded version). Keep `SUBDIVISION_ENCODED`.

# %%
print("\n" + "=" * 60)
print("FEATURE / TARGET SPLIT")
print("=" * 60)

drop_cols = ['FLOODS', 'YEAR', 'SUBDIVISION']
X = df.drop(columns=drop_cols)
Y = df['FLOODS']

print(f"Features (X): {X.shape}")
print(f"Target   (Y): {Y.shape}")
print(f"Feature columns: {list(X.columns)}")

# %% [markdown]
# ## 5. Feature Scaling — StandardScaler

# %%
print("\n" + "=" * 60)
print("FEATURE SCALING")
print("=" * 60)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
print("StandardScaler fitted and applied.")
print(f"Mean after scaling (sample): {X_scaled.iloc[:, :3].mean().round(6).to_dict()}")
print(f"Std  after scaling (sample): {X_scaled.iloc[:, :3].std().round(6).to_dict()}")

# %% [markdown]
# ## 6. Train-Test Split

# %%
print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_train : {y_train.shape}  (flood rate: {y_train.mean():.2%})")
print(f"y_test  : {y_test.shape}  (flood rate: {y_test.mean():.2%})")

# %% [markdown]
# ## 7. Save Scaler

# %%
os.makedirs('../models', exist_ok=True)
scaler_path = '../models/scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"\nScaler saved to: {scaler_path}")
