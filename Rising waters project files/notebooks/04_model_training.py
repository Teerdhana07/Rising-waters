# %% [markdown]
# # 04 - Model Training & Evaluation
# Rising Waters: Flood Prediction System
#
# Self-contained script: loads raw data → preprocesses → trains 4 models
# → evaluates → saves best model + scaler.

# %%
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# %%
# ======================================================================
# STEP 1: Load Dataset
# ======================================================================
df = pd.read_csv('../dataset/flood_data.csv')
print(f"Dataset loaded: {df.shape}\n")

# %%
# ======================================================================
# STEP 2: Preprocessing (inline — mirrors 03_preprocessing.py)
# ======================================================================

# --- 2a. Missing values ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# --- 2b. Outlier capping (IQR) ---
cap_cols = [c for c in numeric_cols if c not in ['FLOODS', 'YEAR']]
for col in cap_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

# --- 2c. Feature / target split ---
# Use ONLY the 6 features that the Flask web app collects from users.
# This ensures the saved model & scaler match app.py's input array.
FEATURE_COLS = ['ANNUAL_RAINFALL', 'CLOUD_COVERAGE', 'JUN-SEP', 'MAR-MAY', 'OCT-DEC', 'JAN-FEB']
X = df[FEATURE_COLS]
Y = df['FLOODS']

# --- 2e. Feature scaling ---
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# --- 2f. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

print("Preprocessing complete.")
print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"  y_train: {y_train.shape} (flood rate: {y_train.mean():.2%})")
print(f"  y_test : {y_test.shape} (flood rate: {y_test.mean():.2%})\n")

# %%
# ======================================================================
# STEP 3: Define Models
# ======================================================================
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    ),
}

# %%
# ======================================================================
# STEP 4: Train & Evaluate Each Model
# ======================================================================
results = {}

for name, model in models.items():
    print("=" * 70)
    print(f"  MODEL: {name}")
    print("=" * 70)

    # Train
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    results[name] = {'accuracy': acc, 'model': model}

    print(f"\n  Accuracy: {acc:.2f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{cr}")
    print()

# %%
# ======================================================================
# STEP 5: Model Comparison
# ======================================================================
print("=" * 70)
print("  MODEL COMPARISON")
print("=" * 70)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()]
}).sort_values('Accuracy', ascending=False).reset_index(drop=True)

print(comparison_df.to_string(index=False))

# %%
# --- Bar chart ---
os.makedirs('../documentation', exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2ecc71' if i == 0 else '#3498db'
          for i in range(len(comparison_df))]
bars = ax.bar(comparison_df['Model'], comparison_df['Accuracy'], color=colors,
              edgecolor='black', linewidth=0.8)

# Add value labels on bars
for bar, acc in zip(bars, comparison_df['Accuracy']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{acc:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylim(0, 1.05)
ax.set_ylabel('Accuracy', fontsize=13)
ax.set_title('Model Accuracy Comparison — Flood Prediction',
             fontsize=15, fontweight='bold')
ax.set_xlabel('Model', fontsize=13)

plt.tight_layout()
plt.savefig('../documentation/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_comparison.png")

# %%
# ======================================================================
# STEP 6: Save Best Model & Scaler
# ======================================================================
os.makedirs('../models', exist_ok=True)

best_name = comparison_df.iloc[0]['Model']
best_acc = comparison_df.iloc[0]['Accuracy']
best_model = results[best_name]['model']

model_path = '../models/floods.save'
scaler_path = '../models/scaler.pkl'

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)

print(f"\n{'=' * 70}")
print(f"  BEST MODEL: {best_name} (Accuracy: {best_acc:.2f})")
print(f"{'=' * 70}")
print(f"  [OK] Model saved  -> {model_path}")
print(f"  [OK] Scaler saved -> {scaler_path}")
print(f"{'=' * 70}")
