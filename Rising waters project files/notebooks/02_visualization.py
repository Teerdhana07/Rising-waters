# %% [markdown]
# # 02 - Exploratory Data Analysis & Visualization
# Rising Waters: Flood Prediction System
#
# Univariate, multivariate, and correlation analysis of flood data.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# %%
# Load dataset
df = pd.read_csv('../dataset/flood_data.csv')
print(f"Dataset loaded: {df.shape}")

# Set seaborn style
sns.set_style('darkgrid')

# Create documentation folder for saving plots
os.makedirs('../documentation', exist_ok=True)

# %% [markdown]
# ## 1. Univariate Analysis — Distribution Plots

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Key Features', fontsize=16, fontweight='bold')

sns.histplot(df['ANNUAL_RAINFALL'], kde=True, bins=30, color='steelblue', ax=axes[0, 0])
axes[0, 0].set_title('Annual Rainfall Distribution')
axes[0, 0].set_xlabel('Annual Rainfall (mm)')

sns.histplot(df['CLOUD_COVERAGE'], kde=True, bins=30, color='coral', ax=axes[0, 1])
axes[0, 1].set_title('Cloud Coverage Distribution')
axes[0, 1].set_xlabel('Cloud Coverage (%)')

sns.histplot(df['JUN-SEP'], kde=True, bins=30, color='seagreen', ax=axes[1, 0])
axes[1, 0].set_title('Monsoon Rainfall (JUN-SEP) Distribution')
axes[1, 0].set_xlabel('JUN-SEP Rainfall (mm)')

sns.histplot(df['MAR-MAY'], kde=True, bins=30, color='goldenrod', ax=axes[1, 1])
axes[1, 1].set_title('Pre-Monsoon Rainfall (MAR-MAY) Distribution')
axes[1, 1].set_xlabel('MAR-MAY Rainfall (mm)')

plt.tight_layout()
plt.savefig('../documentation/univariate_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: univariate_distributions.png")

# %% [markdown]
# ## 2. Multivariate Analysis — Pairplot

# %%
key_features = ['ANNUAL_RAINFALL', 'JUN-SEP', 'MAR-MAY', 'CLOUD_COVERAGE', 'FLOODS']
pairplot_df = df[key_features].copy()
pairplot_df['FLOODS'] = pairplot_df['FLOODS'].astype(str)

g = sns.pairplot(pairplot_df, hue='FLOODS', palette={
    '0': 'steelblue', '1': 'crimson'
}, diag_kind='kde', plot_kws={'alpha': 0.5, 's': 20})
g.figure.suptitle('Pairplot of Key Features by Flood Status', y=1.02,
                   fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('../documentation/pairplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: pairplot.png")

# %% [markdown]
# ## 3. Scatter Plot — Annual Rainfall vs Cloud Coverage

# %%
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    df['ANNUAL_RAINFALL'], df['CLOUD_COVERAGE'],
    c=df['FLOODS'], cmap='coolwarm', alpha=0.6, edgecolors='k', linewidths=0.3, s=30
)
ax.set_xlabel('Annual Rainfall (mm)', fontsize=12)
ax.set_ylabel('Cloud Coverage (%)', fontsize=12)
ax.set_title('Annual Rainfall vs Cloud Coverage (colored by FLOODS)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('FLOODS')

plt.tight_layout()
plt.savefig('../documentation/scatter_rainfall_cloud.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: scatter_rainfall_cloud.png")

# %% [markdown]
# ## 4. Box Plots — Rainfall Features Grouped by FLOODS

# %%
rainfall_features = ['ANNUAL_RAINFALL', 'JUN-SEP', 'MAR-MAY', 'OCT-DEC', 'JAN-FEB']

fig, axes = plt.subplots(1, len(rainfall_features), figsize=(22, 6))
fig.suptitle('Rainfall Features by Flood Status', fontsize=16, fontweight='bold')

for i, feat in enumerate(rainfall_features):
    sns.boxplot(x='FLOODS', y=feat, data=df, ax=axes[i],
                palette={0: 'steelblue', 1: 'crimson'})
    axes[i].set_title(feat)
    axes[i].set_xlabel('FLOODS')

plt.tight_layout()
plt.savefig('../documentation/boxplots_by_floods.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: boxplots_by_floods.png")

# %% [markdown]
# ## 5. Correlation Heatmap

# %%
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])

fig, ax = plt.subplots(figsize=(16, 12))
corr = numeric_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.5, ax=ax, annot_kws={'size': 8})
ax.set_title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('../documentation/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: correlation_heatmap.png")

# %% [markdown]
# ## 6. Descriptive Statistics by Flood Status

# %%
print("=" * 80)
print("DESCRIPTIVE STATISTICS — Grouped by FLOODS")
print("=" * 80)

grouped_desc = df.groupby('FLOODS').describe().T
print(grouped_desc.to_string())
