import pandas as pd
import numpy as np
from scipy import stats
import lmoments3
from matplotlib import pyplot as plt

# Read the dataset
df = pd.read_csv("https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data")
attributeNames = df.columns.tolist()

# Select only the numeric columns
numeric_df = df.select_dtypes(include=np.number)

# Convert DataFrame to NumPy array
X = np.array(numeric_df)
N, M = X.shape

# Summary statistics: mean, mode, IQR, sd, skewness, (L-moments), dependence (cov, cor)

# Calculate mean
mu = np.mean(X, axis=0)

# Find the mode of each attribute
print("Mode of each attribute:")
for col in numeric_df.columns:
    mode = stats.mode(numeric_df[col])
    print(f"{col}: {mode}")
    print(f"{col} Data:")
    print(numeric_df[col])
    
# IQR
print("Interquartile range (IQR) of each attribute:")
for col in numeric_df.columns:
    q1 = np.percentile(X[:, numeric_df.columns.get_loc(col)], 25)
    q3 = np.percentile(X[:, numeric_df.columns.get_loc(col)], 75)
    iqr = q3 - q1
    print(f"IQR of {col}: {iqr}")

# sd
print("Standard Deviation (sd) of each attribute:")
for col in numeric_df.columns:
    s = np.std(X[:, numeric_df.columns.get_loc(col)])
    print(f"sd of {col}: {s}")


# skewness s
print("Standard Deviation (sd) of each attribute:")
for col in numeric_df.columns:
    s = stats.skew(X[:, numeric_df.columns.get_loc(col)])
    print(f"skewness of {col}: {s}")


# (L-moments)
lmoments = lmoments3.lmom_ratios(X)
print(lmoments)

# Compute covariance matrix
Sigma = np.cov(X, rowvar=False)

# Print covariance matrix
#print("Covariance matrix:")
#print(Sigma)

# Calculate correlation matrix
rho = np.corrcoef(X, rowvar=False)

# Print correlation matrix
#print("Correlation matrix:")
#print(rho)

plt.plot(X)
