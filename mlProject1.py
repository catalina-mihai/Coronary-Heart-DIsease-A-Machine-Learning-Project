import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, show
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd
import lmoments3

# Ex1
###############################################################################

df = pd.read_csv("https://hastie.su.domains/ElemStatLearn/datasets/SAheart.data")
attributeNames = df.columns.tolist()

# Extract values from the desired column and range of rows
column_index = 7  # Column index (0-based)
start_row = 0  # Starting row index (0-based)
end_row = 462  # Ending row index (exclusive)

# In pandas, we can directly slice the DataFrame
classLabels = df.iloc[start_row:end_row, column_index]

chd_labels = df['chd']

# For classification
# Define a custom function to categorize the values
def categorize(x):    
    if x < 18.5:
        return 1  # 'underweight'
    elif 18.5 <= x < 25:
        return 2  # 'normal weight'
    elif 25 <= x < 30:
        return 3  # 'overweight'
    else:
        return 4  # 'obese'

# Apply the custom function to create a new column
df['BMI.categories'] = classLabels.apply(categorize)

attributeNames = df.columns.tolist()

columns = df.columns.tolist()
columns.insert(-1, columns.pop(columns.index('BMI.categories')))
df = df[columns]

# Convert 'famhist' categorical attribute to numeric
famhist_class = [df.famhist.unique()[1], df.famhist.unique()[0]]
classDict = dict(zip(famhist_class, range(len(famhist_class))))
df['famhist'] = df['famhist'].map(classDict)
chd_class = [df.chd.unique()[1], df.chd.unique()[0]]
classDict = dict(zip(chd_class, range(len(chd_class))))
df['chd'] = df['chd'].map(classDict)
np.std(df.famhist)

# Like Stone, we change chd to be the response variable: y=1 having chd, y=0 not having chd
y = df.chd
className = np.array(['chd0', 'chd1'])

# For the PCA drop chd, famhist, row.names
columns_to_drop = ['row.names', 'BMI.categories', 'chd', 'famhist']
df = df.drop(columns=columns_to_drop)

# Update attributeNames
attributeNames = df.columns.tolist()

# Select only the numeric columns
numeric_df = df.select_dtypes(include=np.number)

# Convert DataFrame to NumPy array
X = np.array(numeric_df)

N, M = X.shape

# Ex 2 PCA
###############################################################################

# Standardize the data
Y1 = X - np.ones((N, 1)) * X.mean(0)
Y2 = Y1 / np.std(Y1, axis=0)
Ys = [Y1, Y2]
titles = ['Zero-mean', 'Zero-mean and unit variance']
threshold = 0.9
i, j = 0, 1  # Choose two PCs to plot

# Create the plot
plt.figure(figsize=(15, 20))
plt.subplots_adjust(hspace=.4)
plt.suptitle('Heart Disease: Effect of Standardization', fontsize=16)
nrows, ncols = 3, 2
for k in range(2):  # Loop over Y1 and Y2
    # Obtain the PCA solution by calculating the SVD
    U, S, Vh = svd(Ys[k], full_matrices=False)
    V = Vh.T
    
    # Flip the directionality of the principal directions for Y2
    if k == 1:
        V = -V
        U = -U
    
    # Compute variance explained
    rho = (S**2) / np.sum(S**2)
    
    # Compute the projection onto the principal components
    Z = U * S
    
    # Plot projection
    plt.subplot(nrows, ncols, 1 + k)
    for c in [0, 1]:
        plt.plot(Z[y == c, i], Z[y == c, j], '.', alpha=.5, label=f'Class {className[c]}')
    plt.xlabel('PC' + str(i + 1))
    plt.ylabel('PC' + str(j + 1))
    plt.title(titles[k] + '\n' + 'Projection')
    plt.legend()
    plt.axis('equal')
    
    # Plot attribute coefficients in principal component space
    plt.subplot(nrows, ncols, 3 + k)
    for att in range(V.shape[1]):
        plt.arrow(0, 0, V[att, i], V[att, j], color='r', alpha=0.5)
        plt.text(V[att, i], V[att, j], attributeNames[att], color='r')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('PC' + str(i + 1))
    plt.ylabel('PC' + str(j + 1))
    plt.grid()
    
    # Add a unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'b--')
    plt.title(titles[k] + '\n' + 'Attribute coefficients')
    plt.axis('equal')
    
    # Plot cumulative variance explained
    plt.subplot(nrows, ncols, 5 + k)
    plt.plot(range(1, len(rho) + 1), rho, 'x-', label='Individual')
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-', label='Cumulative')
    plt.axhline(y=threshold, color='k', linestyle='--', label='Threshold')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.title(titles[k] + '\n' + 'Variance explained')
    plt.legend()
    plt.grid()

plt.show()

# Standardize the data (mean = 0, variance = 1) FROM ex2.1.3.py
Y = X - np.ones((N,1))*X.mean(0)
Y = Y / np.std(Y, axis=0)

# PCA using Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(Y, full_matrices=False)
V = Vt.T

# Calculate the variance explained by each principal component
rho = (S**2) / np.sum(S**2)

# Project the centered data onto principal component space
Z = Y @ V

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components')
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

# Plot the PCA component coefficients
pcs = [0,1,2, 3]
legendStrs = ['PC'+str(e+1) for e in pcs]
bw = .2
r = np.arange(1,M+1)
plt.figure(figsize=(10, 7))
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Heart Disease: PCA Component Coefficients')
plt.show()


#visualize pca


# Standardize the data (mean = 0, variance = 1) FROM ex2.1.3.py
Y = X - np.ones((N,1))*X.mean(0)
Y = Y / np.std(Y, axis=0)

# PCA using Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(Y, full_matrices=False)
V = Vt.T

# Calculate the variance explained by each principal component
rho = (S**2) / np.sum(S**2)

# Project the centered data onto principal component space
Z = Y @ V

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()


# From ex2_1_5.py try plotting the
pcs = [0,1,2, 3]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Heart Disease: PCA Component Coefficients')
plt.show()

# Ex 3 Summary statistics
###############################################################################


# Summary statistics: mean, mode, IQR, sd, skewness, (L-moments), dependence (cov, cor)

# Calculate mean
mu = np.mean(X, axis=0)

# Calculate mode
# Find the mode of each attributeprint("Mode of each attribute:")
for col in numeric_df.columns:
    mode = stats.mode(numeric_df[col])
    print(f"{col}: {mode}")
    print(f"{col} Data:")
    print(numeric_df[col])
    
# min and max    
for col in numeric_df.columns:
    max_value = numeric_df[col].max()
    min_value = numeric_df[col].min()
    print(f"{col}: Max = {max_value}, Min = {min_value}")
    print(f"{col} Data:")
    print(numeric_df[col])
    
    

    
# IQR
print("Interquartile range (IQR) of each attribute:")
iqrs = []
q25s = []
q75s = []
q0s = []
q50s = []
q100s = []
for col in numeric_df.columns:
    q1 = np.percentile(X[:, numeric_df.columns.get_loc(col)], 25)
    q3 = np.percentile(X[:, numeric_df.columns.get_loc(col)], 75)
    iqr = q3 - q1
    print(f"IQR of {col}: {iqr}")
    iqrs.append(iqr)
    q25s.append(q1)
    q75s.append(q3)
    q0 = np.percentile(X[:, numeric_df.columns.get_loc(col)], 0)
    q2 = np.percentile(X[:, numeric_df.columns.get_loc(col)], 50)
    q4 = np.percentile(X[:, numeric_df.columns.get_loc(col)], 100)
    q0s.append(q0)
    q50s.append(q2)
    q100s.append(q4)
    
# variance
print("Variance (var) of each attribute:")
variances = []

for col in range(M):
    v = np.var(X[:, col])
    variances.append(v)
    print(f"variances of {numeric_df.columns[col]}: {v}")

# sd
print("Standard Deviation (sd) of each attribute:")
standard_deviations = []

for col in range(M):
    s = np.std(X[:, col])
    standard_deviations.append(s)
    print(f"sd of {numeric_df.columns[col]}: {s}")



# skewness s (optional summary statisitcs)
print("Skewness of each attribute:")
for col in numeric_df.columns:
    s = stats.skew(X[:, numeric_df.columns.get_loc(col)])
    print(f"skewness of {col}: {s}")


# (L-moments) (what is the interpretation again??)

print("L-moments of each attribute:")
for col in numeric_df.columns:
    l = lmoments3.lmom_ratios(X[:, numeric_df.columns.get_loc(col)])
    print(f"L-moments of {col}: {l}")


# Compute covariance matrix
Sigma = np.cov(X, rowvar=False)

# Print covariance matrix
print("Covariance matrix:")
print(Sigma)

# Calculate correlation matrix
covmatrix = np.corrcoef(X, rowvar=False)

# Print correlation matrix
print("Correlation matrix:")
print(covmatrix)


# Ex 4: Probability densities and data visualization with PYTHON
