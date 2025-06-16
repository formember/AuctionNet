import pandas as pd
import seaborn as sns
from scipy.stats import norm
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
data = pd.read_csv("./data/splited_data/period-7_0.csv")
data = data[data["advertiserNumber"]==0]
x = data["pValue"].values
y = data["leastWinningCost"].values
# x = x[x < 0.0004]  # Filter out values greater than 0.001
# x = x[x > 0.0]  # Filter out values less than 0.0001

# # Fit x to a mixture of two normal distributions

# x = x.reshape(-1, 1)  # Reshape for GaussianMixture
# gmm = GaussianMixture(n_components=1, random_state=0, covariance_type='full',max_iter=200000, tol=1e-6)
# gmm.fit(x)

# # Extract parameters
# means = gmm.means_.flatten()
# print(f"Mean: {means[0]}")
# stds = np.sqrt(gmm.covariances_).flatten()
# weights = gmm.weights_

# # Generate the range for plotting
# x_range = np.linspace(x.min(), x.max(), 1000)

# # Calculate the normal distribution
# pdf = weights[0] * norm.pdf(x_range, means[0], stds[0])

# # Plot the original data and the fitted distribution
# plt.figure(figsize=(10, 6))
# sns.histplot(x.flatten(), bins=200, kde=False, stat="density", color="gray", label="Histogram of x", alpha=0.5)
# plt.plot(x_range, pdf, label=f"Fitted Distribution (mean={means[0]:.2e}, std={stds[0]:.2e})", color="blue")
# plt.title("Fitting x to a Single Normal Distribution")
# plt.xlabel("x")
# plt.ylabel("Density")
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.scatter(x, y, alpha=0.5)
# plt.title("Scatter Plot of pValue vs leastWinningCost")
# plt.xlabel("pValue")
# plt.ylabel("leastWinningCost")
# plt.grid(True)
# plt.show()
import matplotlib.pyplot as plt

t = y / x
t = t[t < 500]  # Filter out values greater than 1000
# Plot the probability density function for pValue
plt.figure(figsize=(15, 5))

# First subplot: CPA
plt.subplot(1, 3, 1)
sns.kdeplot(t, shade=True, color="blue", label="CPA")
plt.title("Probability Density Function of CPA")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)

# Second subplot: pValue
plt.subplot(1, 3, 2)
sns.kdeplot(x, shade=True, color="red", label="pValue")
plt.title("Probability Density Function of pValue")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)

# Third subplot: leastWinningCost
plt.subplot(1, 3, 3)
sns.kdeplot(y, shade=True, color="green", label="leastWinningCost")
plt.title("Probability Density Function of leastWinningCost")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)

plt.title("Probability Density Functions of pValue and leastWinningCost")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
