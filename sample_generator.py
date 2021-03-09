import numpy as np
from numpy.random import multivariate_normal
import sys
import matplotlib.pyplot as plt

# USAGE: python sample_generator <n_dim> <n_samples>
# Generates samples from multivariate normal distribution using in-built function from numpy and saves to text file
# Requires the mean (1-D array of length N) and covariance matrix of the distribution (must be symmetric and positive-semidefinite for proper sampling).
n_dim = int(sys.argv[1])
n_samples = int(sys.argv[2])

# Mean
mean_val = 0
mean = mean_val * np.ones(n_dim) # Simple example where all dimensions have same mean of 0

# Covariance
cov_val = 1
cov = np.zeros((n_dim, n_dim))
np.fill_diagonal(cov, cov_val) # Diagonal covariance (has non-negative elements only on the diagonal)

# Generate samples
np.random.seed(0)
if n_dim == 2: # Can use plot to illustrate the process if 2 dimensional
    x, y = multivariate_normal(mean, cov, n_samples).T # transpose index and columns of data frame
    plt.plot(x, y, 'x')
    plt.axis('equal')   
    plt.title('2-D Multivariate Normal samples using mean of 0 and covariance of 1')
    plt.show()
else:
    y = multivariate_normal(mean, cov, n_samples)

# Save samples to file
file_name = 'samples_' + str(n_dim) + '.txt'
np.savetxt(file_name, y)