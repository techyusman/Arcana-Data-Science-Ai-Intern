import numpy as np
import scipy.stats as stats

# ========================================================
# 1. Normal (Gaussian) Distribution
# ========================================================
# A continuous probability distribution characterized by a bell-shaped curve.
print("--- Normal Distribution ---")
mu = 0     # Mean
sigma = 1  # Standard Deviation

# Calculate Probability Density Function (PDF) for a specific value
# What is the density at x = 0?
x = 0
pdf_value = stats.norm.pdf(x, mu, sigma)
print("PDF at x=0 (mean): ", pdf_value)

# Calculate Cumulative Distribution Function (CDF)
# What is the probability of getting a value less than or equal to 0?
cdf_value = stats.norm.cdf(x, mu, sigma)
print("CDF at x=0 (P(X <= 0)): ", cdf_value)

# Generate random samples from a Normal Distribution
normal_samples = np.random.normal(mu, sigma, 5)
print("Random Samples: ", normal_samples)
print("\n" + "="*50 + "\n")


# ========================================================
# 2. Binomial Distribution
# ========================================================
# A discrete probability distribution of the number of successes in a sequence 
# of n independent experiments.
print("--- Binomial Distribution ---")
n = 10    # Number of trials
p = 0.5   # Probability of success on each trial (e.g., flipping a fair coin)

# Calculate Probability Mass Function (PMF)
# What is the probability of getting exactly 5 successes?
k = 5
pmf_value = stats.binom.pmf(k, n, p)
print("PMF at k=5 (Probability of exactly 5 successes): ", pmf_value)

# Calculate Cumulative Distribution Function (CDF)
# What is the probability of getting 5 or fewer successes?
cdf_value = stats.binom.cdf(k, n, p)
print("CDF at k=5 (P(X <= 5)): ", cdf_value)

# Generate random samples from a Binomial Distribution
binomial_samples = np.random.binomial(n, p, 5)
print("Random Samples (number of successes in 10 trials): ", binomial_samples)
print("\n" + "="*50 + "\n")


# ========================================================
# 3. Uniform Distribution
# ========================================================
# A continuous distribution where all outcomes are equally likely between two bounds.
print("--- Uniform Distribution ---")
low = 0
high = 10

# Calculate PDF
# For a uniform distribution, the PDF is constant between 'low' and 'high'
# Since the length of the interval is 10, the density is 1/10 = 0.1
pdf_value = stats.uniform.pdf(5, loc=low, scale=high-low)
print("PDF at x=5: ", pdf_value)

# Calculate CDF
# Probability of getting a value <= 5
cdf_value = stats.uniform.cdf(5, loc=low, scale=high-low)
print("CDF at x=5 (P(X <= 5)): ", cdf_value)

# Generate random samples
uniform_samples = np.random.uniform(low, high, 5)
print("Random Samples: ", uniform_samples)
