"""
Week 1 warm-up: vectors, matrices, NumPy basics.

Run it with:
    python3 week1_basics/01_vectors_matrices.py

Fill in every `...` below, then run the file again. Each print() tells you
what the expected shape/value looks like so you can check your own work.

This connects directly to the real project: by Phase 3, each day in the
dataset becomes one row (a "feature vector") like the toy example at the
bottom, and stacking many days together makes a matrix -- exactly what
you're practicing here.
"""

import numpy as np

# --- 1. A vector is just a 1D array of numbers -----------------------------
# Make a vector of the numbers 1 through 5.
v = ...  # e.g. np.array([1, 2, 3, 4, 5])
print("vector v:", v)
print("shape:", ...)  # v.shape -- should print (5,)

# --- 2. Basic vector arithmetic is element-wise -----------------------------
# Add 10 to every element of v, without writing a loop.
v_plus_10 = ...
print("v + 10:", v_plus_10)

# --- 3. A matrix is a 2D array: rows x columns ------------------------------
# Build this 3x3 matrix:
# [[1, 2, 3],
#  [4, 5, 6],
#  [7, 8, 9]]
M = ...
print("matrix M:\n", M)
print("shape:", ...)  # should be (3, 3)

# --- 4. Matrix-vector multiplication (the core operation behind neural
# networks -- see roadmap page "z = w1x1 + w2x2 + ... + b") --------------
w = np.array([1, 0, -1])
z = ...  # M @ w
print("M @ w:", z)

# --- 5. Standardization: z = (x - mean) / std -------------------------------
# From roadmap section 6 -- this is how you'd normalize a real feature
# column (e.g. daily SPY returns) before feeding it to a model.
returns = np.array([0.01, -0.02, 0.015, 0.0, -0.005, 0.02])
standardized = ...  # (returns - returns.mean()) / returns.std()
print("standardized returns:", standardized)

# --- 6. A toy daily feature vector X_t (preview of Phase 3) -----------------
# Roadmap's example feature vector for one day: SPY 1-day return, realized
# vol, Fed-cut probability, and the 10Y-2Y yield spread. Build it as one
# NumPy vector, then stack 3 fictional days into a matrix.
day1 = np.array([0.004, 0.012, 0.62, -0.35])
day2 = np.array([-0.002, 0.014, 0.65, -0.30])
day3 = np.array([0.001, 0.011, 0.60, -0.33])

X = ...  # np.stack([day1, day2, day3]) -- shape should be (3, 4): 3 days, 4 features
print("feature matrix X:\n", X)
print("shape:", ...)  # (3, 4)
