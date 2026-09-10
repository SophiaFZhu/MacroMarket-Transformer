"""
Week 1 warm-up: vectors, matrices, NumPy basics.

Run it with:
    python3 week1_basics/01_vectors_matrices.py
"""

import numpy as np

# --- 1. A vector is just a 1D array of numbers -----------------------------
v = np.array([1, 2, 3, 4, 5])
print("vector v:", v)
print("shape:", v.shape)  # (5,)

# --- 2. Basic vector arithmetic is element-wise -----------------------------
v_plus_10 = v + 10
print("v + 10:", v_plus_10)

# --- 3. A matrix is a 2D array: rows x columns ------------------------------
M = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])
print("matrix M:\n", M)
print("shape:", M.shape)  # (3, 3)

# --- 4. Matrix-vector multiplication (the core operation behind neural
# networks -- see roadmap page "z = w1x1 + w2x2 + ... + b") --------------
w = np.array([1, 0, -1])
z = M @ w
print("M @ w:", z)

# --- 5. Standardization: z = (x - mean) / std -------------------------------
returns = np.array([0.01, -0.02, 0.015, 0.0, -0.005, 0.02])
standardized = (returns - returns.mean()) / returns.std()
print("standardized returns:", standardized)

# --- 6. A toy daily feature vector X_t (preview of Phase 3) -----------------
day1 = np.array([0.004, 0.012, 0.62, -0.35])
day2 = np.array([-0.002, 0.014, 0.65, -0.30])
day3 = np.array([0.001, 0.011, 0.60, -0.33])

X = np.stack([day1, day2, day3])
print("feature matrix X:\n", X)
print("shape:", X.shape)  # (3, 4)
