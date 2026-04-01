# model/generate_data.py
# This script creates a fake fraud dataset so we don't need Kaggle.
# Real project would use real data — structure is identical.

import pandas as pd          # pandas = Python's spreadsheet library
import numpy as np           # numpy = fast math on arrays
from sklearn.datasets import make_classification
# make_classification = sklearn utility to generate fake classification data

# np.random.seed fixes the randomness so you get the same data every run
# without this, data changes each run — hard to reproduce results
np.random.seed(42)

# make_classification creates a dataset with:
X, y = make_classification(
    n_samples=100_000,    # 100,000 rows (transactions)
    n_features=28,        # 28 input features (like V1..V28 in real data)
    n_informative=15,     # 15 of those features actually predict fraud
    n_redundant=5,        # 5 are combinations of informative ones
    weights=[0.98, 0.02], # 98% legit, 2% fraud — realistic imbalance
    random_state=42       # same seed = same data every run
)
# X = feature matrix (100000 rows × 28 columns)
# y = labels (0 = legit, 1 = fraud)

# Convert numpy arrays to a pandas DataFrame (like a spreadsheet in Python)
df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
# pd.DataFrame(data, columns=names)
# f"V{i}" = f-string, creates "V1", "V2", ... "V28"

df["Amount"] = np.abs(np.random.exponential(scale=100, size=100_000))
# Add an "Amount" column — transaction amount
# exponential distribution = most transactions are small, few are large
# np.abs = make all values positive (amounts can't be negative)

df["Class"] = y
# Add the label column — 0 or 1

# Save to CSV — our training script will read this
df.to_csv("creditcard.csv", index=False)
# index=False = don't write the row numbers as a column
print(f"Dataset created: {len(df)} rows, {df['Class'].sum()} frauds")