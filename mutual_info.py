import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Load the data from the file
# Assuming the file is named 'data.txt'
data = np.loadtxt('output.txt')

# Separate features and target
X = data[:, :-1]  # First 10 columns are features (time series)
y = data[:, -1]   # Last column is the target

# Reshape X to treat the 10 floats as a single feature vector for each sample
# Here we don't need to reshape because we are considering it as single feature vector already.

# Compute mutual information
mi = mutual_info_classif(X, y.reshape(-1, 1), discrete_features=False)

# Display the mutual information value
print(f"Mutual Information: {mi[0]}")

# # Alternatively, you can store the result in a pandas DataFrame for better visualization
# mi_df = pd.DataFrame({'Feature': ['Time Series Feature'], 'Mutual Information': mi})
# print(mi_df)
