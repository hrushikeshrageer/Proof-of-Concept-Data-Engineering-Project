import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Example: X is your features (DataFrame or array), y is your labels.
df = pd.DataFrame(X)
df['target'] = y

# Number of samples per class you want (e.g., 50 for each class)
n_samples = 50
train_dfs = []
for label in df['target'].unique():
    train_dfs.append(df[df['target'] == label].sample(n=n_samples, random_state=42))

train_df = pd.concat(train_dfs)
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']

# To create a test set, remove these samples from the original:
remaining = df.drop(train_df.index)
X_test = remaining.drop('target', axis=1)
y_test = remaining['target']

# Now X_train and y_train have exactly equal class distributions
