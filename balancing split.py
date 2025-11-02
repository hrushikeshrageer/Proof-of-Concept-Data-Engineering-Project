import numpy as np

class0_idx = np.where(y == 0)[0]
class1_idx = np.where(y == 1)[0]

np.random.seed(42)
train_idx = np.concatenate([
    np.random.choice(class0_idx, 50, replace=False),
    np.random.choice(class1_idx, 50, replace=False)
])

X_train = X[train_idx]
y_train = y[train_idx]

test_idx = np.setdiff1d(np.arange(len(y)), train_idx)
X_test = X[test_idx]
y_test = y[test_idx]
