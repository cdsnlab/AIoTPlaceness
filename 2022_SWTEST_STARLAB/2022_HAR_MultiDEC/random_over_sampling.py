from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler

import numpy as np

X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.01, 0.05, 0.94],
                           class_sep=0.8, random_state=0)

X_idx = np.arange(y.shape[0]).reshape(-1, 1)

# print(X_idx, y)

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X, y)


ros = RandomOverSampler(random_state=0)
X_idx2, y_resampled2 = ros.fit_resample(X_idx, y)
X_resampled2 = X[X_idx2.flatten()]

print(y_resampled, y_resampled2)

print(np.sum(y_resampled != y_resampled))
print(np.sum(X_resampled != X_resampled2))

print(X_resampled, X_resampled2, y_resampled)