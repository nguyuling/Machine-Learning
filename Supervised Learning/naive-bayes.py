# create synthetic data
from sklearn.datasets import make_blobs
X, y = make_blobs(300, 2, centers=2, random_state=2, cluster_std=1.5)
# generate 300 samples with 2 features, split into 2 clusters with fixed randomness

# data visualization
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
# plt.show()

# model training
from sklearn.naive_bayes import GaussianNB
model_GNB = GaussianNB()
model_GNB.fit(X, y)

# model testing
import numpy as np
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model_GNB.predict(Xnew)

# data visualization
plt.figure(figsize=(10, 8))
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='#228B22', marker='o', s=60, edgecolors='k', alpha=0.8, label='Train: Class 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='#FFD700', marker='o', s=60, edgecolors='k', alpha=0.8, label='Train: Class 1')
lim = plt.axis()
plt.scatter(Xnew[ynew == 0, 0], Xnew[ynew == 0, 1], c='#228B22', marker='.', s=60, alpha=0.2, label='Test: Class 0')
plt.scatter(Xnew[ynew == 1, 0], Xnew[ynew == 1, 1], c='#FFD700', marker='.', s=60, alpha=0.2, label='Test: Class 1')

plt.axis(lim)
plt.title('Gaussian NB: Training Clusters vs Predicted Test Grid')
plt.grid(True, alpha=0.3)

leg = plt.legend(loc='upper left', frameon=True)
for lh in leg.legend_handles: 
    lh.set_alpha(1.0)

plt.tight_layout()
plt.savefig('Model-Image/naive-bayes.png', dpi=300, bbox_inches='tight')

# probability predictions
yprob = model_GNB.predict_proba(Xnew)
print(yprob[-10:].round(3)) 
# display the last 10 data's probabilties of being class 0 and 1