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
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='summer', alpha=0.1)
plt.axis(lim)
plt.tight_layout()
plt.savefig('Model-Image/naive-bayes.png', dpi=300, bbox_inches='tight')

# probability predictions
yprob = model_GNB.predict_proba(Xnew)
print(yprob[-10:].round(3)) 
# display the last 10 data's probabilties of being class 0 and 1