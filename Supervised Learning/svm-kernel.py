# load data
from sklearn.datasets import load_iris
iris = load_iris()

# define feature & target
X = iris.data[:, :2]
y = iris.target

# plot svm boundaries
import numpy as np
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]

# model training
from sklearn.svm import SVC
C = 1.0
svc_classifier = SVC(kernel='linear', C=C).fit(X, y)
Z = svc_classifier.predict(X_plot)
Z = Z.reshape(xx.shape)

# data visualization
import matplotlib.pyplot as plt
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.2)
plt.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer', edgecolors='k', alpha=0.8, s=60)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM with Kernel: Classifier)', weight='bold')
plt.grid(True, alpha=0.3)
handles, _ = scatter.legend_elements()
class_labels = list(iris.target_names)
plt.legend(handles, class_labels, loc='upper right', frameon=True, facecolor='white')
plt.tight_layout()
plt.savefig('Model-Image/svm-kernel.png', dpi=300, bbox_inches='tight')

# tuning parameters
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'coef0': [0.0, 0.1, 0.5],
    'gamma': ['scale', 'auto']
}
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X, y)
print('Best parameters: ', grid_search.best_params_)
print('Best accuracy: ', grid_search.best_score_)