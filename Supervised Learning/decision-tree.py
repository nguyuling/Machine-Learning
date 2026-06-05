# data loading
from sklearn.datasets import load_iris
iris = load_iris()

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)
 
# data training
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

# data prediction
y_pred = dtc.predict(X_test)

# model accuracy 
import numpy as np
accuracy = np.sum(y_pred == y_test) / len(y_test)
print('Accuray: ', accuracy)

# data visualization
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
# plt.figure(figsize=(20, 13))
plot_tree(dtc, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.savefig('Model-Image/decision-tree.png', dpi=300, bbox_inches='tight')