# load dataset
import pandas as pd
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(path, names=headernames)

# define feature and target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)

# features data scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# model training
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)

# model testing
y_pred = classifier.predict(X_test)

# model performance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:\n", accuracy_score(y_test, y_pred))

# use PCA to reduce to 2d for visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# encode labels to numeric values for visualization
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# model training
knn_neighbors = KNeighborsClassifier(n_neighbors=3)
knn_neighbors.fit(X_train_pca, y_train_encoded)

# model testing (select 1 random test data)
test_point_index = 0
test_point = X_test_pca[test_point_index:test_point_index+1]
true_label_test_point = y_test[test_point_index]
distances, neighbor_indices = knn_neighbors.kneighbors(test_point)
nearest_neighbors = X_train_pca[neighbor_indices[0]]

# data visualization
import matplotlib.pyplot as plt
scatter_train = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_encoded, cmap='summer', edgecolors='k', s=60, alpha=0.8)
scatter_test = plt.scatter(test_point[:, 0], test_point[:, 1], c='red', marker='X', s=120, zorder=5)
for i in range(len(nearest_neighbors)):
    plt.plot([test_point[0, 0], nearest_neighbors[i, 0]],
             [test_point[0, 1], nearest_neighbors[i, 1]],
             '--', color='gray', linewidth=1)

handles, labels = scatter_train.legend_elements()
class_names = list(label_encoder.classes_)
plt.title('KNN (N=3): Iris Species Prediction')
plt.grid(True, alpha=0.3)
plt.legend(
    handles + [scatter_test], 
    class_names + [f'Test Point (True: {true_label_test_point})'],
    loc='upper right'
)

plt.tight_layout()
plt.savefig('Model-Image/knn.png', dpi=300, bbox_inches='tight')