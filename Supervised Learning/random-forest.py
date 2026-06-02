# data loading
import pandas as pd
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
headers = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# define features and target
X = iris.iloc[:, :-1]
y = iris.iloc[:, -1]

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# model training
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

# model testing
y_pred = rfc.predict(X_test)

# model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred, average='weighted'))
print('Recall: ', recall_score(y_test, y_pred, average='weighted'))
print('f1 score: ', f1_score(y_test, y_pred, average='weighted'))