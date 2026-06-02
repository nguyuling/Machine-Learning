
# Data Preprocessing
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler


# Generate Data
import sklearn.datasets
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris


# Model Selection
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# Supervised Learning
import sklearn.linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

import sklearn.naive_bayes
from sklearn.naive_bayes import GaussianNB

import sklearn.tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier 

import sklearn.svm
from sklearn.svm import SVM


# Unseupervised Learning
import sklearn.cluster
from sklearn.cluster import KMeans


# Model Evaluation
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report