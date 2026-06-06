## Supervised Learning
A type of machine learning that trained the model using labeled dataset to predict outcomes.

<h3>1.  K-Nearest Neighbors<h3>

```
neighbors.KNeighborsClassifier
```

- **Classification**: Assign the test data point to the class that appears most frequently among the k-nearest neighbors
- **Regression**: Assign the test data point the average of the k-nearest neighbors' values

<p align="center">
    <img src="Model-Image/knn.png" width="450" alt="K-Nearest Neighbors">
</p>

<table>
    <thead>
        <tr>
            <th align="center" width="50%">Strength</th>
            <th align="center" width="50%">Weakness</th>
        </tr>
    </thead>
    <tbody>
            <tr>
                <td valign="top">
                    <ul>
                        <li>Simple and easy to understand.</li>
                        <li>Versatile as it can be used for classification and regression.</li>
                    </ul>
                </td>
                <td valign="top">
                    <ul>
                        <li>High memory storage required.</li>
                        <li>Does not work well on datasets with many features.</li>
                        <li>Slow prediction if N is big.</li>
                    </ul>
                </td>
            </tr>
    </tbody>
</table>


<details>
<summary><h3>Linear Regression<h3></summary>

`linear_model.LinearRegression`

| Classification | Regression |
| -------------- | ---------- |
| N/A | Use a single feature to predict the target based on the line of regression (best-fit line) |

<p align="center">
    <img src="Model-Image/linear-reg.png" width="450" alt="Linear Regression">
</p>

<table>
    <thead>
            <tr>
                <th align="center" width="50%">Strength</th>
                <th align="center" width="50%">Weakness</th>
            </tr>
    </thead>
    <tbody>
        <tr>
            <td valign="top">
                <ul>
                    <li>Fast to train and predict.</li>
                    <li>Easy to understand using formulas.</li>
                </ul>
            </td>
            <td valign="top">
                <ul>
                    <li>Coeeficient might be hard to interpret if the dataset has highly correlated features.</li>
                    <li>Does not work well on samll datasets.</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>
</details>


<details>
<summary><h3>Naive Bayes<h3></summary>

`naive_bayes.GaussianNB` (features are continuous variables)
`naive_bayes.BernoulliNB` (features are discrete counts)
`naive_bayes.MultinomialNB` features that are binary)

| Classification | Regression |
| -------------- | ---------- |
| Calculates the probability of a sample belonging to a particular class based on the probabilities of its features | N/A |

<p align="center">
    <img src="Model-Image/naive-bayes.png" width="450" alt="Naive Bayes">
</p>

<table>
    <thead>
        <tr>
            <th align="center" width="50%">Strength</th>
            <th align="center" width="50%">Weakness</th>
        </tr>
    </thead>
  <tbody>
        <tr>
            <td valign="top">
                <ul>
                    <li>Highly scalable.</li>
                    <li>Reuqire less training data.</li>
                    <li>Can handle continuous, discrete and binary data.</li>
                </ul>
            </td>
            <td valign="top">
                <ul>
                    <li>Strong feature independence, as it is almost impossible to have a set of features which are completely independent of each other.</li>
                    <li>If the category of a data has zero frequency, NB cannot make prediction.</li>
                </ul>
            </td>
        </tr>
  </tbody>
</table>
</details>


<details>
<summary><h3>Decision Trees<h3></summary>

`tree.DecisionTreeClassifier`

| Classification | Regression |
| -------------- | ---------- |
| Classify data into different classes based on the values of input features | Predict continuous values based on the values of input features |

<p align="center">
    <img src="Model-Image/decision-tree.png" width="450" alt="Decision Tree">
</p>

<table>
    <thead>
        <tr>
            <th align="center" width="50%">Strength</th>
            <th align="center" width="50%">Weakness</th>
        </tr>
    </thead>
  <tbody>
        <tr>
            <td valign="top">
                <ul>
                    <li>Easily visualized and understood.</li>
                    <li>No preprocessing like normalization or standardization.</li>
                    <li>Input feature can be a mix of different data types.</li>
                </ul>
            </td>
            <td valign="top">
                <ul>
                    <li>Tend to overfit and provide poor generalization.</li>
                </ul>
            </td>
        </tr>
  </tbody>
</table>
</details>


<details>
<summary><h3>Random Forest<h3></summary>

`ensemble.RandomForestClassifier`

| Classification | Regression |
| -------------- | ---------- |
| Mode of decision trees' prediction | Mean of decision trees' prediction |

<p align="center">
    <img src="Model-Image/random-forest.png" width="450" alt="Random Forest">
</p>

<table>
    <thead>
        <tr>
            <th align="center" width="50%">Strength</th>
            <th align="center" width="50%">Weakness</th>
        </tr>
    </thead>
  <tbody>
        <tr>
            <td valign="top">
                <ul>
                    <li>Robustness to overfitting.</li>
                    <li>High accuracy.</li>
                    <li>RF can provide information about the importance of each feature in the model.</li>
                </ul>
            </td>
            <td valign="top">
                <ul>
                    <li>More time and resources consuming.</li>
                    <li>More complex and less intuitive.</li>
                </ul>
            </td>
        </tr>
  </tbody>
</table>
</details>


<details>
<summary><h3>Support Vector Machine<h3></summary>

`svm.SVC`

| Classification | Regression |
| -------------- | ---------- |
| Find the hyperplane with maximum margin that separates the data points into different classes. | Similar to classification.  |

<p align="center">
    <img src="Model-Image/svm.png" width="450" alt="SVM">
    <img src="Model-Image/svm-kernel.png" width="450" alt="SVM-kernel">
</p>

<table>
    <thead>
        <tr>
            <th align="center" width="50%">Strength</th>
            <th align="center" width="50%">Weakness</th>
        </tr>
    </thead>
  <tbody>
        <tr>
            <td valign="top">
                <ul>
                    <li>Works well with high dimensional space (with kernel).</li>
                    <li>Requires very less memory.</li>
                </ul>
            </td>
            <td valign="top">
                <ul>
                    <li>Time consuming and no suitable with large datasets.</li>
                    <li>Do not work well with overlapping classes.</li>
                </ul>
            </td>
        </tr>
  </tbody>
</table>
</details>

## Unsupervised Learning