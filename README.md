# Supervised Learning
- a type of machine learning that trained the model using labeled dataset to predict outcomes

## K-Nearest Neighbors
`neighbors.KNeighborsClassifier`

| Classification | Regression |
| -------------- | ---------- |
| Assign the test data point to the class that appears most frequently among the k-nearest neighbors | Assign the test data point the average of the k-nearest neighbors' values |

![K-Nearest Neighbors](Model-Image/knn.png)

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

## Linear Regression
`linear_model.LinearRegression`
| Classification | Regression |
| -------------- | ---------- |
| N/A | Use a single independent variable to predict the dependent variable based on the line of regression (best-fit line) |

![Linear Regression](Model-Image/linear_reg.png)

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

 # Unsupervised Learning