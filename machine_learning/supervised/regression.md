### Regression Tasks

For when the target variable is continuous.

#### Algorithms

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=42)

# create regressor
reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

# score the model using R^2
reg.score(x_test, y_test)

# calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
```

- **linear regression** in 2D fits a line to a data set
  - has form `y = ax + b`
      - target y
      - feature x
      - a, b are parameters of the model we want to learn
  - can choose parameters by finding the ones that minimize error
      - error calculated by **error/cost/loss function**

- **linear regression** in > 2D
  - has form `y = a1x1 + a2x2 + ... + b` for (x1, x2, ...) in features
  - aka chooses coefficient for each feature variable
    - can determine very large coefficients == overfitting
    - use **regularization** to penalize large coefficients

- error function could be distance between observations and the fitted line
    - vertical distance between point and line aka **residual**

- **ordinary least squares (OLS)**: minimize sum of squares of residuals
    - if just summing residuals, large negative residual could cancel out large positive residual, so use squared residual (z ** 2 >= 0)

- **root mean squared error (RMSE)**: square root of the average of squared errors

#### Measuring Model Performance

- **r-squared aka coefficient of determination**: square of the sample **correlation coefficient (r)** between the observed outcomes and the observed predictor values
    - the **correlation coefficient r** is often **Pearson's r**
    - if the predicted values are the same as observed values, correlation is high and approaches +1

- when measuring performance, value of chosen statistic may be closely tied to the train/test split i.e. pecularities in a specific test data
    - to compensate, can use **k-fold cross validation (CV)**
        - split data into k segments aka **folds**
        - for fold i in 0..k
          - use fold i as test data, and rest for training
          - calculate performance statistic
        - then more broadly look at k performance statistic measurements and consider e.g. their mean, confidence interval, etc.
        - cross-validating your model allows you to more confidently evaluate its predictions
        - because cross validating means training and predicting multiple times, it is more computationally expensive

      ```python
      from sklearn.model_selection import cross_val_score
      from sklearn.linear_model import LinearRegression

      reg = LinearRegression()
      # get 5-fold CV results; defaults to R^2
      cv_results = cross_val_score(reg, X, y, cv=5)
      ```

#### Regularization

- **linear regression** in > 2D
  - has form `y = a1x1 + a2x2 + ... + b` for (x1, x2, ...) in features
  - aka chooses coefficient for each feature variable
    - can determine very large coefficients == overfitting
    - use **regularization** to penalize large coefficients

- **ridge regression**: `OLS + alpha * sum(a.i**2 for i in 0..n)` where n == # features
    - large coefficients (`a.i`) thus add to the loss function and are penalized
    - regularization term `a.i**2` is the **L2 norm** of the coefficients
    - value for alpha must be chosen and tuned
    - use `sklearn.linear_model.Ridge` for regressor class e.g. `reg = Ridge(alpha=0.1, normalize=True)`

- **lasso regression/L1 regularization**: `OLS + alpha * sum(abs(a.i) for i in 0..n)` where n == # features
    - use `sklearn.linear_model.Lasso` for regressor class e.g. `reg = Lasso(alpha=0.1)`
    - regularization term `abs(a.i)` is the **L1 norm** of the coefficients
    - using lasso regression error function tends to result in coefficients where less important features have coefficient values closer to 0
        - makes lasso convenient for feature selection (identify those features whose coefficients are farthest from 0)

      ```python
      from sklearn.linear_model import Lasso

      lasso = Lasso(alpha=0.4, normalize=True)
      lasso.fit(X, y)

      # get lasso regressor coefficient values
      lasso_coef = lasso.coef_

      # plot feature names along x-axis, their coefficient values along y
      plt.plot(range(len(df_columns)), lasso_coef)
      plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
      plt.margins(0.02)
      plt.show()
      ```

- **elastic penalty**: `a * L1 + b * L2`
  - use `sklearn.linear_model.ElasticNet` for regressor class e.g. `reg = ElasticNet(l1_ratio=0.5, l2_ratio=0.5)`

