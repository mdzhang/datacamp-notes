# Machine Learning

- **machine learning**: ability of computers to learn and make decisions from data w/o being explicitly programmed

- **unsupervised learning**: uses unlabeled data
    - **clustering**: group data into categories, where categories not known beforehand
    - **reinforcement learning**: software agent acts within environment, actions are rewarded or punished, agent uses these responses to optimize behavior

## Supervised Learning

- **supervised learning**: uses labeled data
    - has **predictor variables/features/independent variables** and a **target/response/dependent variable**
    - **labeled data** has known output i.e. target variable value is known
    - goal is to build a model that can predict target variable given predictor variables
        - to automate time consuming or expensive manual task
        - to make predictions about the future
    - if target variable is continuous, it's a **regression task**
    - if target variable is discrete, it's a **classification task**
        - binary decisions have discrete set of values i.e. true or false

- **training data**: already labeled data classifier uses to learn
- **test data**: already labeled data used to measure model/classifier performance
- **hold-out set**: data used to evaluate how a model performs on unseen data when it's fitted using cross validation

- running a model on a dataset aka "training a model on", "fitting a model to", "fitting a classifier to" data
- split data into **training data** and **test data**
    - don't reuse training data when measuring model performance - model has already seen it, not indicative of ability to generalize
    - split test/training data so that both reflect larger data set, each reflects distribution of original data set

- **overfitting**
    - sensitive to noise in the specific labeled data you have rather than reflecting general trends in the data

- **imputting data**: make an educated guess as to what the missing values in a dataset could be
    - e.g. replace with mean of non-missing values

### Classification Tasks

For when the target variable is discrete.

#### Algorithms

- **k-Nearest neighbors (k-NN)**: find k=3 nearest neighbors, use whatever category majority of neighboring observations fall into to label new observation
    - can visualize **decision boundary** which highlights areas of graph and indicates how, were x values to fall there, they would be classified
    - higher values for k = smoother decision boundary = less complex model
        - too high and can lead to **underfitting**
    - lower values for k = ?? decision boundary = more complex model = can lead to **overfitting**

```python
from sklearn.neighbors import KNeighborsClassifier

# use .values to convert to numpy arrays
# target (y) and feature (x) arrays must have same length
# feature (x) matrix of dimensions i x j has i rows and j columns
#    where there is a row for each observation
#    and j == # different features under consideration
y = df['party'].values
x = df.drop('party', axis=1).values

knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(x, y)

y_pred = knn.predict(x_new)
```

- **logistic regression (logreg)**
  - outputs probability `p`; if `p < 0.5`, label `1` aka `true`; else if `p > 0.5`, label `0` aka `false`
  - use `sklearn.linear_model.LogisticRegression` for regressor class e.g. `reg = LogisticRegression()`

- **receiver operating characteristic curve (ROC)**: curve resulting from graphing different `p` thresholds against false positive rate aka **fallout** (x-axis) and true positive rate aka **recall** (y-axis)
    ```python
    from sklearn.metrics import roc_curve

    # get probability of sample being in a particular class (0 or 1)
    # predict_proba() returns matrix with row for each observation
    #   that has 2 values - probability of being 0, probability of being 1
    y_pred_prob = logreg.predict_proba(X_test)[:,1]

    # get false positive rate, true positive rate, p thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    ```

    - better models have larger **area under [ROC] curve (AUC)** since these means true positive rate approaches 1, false positive rate approaches 0
    - ???

    ```python
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score

    # AUC
    y_pred_prob = logreg.predict_proba(X_test)[:,1]
    roc_auc_score(y_test, y_pred_prob)

    # AUC with cross validation
    cv_results = cross_val_score(reg, X_test, y_test, cv=5, scoring='roc_auc')
    ```

#### Measuring Model Performance

- measure model performance by running it on a dataset and calculating a performance metric
- **accuracy**: total # correct predictions / total # data points
    - can graph model complexity against accuracy for a **model complexity curve**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=21, stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

# see how classifier would fare on test data
# print accuracy as a fraction
knn.score(x_test, y_test)
```

- **accuracy** not always the best measure of model performance
    - e.g. when a **class imbalance** exists e.g. 99% email not spam, 1% spam
    - **confusion matrix**:

        ||Predicted: spam|Predicted: not spam|
        |Actual: spam| True positive | False negative |
        |Actual: not spam| False positive | True negative |
    - **accuracy** can also be defined as `(tp + tn) / (tp + tn + fp + fn)`
        - proportion of correct predictions
    - **precision aka Positive Predicted Value (PPV)**: `tp / (tp + fp)`
        - high precision == low false positive rate
        - not many emails wrongly predicted as spam
    - **sensitivity/hit rate/true positive rate/recall**: `tp / (tp + fn)`
        - high recall == predicted most spam emails correctly
    - **F1 score**: `2 * (precision * recall) / (precision + recall)`
        - harmonic mean of precision and recall

```python
from sklearn.metrics import classification_report, confusion_matrix

# same as above
...

confusion_matrix(y_test, y_pred)

# returns all matrics defined above (accuracy, precision, recall, F1-score)
classification_report(y_test, y_pred)
```

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

## Hyperparameter Tuning

- models, algorithms below often need additional parameter values (see k in k-NN, coefficients in linear regression, alpha in ridge/lasso regression) that cannot be learned by fitting the model aka they must be chosen
    - aka **hyperparameters**
    - **hyperparameter tuning**: selecting best value for hyperparameters to get optimal model performance
      - select different value
      - fit all of them to your model separately
      - measure performance
          - use **cross-validation** to avoid overfitting parameters to test/train split
      - select most performant values

- **grid search cross-validation (cv)**: creates a grid where x-axis has values for one parameter, y-axis for another
    - perform cross-validation for each possible combination and store value in grid
        - computationally expensive!
    - select highest value in grid aka parameter values w/ best performance

    ```python
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier()

    # can specify L1/L2 regularization with penalty entry
    param_grid = {'n_neighbors': np.arange(1, 50), 'penalty': ['l1', 'l2']}

    # create grid
    knn_cv = GridSearchCV(knn, param_grid, cv=5)
    # fit to data, perform grid search
    knn_cv.fit(x, y)

    knn_cv.best_params_
    knn_cv.best_score_
    ```

- in `sklearn.Pipeline`:

    ```python
    steps = [('scaler', StandardScaler()),
            ('imputation', imp),
            ('SVM', clf)]
    pipeline = Pipeline(steps)

    # pass hypertuning params using <stage name>__<param name> as key
    param_grid = {'knn__n_neighbors': np.arange(1, 50)}

    # create grid
    cv = GridSearchCV(pipeline, param_grid, cv=5)
    cv.fit(X_train, y_train)
    cv.predict(X_test)
    ```

## Data preprocessing

- imputing data

```python
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# create an imputer that will replace missing values ('NaN') with
# the most frequently occuring value in that column (axis=0)
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# create a classifier
clf = SVC()

# create pipeline
# last element must have a classifier, all other entries must be transformers
steps = [('imputation', imp),
        ('SVM', clf)]
pipeline = Pipeline(steps)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# predict the labels of the test set
y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
```

- **normalizing data** aka **scaling and centering** brings all data values to be on a similar scale to avoid one unduly influencing a model
    - multiple normalization methods e.g.
      - convert to [0, 1] range by subtracting the minimum and dividing by the range
      - convert to [-1, 1] range
      - data should be centered around 0 and have variance one by subtracting the mean and dividing by the variance aka **standardization**

      ```python
      from sklearn.preprocessing import scale

      X_scale = scale(X)
      ```

    - can also use in pipeline

      ```python
      from sklearn.preprocessing import StandardScaler
      from sklearn.pipeline import Pipeline

      steps = [('scaler', StandardScaler()),
              ('imputation', imp),
              ('SVM', clf)]
      pipeline = Pipeline(steps)
      ```
