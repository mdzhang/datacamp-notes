## Classification

- **classification tasks** are tasks that attempt to predict a target variable with discrete possible values

### Classifiers

##### kNN

- **k-Nearest neighbors (k-NN)**: find k=3 nearest neighbors, use whatever category majority of neighboring observations fall into to label new observation
    - can visualize **decision boundary** which highlights areas of graph and indicates how, were x values to fall there, they would be classified
      - **decision region**: region in feature space where all instances assigned to 1 class label
        - separated by surfaces known as **decision boundaries**
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

#### Linear Classifiers

- linear models learn **decision boundaries** which separate predicted classes
  - can be a straight line in each plane dividing areas on either side of lines into different decision areas
  - can be nonlinear, non-contiguous regions (e.g. like a contour map)
- **linear classifiers** are classifiers that learn linear decision boundaries
- **linearly separable**: dataset can be perfectly explained by linear classifier; is no linear boundary that perfectly classifies all points

- **dot product**: multiple vectors element-wise and sum
- underlying function of form:
  - `output = coefficient * features + intercept` (where `*` is dot product)
  - predict 1 class if output is positive, another if it is negative
  - changing coefficients can change orientation of decision boundary
      - along decision boundary, model outputs 0
  - changing intercept can shift orientation boundary


##### Linear regression

- **linear regression (linreg)**
  - in `sklearn`, minimizes sum of squared error i.e. sum of squared error is the **loss function**
      - need to choose coefficients that minimize loss
      - done by `linreg.fit` function

##### Logistic regression

- **logistic regression (logreg)**
  - outputs probability `p`; if `p < 0.5`, label `1` aka `true`; else if `p > 0.5`, label `0` aka `false`
  - use `sklearn.linear_model.LogisticRegression` for regressor class e.g. `lr = LogisticRegression()`
  - `lr.predict(X)` outputs a vector of `0` or `1`s
  - `lr.coef_ @ X[10] + lr.intercept_` to get prediction for 10th value in X
  - can't use e.g. sum of squared error as loss function b/c we're using classification, not continuous numeric values
  - can use e.g. `0-1 loss` i.e. # of errors (count 0 for correct, 1 for an error)
  - hard to minimize/optimize `0-1 loss`
  - can use **log loss** instead

      ```python
      def log_loss(raw_model_output):
         return np.log(1+np.exp(-raw_model_output))
      ```

  - in `sklearn`, `C` is the inverse of regularization strength (smaller C => more regularization)
      - regularization combats model overfitting, by making coefficients smaller
          - regularized loss = original loss + large coefficient penalty
          - if you were missing a value for a feature, would not throw off the model as much; regularization like a compromise between not using it at all and fully using it
          - prevent overfitting by leaning on a single feature too heavily
      - more regularization => worse training data accuracy, but (almost always) better test data accuracy
      - more regularization => smaller coefficients => output closer to 0 => probabilities closer to 0.5 after piping through sigmoid function => lower confidence
  - penalty of `L2` (Ridge) loss by default, but can set e.g. `LogisticRegression(penalty='l1')`

  - use grid search CV to get best C

      ```python
      lr = LogisticRegression(penalty='l1')

      searcher = GridSearchCV(lr, {'C':[0.001, 0.01, 0.1, 1, 10]})
      searcher.fit(X_train, y_train)

      print("Best CV params", searcher.best_params_)
      print("Best CV accuracy", searcher.best_score_)

      best_lr = searcher.best_estimator_
      coefs = best_lr.coef_
      print("Total number of features:", coefs.size)
      print("Number of selected features:", np.count_nonzero(coefs))
      ```

##### Multiclass Logistic Regression

- **multi class logistic regression**
    - **one vs rest strategy**: train a binary classifier for each class (1 for is class, 0 for is not a specific class)
      - performs logistic regression on each column separately, separate classifier per column
      - predict probabilities for each binary classifier, take largest output

    ```python
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression

    clf = OneVsRestClassifier(LogisticRegression())
    ```

    - **multinomial/softmax strategy**: fit a single classifier for all classes
        - loss more directly aligned with accuracy, better accuracy than one vs rest
        - use

    ```python
    from sklearn.linear_model import LogisticRegression

    # multinomial regression
    # solver specifies the loss function
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    ```

##### Linear SVC

- **linear svc**
  - use `sklearn.svm.LinearSVC`
  - use **hinge loss**

      ```python
      def hinge_loss(raw_model_output):
         return np.maximum(0,1-raw_model_output)
      ```
  - by convention, use L2 loss
  - **support vector model (SVM)**:
      - if graphing raw model output against loss
        - support vector includes training examples not in flat part of loss diagram
        - flat part of loss diagram is where correct predictions are (correct within threshold)
      - example that is incorrectly classified or close to the boundary
      - if example not in support vector, has no effect on model if removed
      - acceptable "closeness" determined by regularization strength
      - maximizes **margin** for linearly separable datasets i.e. the distance between boundary and closest example

- **svc**: non-linear SVM
  - use `sklearn.svm.SVC`
      - default is to use **radial basis function (rbf)**
      - `SVC(gamma=0.1)`: smaller gamma means smoother boundary; too high results in overfitting
  - learns to surround data points for a given class
  - kernel SVMs are fast if there are few support vectors

  ```python
  svm = SVC(kernel="linear")
  svm.fit(X, y)

  print("Number of support vectors", len(svm.support_))
  X_small = X[svm.support_]
  y_small = y[svm.support_]
  ```

  - kernel SVMs implement feature transformations in a computationally efficient way
    - for features that are not linearly separable, can sometimes apply a transformation to make them linearly separable
      - e.g. if class 1 is concentrated around (0, 0) but otherwise surrounded by class 2, transforming X_n = X ** 2 allows for a linear boundary; in original representation would need a contour/ellipse

- **stochastic gradient descent (SGD)**:
  - see `sklearn.linear_model.SGDClassifier`
      - specify log loss or hinge loss via `loss='log'` or `loss='hinge'` param
  - scales well to large datasets
  - uses `alpha` instead of `C` to indicate regularization; higher alpha == more regularization

## Measuring Model Performance

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

- **log loss**: a loss function to be minimized; can measure model performance by seeing which has smallest error
    - looks at actual value, predicted value, and confidence (`p`)
    - penalizes high confidence when wrong
    - minimized most when correct and confident

```python
def compute_log_loss(predicted, actual, eps=1e-14):
  """Compute log loss

  eps: log(0) is infinity, so offset predicted values slightly from 0 or 1
  """
  # values smaller than eps become eps, values larger than 1 - eps become 1 - eps
  predicted = np.clip(predicted, eps, 1 - eps)
  loss = -1 * np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1-predicted))
  return loss
```

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
