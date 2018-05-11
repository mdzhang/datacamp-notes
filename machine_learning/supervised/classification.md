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

- **train/test split** may fail if split such that training set never sees a label that occurs in test split

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

- **multi class logistic regression**
    - performs logistic regression on each column separately, separate classifier per column

    ```python
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression

    clf = OneVsRestClassifier(LogisticRegression())
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
