# Measuring Model Performance

- measure model performance by running it on a dataset and calculating a performance metric

## Measuring Predictive Performance

### Accuracy

- **accuracy**: total # correct predictions / total # data points
    - not always the best measure of model performance e.g. when a **class imbalance** exists e.g. 99% email not spam, 1% spam
    - can graph model complexity against accuracy for a **model complexity curve**
    - `sklearn` models often have a `.score()` method that uses accuracy as a default
    - commonly used for multiclass classification

### Confusion Matrix

- **confusion matrix**:

        |                | Predicted: spam | Predicted: not spam |
        |Actual: spam    | True positive   | False negative      |
        |Actual: not spam| False positive  | True negative       |

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

    # create model, fit to data, predict on y_test
    ...

    confusion_matrix(y_test, y_pred)

    # returns all matrics defined above (accuracy, precision, recall, F1-score)
    classification_report(y_test, y_pred)
    ```

### ROC

- **receiver operating characteristic curve (ROC)**:
    - curve resulting from graphing different `p` thresholds against
      - false positive rate aka **fallout** (x-axis)
      - true positive rate aka **recall** (y-axis)
    - the probability that a randomly chosen positive data point will have a higher rank than a randomly chosen negative point (?)

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

### Loss Functions

- **loss functions** aka **objective functions** quantify how far off a prediction is from an actual result for real number value predictions
    - goal of learners is to yield minimum value of loss function

#### Log Loss

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

## Measuring (and Penalizing) Complexity

- **regularization** penalizes complex models

- **gamma**
  - measure for tree-based learners that determines whether a node will split based on expected reduction in loss (?)
  - higher values for gamma lead to fewer splits

- **alpha** aka **L1 regularization**
  - in tree-based learners
    - penalizes leaf (not feature) weights
    - higher alpha => stronger regularization => leaf weights approach 0

- **lambda** aka **L2 regularization**
  - leaf weights decrease smoothly v L1 strong sparsity constraints (?)

