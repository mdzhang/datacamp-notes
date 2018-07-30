# Supervised Learning

- **supervised learning**: uses labeled data
    - has **predictor variables/features/independent variables** and a **target/response/dependent variable**
    - **labeled data** has known output i.e. target variable value is known
    - goal is to build a model that can predict target variable given predictor variables
        - to automate time consuming or expensive manual task
        - to make predictions about the future
    - if target variable is continuous, it's a **regression task**
    - if target variable is discrete, it's a **classification task**
        - binary decisions have discrete set of values i.e. true or false

- **weak learner**: a model that performs only slightly better than random guessing
- **strong learner**

- **training data**: already labeled data classifier uses to learn
- **test data**: already labeled data used to measure model/classifier performance
- **hold-out set**: data used to evaluate how a model performs on unseen data when it's fitted using cross validation

- running a model on a dataset aka "training a model on", "fitting a model to", "fitting a classifier to" data
- split data into **training data** and **test data**
    - don't reuse training data when measuring model performance - model has already seen it, not indicative of ability to generalize
    - split test/training data so that both reflect larger data set, each reflects distribution of original data set

- **overfitting**
    - sensitive to noise in the specific labeled data you have rather than reflecting general trends in the data
    - predicts well on training data, but poorly on test data, new data sets
    - low training set error, high test set error
    - model may be too complex
    - predictive power is low
    - remedy:
      - decrease model **complexity**
      - gather more data

- **underfitting**:
    - model may be too simple; not complex enough to capture dependencies between features and labels
    - low training accuracy
    - training/test set errors roughly equivalent, but both also relatively high
    - measured error exceeds desired error
    - remedy:
      - increase model **complexity**
      - gather more relevant features

- **imputting data**: make an educated guess as to what the missing values in a dataset could be
    - e.g. replace with mean of non-missing values

- **train/test split** may fail if split such that training set never sees a label that occurs in test split

## Bias-Variance tradeoff

- suppose you have a model `f^` that tries to approximate `f`, the true relationship between features and labels

- **generalization error**: how poorly a model generalizes on unseen data
    - definition
      - `error(f^) = bias ** 2 + variance + irreducible error`
      - **irreducible error**: contribution of noise that accompanies all data generation
      - **bias**: on average, how different are `f` and `f^`
          - high bias leads to **underfitting**
          - accuracy
      - **variance**: how much `f^` is inconsistent over different training sets
          - high variance models lead to **overfitting**
          - precision
    - measurement
      - difficult b/c
        - `f` is unknown
        - only have 1 dataset to measure variance for
        - noise is unpredictable, can't gauge irreducible error
      - can estimate by doing train/test split and using error on test set as approximator for generalization error
          - with **K-fold cross validation**, can calculate the error on each fold, and take the average of all errors

- model **complexity** sets flexibility to approximate true function f
    - e.g. increasing max depth increases complexity of decision tree
    - when model complexity increases, bias decreases and variance increases
        - goal to find model complexity that minimizes generalization error

- **bias-variance tradeoff**: as bias increases, variance decreases
