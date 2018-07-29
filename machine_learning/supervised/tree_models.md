# ML with Tree-based Models

## CARTs Overview

- **classification and regression trees (CARTs)** / **decision trees**
    - definition
      - supervised learning models used for problems involving classification and regression
      - can answer if-else questions about features to infer labels
      - produces rectangular decision regions (v linear boundaries in linear regression models)
    - structure
      - binary tree consist of nodes
        - ea node is a question (root or internal node) or prediction (leaf node)
        - leaves have a predominant classification label i.e. leaves are **pure**
        - to product purest leaves, at each node, tree asks question `g(f, sp)` where `f` is a feature and `sp` is a **split point**
            - picks `f` and `sp` by seeing which values maximize **information gain**
            - if `IG = 0`, node is a leaf
        - can measure impurity of a node w/ e.g. **gini index** or **entropy**
            - mostly have the same result, gini is faster to compute and is the sklearn default

      ```python
      dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)
      dt_entropy.fit(X_train, y_train)
      ```
    - pros
      - flexibility: can capture non-linear relationships between features and models
      - easier preprocessing: don't require feature scaling/standardization
      - simple to understand, use, interpret
    - cons
      - prone to memorizing the noise present in a dataset (high variance, overfitting)
      - classification trees can only product orthogonal decision boundaries
      - sensitive to small variations in training set - variable can be removed, and CARTs parameters may change drastically
    - evaluation
      - **mean squared error (MSE)** to guage error (`sklearn.metrics.mean_squared_error`)
      - typical accuracy measurements (`sklearn.metrics.accuracy_score`)
      - can measure importance of each feature in prediction
          - how much tree uses a feature to reduce impurity; value is a percentage indicating weight of that feature in training and prediction

- **classification tree**: decision tree for categorical data

      ```python
      from sklearn.tree import DecisionTreeClassifier
      from sklearn.metrics import accuracy_score

      # max_depth is max levels in tree
      dt = DecisionTreeClassifier(max_depth=6, random_state=SEED)

      dt.fit(X_train, y_train)
      y_pred = dt.predict(X_test)

      acc = accuracy_score(y_test, y_pred)

      # get feature weight values
      dt.feature_importances_
      ```

      - plot feature importance

      ```python
      importances = pd.Series(data=dt.feature_importances_,
                              index=X_train.columns)

      importances_sorted = importances.sort_values()

      importances_sorted.plot(kind='barh', color='lightgreen')
      ```

- **regression tree**

      ```python
      from sklearn.tree import DecisionTreeRegressor
      from sklearn.model_selection import train_test_split
      from sklearn.metrics import mean_squared_error as MSE

      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
      # stopping condition that each leaf needs at least 10% of training data
      dt = DecisionTreeRegressor(max_depth=4, random_state=3, min_samples_leaf=0.1)

      dt.fit(X_train, y_train)
      y_pred = dt.predict(X_test)

      mse_dt = MSE(y_test, y_pred)
      rmse_dt = mse_dt ** (1/2)
      ```

- **k-fold cross validation**

    ```python
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error as MSE
    from sklearn.model_selection import cross_val_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    dt = DecisionTreeRegressor(max_depth=4, random_state=3, min_samples_leaf=0.1)

    # set n_jobs to leverage all cpus
    MSE_CV = - cross_val_score(reg, X_test, y_test, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    print(MSE_CV.mean())
    RMSE_CV = (MSE_CV_scores.mean())**(1/2)

    print(MSE(y_train, y_predict_train)
    print(MSE(y_test, y_predict_test)
    ```

## Ensemble learning

- **ensemble methods**: compensate for weakness of CARTs by aggregating the predictions of trees that are trained differently
- **ensemble learning**
    - train different models on same dataset
        - models can use different algorithms e.g. decision tree, linear regression, kNN, etc.
    - let each model make predictions
    - meta-model aggregates predictions of individual models
    - final prediction more robust and less prone to errors

- **voting classifier**: chooses most common prediction among all its classifiers i.e. uses **hard voting**
    - classifiers are fit to the same training set
    - classifiers can use different algorithms

    ```python
    from sklearn.ensemble import VotingClassifier

    # instantiate classifiers
    ...

    classifiers = [
      ('logistic regression', lr),
      ('k nearest neighbors', knn),
      ('classification tree', dt),
    ]

    # fit individual classifiers
    ...

    vc = VotingClassifier(estimators=classifiers)
    vc.fit(X_train, y_train)
    y_pred = vc.predict(X_test)
    ```

-  **bootstrap aggregation** (**bagging** for short): ensemble method where you use the same algorithm/classifier, but with each instance of the classifier trained on a different subset of the training set
    - reduces variance of individual models in the ensemble (?)
    - draw bootstrap samples from the original data set w/ replacement i.e. any single observation can be drawn any number of times
        - also means that potentially some observations are not sampled at all
        - on average, model samples 63% of all observations
        - remaining 37% are **out of bag (OOB)** instances
    - use all features of the data to generate predictions
    - estimator predictions are then aggregated
      - majority voting for classification tasks `sklearn.ensemble.BaggingClassifier`
      - average value in regression tasks `sklearn.ensemble.BaggingRegressor`

    ```python
    from sklearn.ensemble import BaggingClassifier

    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16)
    bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1)
    ```

    - OOB instances can be used for evaluation (instead of e.g. k-fold cross validation) since they are not used in training
        - **OOB evaluation** generates a value that averages OOB scores across all estimators
        - evaluation method is often accuracy for classifiers, R2 for regressors

    ```python
    bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1, oob_score=True)
    bc.fit(X_train, y_train)
    oob_accuracy = bc.oob_score_
    ```

- **random forests**
    - definition
      - uses decision tree as base estimator
      - trains estimators on different bootstrap samples, each the same size as training data set
      - decision tree samples `d` features at each node/split w/o replacement (?)
          - `d` defaults to `math.sqrt(len(features))`
    - estimator predictions are then aggregated
      - majority voting for classification tasks `sklearn.ensemble.RandomForestClassifier`
      - average value in regression tasks `sklearn.ensemble.RandomForestRegressor`

