## Hyperparameter Tuning

- models, algorithms below often need additional parameter values (see k in k-NN, coefficients in linear regression, alpha in ridge/lasso regression) that cannot be learned by fitting the model aka they must be chosen
    - aka **hyperparameters**
    - **hyperparameter tuning**: selecting best value for hyperparameters to get **optimal model** performance
      - select different value
      - fit all of them to your model separately
      - measure performance
          - use **cross-validation** to avoid overfitting parameters to test/train split, evaluate a model's ability to generalize and work on unseen data
      - select most performant values
    - **optimal model**  has optimal **score**
        - **score** defaults to accuracy (classification tasks) or R^2 (regression tasks)

- parameters are learned from data, but hyperparameters are not (and are set prior to training)

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

- **random search**

- **bayesian optimization**

- **genetic algorithms**
