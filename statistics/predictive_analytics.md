# Predictive Analytics

## Basic terminology & Overview

- **predictive analytics**: process that aims to predict an event using historical data
- **analytical basetable**:
    - constructed from historical data e.g. on similar events
    - population: group of people or objects to make prediction for
        - basetable has row for each object in population
    - candidate predictors
        - drawbacks of using too many predictor variables:
          - model prone to **overfitting**
          - model harder to maintain and implement
            - should limit running time of model, time needed to create variables
          - hard to interpret - need to interpret coefficients to ensure they make sense
        - goal of **variable selection** is to select variables with optimal performance
    - target: information about event to predict; 1 if event occurs, 0 otherwise
- **incidence**: # targets / pop'n size

- when running predictive analytics
  - build model
  - evaluate model performance w/ AUC
  - evaluate & visualize model w/ cumulative gains graphs and lift curves
  - verify that variables in model and their link to target are interpretable

## Building a model

- **logistic regression**: a predictive modelling technique
    - get a function of the form `y = ax + b` to fit a dataset
    - `y` can be any real number, but want to predict 0 or 1
    - **logit** function transforms any real number `y` to a value in `[0, 1]`
        - takes formula as input, calculates a probability as output
    - allows using linear regression for binary classification problems

    ```python
    from sklearn import linear_model

    logreg = linear_model.LogisticRegression()
    X = basetable[['age']]
    Y = basetable[['target']]

    # fit logistic regression model to data
    logreg.fit(X, Y)

    # coef is value of `a`
    print(logreg.coef_)

    # intercept is value of `b`
    print(logreg.intercept_)
    ```

- **multivariate logistic regression**
    - instead of `y = ax + b`, use `y = a1x1 + a2x2 + ... + b` i.e. add multiple predictors

    ```python
    from sklearn import linear_model

    logreg = linear_model.LogisticRegression()
    X = basetable[['age', 'max_gift', 'income_low']]
    Y = basetable[['target']]

    # fit logistic regression model to data
    logreg.fit(X, Y)

    # now a list
    print(logreg.coef_)

    wont_donate, will_donate = logreg.predict_proba([x1, x2, x3])
    ```

## Evaluating model

- **AUC value (area under ROC curve)**: measures how well model can order values from low to high chance to be a target (?)
    - perfect models have AUC = 1

    ```python
    import numpy as np
    from sklearn.metrics import roc_auc_score

    # true_target is vector of 0s and 1s indicating whether something actually was target (event did happen)
    # prob_target is vector of probabilities between 0 and 1 indicating how likely model ascertained something was the target
    roc_auc_score(true_target, prob_target)
    ```

- predictive model uses candidate predictors as inputs and links them w/ target
    - **forward stepwise variable selection** selects predictors/variables from large set of variables by finding
        - single variable w/ best AUC score
        - second variable w/ best AUC score in combination w/ first variable
        - ...and so on and so forth until all vars added or hit predefined threshold
        - only one of 2+ highly correlated variables may be included, as having 1 decreases the predictive analysis performance of having the 2nd, but increases a predictive models complexity, etc.
    - **overfitting**: when adding new variables to a model decreases that model's AUC when tested on new data
        - accuracy on training data increases, but true performance of model decreases b/c doesn't generalize to new data well
        - can test true performance of a model by splitting data into **train** and **test** sets, and evaluating model performance on test set after fitting it to training set
            - can **partition** into train/test datasets by randomly splitting
              - when target incidence is low, should ensure target incidence is same in train/test i.e. by **stratifying** on the target

      ```python
      from sklearn.cross_validation import train_test_split

      X = basetable.drop("target", 1)
      Y = basetable["target"]

      X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
      ```

      - should choose var count where AUC on test data is highest w/ fewer vars

## Visualizing model

- AUC is single number that doesn't capture all info about model, complex; not intuitive for business stakeholders

- **cumulative gains curve**
  - x-axis has, left to right, % of sample # highest to lower probability observations to be targeted (?)
  - y-axis has, bottom to top, proportion of targets (observations for which event occurs) reached when using the model

  ```python
  import scikitplot as skplt
  import matplotlib.pyplot as plt

  skplt.metrics.plot_cumulative_gain(true_values, predictions)
  plt.show()
  ```

  - which CGC is better is situational e.g. if can only reach certain % of population, choose model that reaches higher proportion of targets t that %

- **lift curve**
    - x-axis has, left to right, % of sample
    - y-axis has, bottom to top, # times more the model reaches target than average (higher the curve, the better)

  ```python
  import scikitplot as skplt
  import matplotlib.pyplot as plt

  skplt.metrics.plot_lift_curve(true_values, predictions)
  plt.show()
  ```

## Verifying interpretability

- **predictor insight graphs (PIG)** show link between predictor variables and target being predicted
    - visual description
      - categories on x-axis
      - frequency counts of categories on left y-axis
      - incidence of target on right y-axis
      - shows bars indicating category frequency; line showing incidence per category
    - to build:
      - discretize continuous variables by breaking continuous value range into bins

      ```python
      # can result in different sized or ugly intervals
      # use this to get an idea of what cuts to use
      basetable['disc_max_gift'] = pd.qcut(basetable['max_gift'], number_bins)
      # see interval frequency counts
      print(basetable.groupby("disc_max_gift").size())

      # to specify cuts (last value is upper bound of last interval)
      basetable['disc_max_gift'] = pd.cut(basetable['max_gift'], [18, 30, 40, 50, 60, 100])
      ```

      - prepare properly formatted table

      ```python
      import numpy as np

      groups = basetable[["target","variable"]].groupby("variable")
      groups["target"].agg({'Incidence' : np.mean, 'Size': np.size}).reset_index()
      ```
      - plot graph

      ```python
      import matplotlib.pyplot as plt
      import numpy as np

      # plot incidence line
      pig_table["Incidence"].plot(secondary_y=True)

      # x-axis category labels
      plt.xticks(np.arange(len(pig_table)), pig_table[variable])
      # center above
      plt.xlim([-0.5, len(pig_table)-0.5])

      # label y-axis
      plt.ylabel("Incidence", rotation = 0, rotation_mode="anchor", ha = "right")
      plt.xlabel(variable)

      # plot frequency count bars
      pig_table["Size"].plot(kind='bar', width = 0.5, color = "lightgray", edgecolor = "none")

      plt.show()
      ```
