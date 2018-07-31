# Data preprocessing

- occurs after data cleaning and exploratory data analysis (EDA)
- prep data for e.g. modeling

## Data cleaning

- removing missing data
  - drop all empty values in all columns and rows: `df.dropna()`
  - drop all empty values in specific rows by index: `df.drop([1, 2, 3])`
  - drop column: `df.drop('col_name', axis=1)`
  - drop all rows with an empty value in specific column: `df[df['col_name'].notnull()]`
  - drop all columns with at least 3 empty values: `df.dropna(axis=1, thresh=3)`

- imputing missing data: replace missing values with some estimator

    ```python
    from sklearn.preprocessing import Imputer

    # create an imputer that will replace missing values ('NaN') with
    # the most frequently occuring value in that column (axis=0)
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

    # by default, would replace np.nan with the column mean
    imp = Imputer()
    ```

- converting data types
  - for specific column: `df['col_name'] = df['col_name'].astype(float)`
      - available types: `float`, `int`, `string`

## Standardizing Data

- **normalization** / **scaling and centering** / **standardization**
  - applies to continuous, numerical data
  - the transformation of continuous data to make it look normally distributed
  - compensates for **numerical noise**
    - high variance
      - models in linear space assume linear independence e.g. kNN, k-means clustering, linear regression
      - some models assume normally distributed data
    - differently scaled data
      - feature on a different magnitude inhibits model's ability to learn from other features, creating bias

- multiple normalization methods e.g.
  - convert to `[0, 1]` range by subtracting the minimum and dividing by the range
  - convert to `[-1, 1]` range

  - **log normalization**: applies log transformation to values to make them approximate normality
      - log transformation i.e. taking natural log (2.718) of value
      - `df.var()` to see variances of each column
      - `np.log(df['col'])` to take log of column

  - **feature scaling**:
    - use when working with features on different scales, using model in linear space
    - centers data around 0 with variance 1 by subtracting the mean and dividing by the variance (strict **standardization**)

    ```python
    from sklearn.preprocessing import scale

    X_scale = scale(X)
    ```

    ```python
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    ```

    ```python
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    steps = [('scaler', StandardScaler()),
            ('imputation', imp),
            ('SVM', clf)]
    pipeline = Pipeline(steps)
    ```

## Feature Engineering

- **feature engineering**: ways to extract from/expand on existing data/features
  - adds new features based on existing ones useful for prediction/clustering task
  - sheds insights into relationships between features
  - dataset dependent

### Encoding categorical data

- encode binary variable as `0` and `1`

    ```python
    users['sub_enc'] = users['subscribed'].apply(lambda val: 1 if val == 'y' else 0)
    ```

    ```python
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    users['sub_enc'] = le.fit_transform(users['subscribed'])
    ```

- **one-hot encoding** converts categorical data types to multiple columns with a value of `0` or `1`
  - aka **binary indicator representation**
  - new columns aka **dummy variables**
  - pandas has `get_dummies` function to encode strings as numbers to speed up processing

    ```python
    import pandas as pd

    pd.get_dummies(df['label'])
    ```

### Engineering numerical features

- take an aggregate of a set of numerical columns to use in place of those columns/features

    ```python
    columns = ['day1', 'day2', 'day3']
    df['mean'] = df.apply(lambda row: row[columns].mean(), axis=1)
    ```

    - reduces data dimensionality

- extract part of timestamp e.g. month, etc to use as new feature

    ```python
    df['date_converted'] = pd.to_datetime(df['date'])
    df['month'] = df['date_converted'].apply(lambda row: row.month)
    ```

### Engineering features from text

- extract text

- vectorize text with **tf-idf**

    ```python
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vec = TfidfVectorizer()
    text_tfidf = tfidf_vec.fit_transform(documents)
    ```

    - can use e.g. **Naive Bayes Classifier** on vectorized text
        - known to perform well on text classification
        - treats each feature as independent from the others
        - efficient, works well on high-dimension data

## Feature Selection

- **feature selection**: select features to be used in modeling to improve model performance
  - remove unnecessary features i.e. don't strengthen model/improve model's predictive power
  - feature set is too large
    - uses too many computational resources
    - hard to interpret or maintain resulting models
    - redundant
      - strongly statistically correlated w/ another feature
        - introduces bias to e.g. linear models that assume variables are independent
        - use `df.corr()` to see highly correlated variables
      - duplicated
      - noisy: exists in another form as another feature
  - reminders
    - often an iterative process
    - helps to know your dataset well
    - understand and keep your modeling task in mind
  - e.g.
    - pick words with highest weights after running tf-idf

- automated feature selection
  - variance threshold
  - univariate statistical tests

### Selecting text features

- after using tf-idf, may only want to keep top X% of weighted words

    ```python
    # dict of words => indexes
    tfidf_vec.vocabulary_

    # use word index in vocab to lookup weight
    tfidf_vec[idx].data

    tfidf_vec[idx].indices
    ```

## Dimensionality Reduction

- **dimensionality reduction**: unsupervised learning method that shrinks # features in feature space
  - useful when you have large # features and no strong candidates for reduction
- **pricipal component analysis (PCA)**: dimensionality reduction method that uses linear transformation to project features into a space where they are completely uncorrelated (?)
    - combines features into components to capture variance (?)
    - difficult to interpret PCA components
    - use at end of preprocessing (since data is transformed and reshaped)

    ```python
    from sklearn.decomposition import PCA

    pca = PCA()
    # by default, same # components as input features
    df_pca = pca.fit_transform(df)

    # see percentage of variance per component
    # can drop components with low percentage of variance
    print(pca.explained_variance_ratio_)
    ```
