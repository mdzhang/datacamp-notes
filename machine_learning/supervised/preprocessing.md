## Data preprocessing

- occurs after data cleaning and exploratory data analysis (EDA)
- prep data for e.g. modeling

##### Removing missing data

- drop all empty values in all columns and rows: `df.dropna()`
- drop all empty values in specific rows by index: `df.drop([1, 2, 3])`
- drop column: `df.drop('col_name', axis=1)`
- drop all rows with an empty value in specific column: `df[df['col_name'].notnull()]`
- drop all columns with at least 3 empty values: `df.dropna(axis=1, thresh=3)`

##### Converting data types

- for specific column: `df['col_name'] = df['col_name'].astype('float')`
    - available types: `float`, `int`, `string`

##### Imputing Data

```python
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# create an imputer that will replace missing values ('NaN') with
# the most frequently occuring value in that column (axis=0)
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# by default, would replace np.nan with the column mean
imp = Imputer()

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

##### Standardizing Data

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

##### Encoding Data (?)

- pandas categorical data types encode strings as numbers to speed up processing
- `get_dummies` converts categorical data types to multiple columns (**dummy variables**) with a value of `0` or `1` i.e. to **binary indicator representation**

```python
X_train, X_test, y_train, y_test = train_test_split(df[['numeric']],
                                                    pd.get_dummies(df['label']),
                                                    random_state=22)
```
