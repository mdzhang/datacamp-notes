## Data preprocessing

- imputing data

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

- different pipeline steps per column

```python
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer

# just get 'text' column in returned data frame
get_text_data = FunctionTransformer(lambda df: df['text'], validate=False)
get_num_data = FunctionTransformer(lambda df: df['numeric'], validate=False)

tpipe = Pipeline([
  ('selector', get_text_data),
  ('vectorizer', CountVectorizer())
])

npipe = Pipeline([
  ('selector', get_num_data),
  ('imputer', Imputer())
])

pl = Pipeline([
  # feature union runs each step and concatenates output of each into wide array
  ('union', FeatureUnion([
    ('numeric', npipe),
    ('text', tpipe)
  ]),
  ('clf', OneVsRestClassifier(LogisticRegression()))
])
```
