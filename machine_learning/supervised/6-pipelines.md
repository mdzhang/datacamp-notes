## Machine Learning Pipelines

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
