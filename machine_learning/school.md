- **human in the loop machine learning system**
  - machine learning algorithm determins a label with some probability
  - human prioritizes time spent based on those probabilities

- pandas data inspection methods
  - `df.info()` shows col names, types, non-null counts
  - `df.describe()` shows summary statistics for numerical columns
  - `df.dtypes.value_counts()` shows # of columns of specific data type
  - `df[:].apply(pd.Series.nunique)` shows # unique values per column (esp useful for categorical data)
  - `df.shape` shows # cols and rows
  - `df.head()` and `df.tail()` let you peak at your data

- pandas categorical data types encode strings as numbers to speed up processing
- `get_dummies` converts categorical data types to multiple columns (**dummy variables**) with a value of `0` or `1` i.e. to **binary indicator representation**

```python
X_train, X_test, y_train, y_test = train_test_split(df[['numeric']],
                                                    pd.get_dummies(df['label']),
                                                    random_state=22)
```

- apply a lambda to each column
    ```python
    categorize_label = lambda x: x.astype('category')
    df[LABELS] = df[LABELS].apply(categorize_label, axis=0)
    ```

- start with basic model
  - less can go wrong than with complex models
  - see how much "signal" can be gleaned with basic methods
  - less computationally expensive
  - think carefully about features
