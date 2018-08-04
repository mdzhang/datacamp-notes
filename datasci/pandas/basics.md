# Pandas Basics

- pandas data inspection methods
  - `df.info()` shows col names, types, non-null counts
  - `df.describe()` shows summary statistics for numerical columns
  - `df.dtypes.value_counts()` shows # of columns of specific data type
  - `df[:].apply(pd.Series.nunique)` shows # unique values per column (esp useful for categorical data)
  - `df.shape` shows # cols and rows
  - `df.head()` and `df.tail()` let you peak at your data


- apply a lambda to each column
    ```python
    categorize_label = lambda x: x.astype('category')
    df[LABELS] = df[LABELS].apply(categorize_label, axis=0)
    ```


