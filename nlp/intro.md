# NLP

- **Natural language processing (NLP)**: make sense of languages using statistics and computers
  - subtopics include:
    - topic identification
    -


## Machine Learning

    - process of processing free form text to create features for a machine learning algorithm
    - **tokenization**: process of splitting long string into segments to store in a list
        - can split along whitespace, punctuation, a pattern/regex
    - **bag of words**: counts # of times a word appears in a text
        - lose information about work order, grammar

      ```python
      from sklearn.feature_extraction.text import CountVectorizer

      # split on whitespace
      TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
      vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

      # learn vocabulary dictionary of all tokens
      vec_alphanumeric.fit(df['some_free_text_column'])
      # feature names are distinct tokens
      vec_alphanumeric.get_feature_names()
      ```

    - **vocabulary**: all tokens that appear in data set
    - **n-grams**: split into lists of `n` consecutive tokens
        - e.g. 2-gram/bi-gram for "red blue green" includes ["red blue", "blue green"]

```python
# Define combine_text_columns()
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """

    # drop non-text columns, fill in NaNs with empty strings
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)
    text_data.fillna("", inplace=True)

    # join all columns into a single column
    # where the value of each row is a concatenated string of the text rows
    return text_data.apply(lambda x: " ".join(x), axis=1)
```
