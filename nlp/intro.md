# NLP

- **Natural language processing (NLP)**: process of processing free form text to create features for a machine learning algorithm
    - **tokenization**: process of splitting long string into segments to store in a list
        - can split along whitespace, punctuation, a pattern/regex
    - **bag of words**: counts # of times a word appears in a text
        - lose information about work order, grammar

      ```python
      from sklearn.feature_extraction.text import CountVectorizer

      # split on whitespace
      TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
      vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)
      vec_alphanumeric.fit(df['some_free_text_column'])
      # feature names are distinct tokens
      vec_alphanumeric.get_feature_names()
      ```
    - **vocabulary**: all tokens that appear in data set
    - **n-grams**: split into lists of `n` consecutive tokens
        - e.g. 2-gram/bi-gram for "red blue green" includes ["red blue", "blue green"]


