# NLP

- **Natural language processing (NLP)**: make sense of languages using statistics and computers
  - subtopics include:
    - topic identification
    - text classification
  - applications include:
    - chatbots
    - translation
    - sentiment analysis

- **regex**: used to match patterns in strings
    - see python `re` module
        - `re.match` matches from string start
        - `re.search` looks through entire string
        - always pass regex first, string next
    - exist many special chars
    - capitalizing negates special chars e.g. `\S` instead of `\s`

- **tokenization**: process of splitting long string/document into segments/smaller chunks/**tokens** to store in a list
    - can split along whitespace, punctuation, a pattern/regex
    - see python `nltk` (natural language toolkit) package

      ```python
      from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize, TweetTokenizer

      # split into sentences
      sentences = sent_tokenize(scene_one)

      # split sentence into words
      tokenized_sent = word_tokenize(sentences[3])

      # find unique words in document
      unique_tokens = set(word_tokenize(scene_one))

      # find @-mentions and hashtags in a tweet
      pattern2 = r"([@#]\w+)"
      regexp_tokenize(tweets[-1], pattern2)

      # tweet tokenizer pulls @-mentions and hashtags
      tknzr = TweetTokenizer()
      all_tokens = [tknzr for t in tweets]
      ```

- pre-processing: preprocessing text before performing e.g. machine learning, other stats methods
  - tokenization
  - lower casing
  - **lemmatization** aka **stemming** where you shorten words to root stems
      - `nltk.stem.WordNetLemmatizer` e.g. `WordNetLemmatizer().lemmatize(token)`
  - remove **stop words** i.e. common words that don't add meaning e.g. 'the', 'and'
      - `nltk.corpus.stopwords` e.g. `... not in stopwords.words('english')`
  - remove punctuation, other unwanted tokens

- **word vector**: multi dimensional representation of a word
    - see relationships between words based on e.g. nearness in a corpus
- **LDA visualization**


## Bag of Words

- **bag of words**: counts # of times a word appears in a text
  - a basic method for finding topics in a text
  - lose information about work order, grammar

##### With nltk

```python
from collections import Counter

from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize, TweetTokenizer

tokens = word_tokenize(article)
lower_tokens = [t.lower() for t in tokens]

cnt = Counter(lower_tokens)
print(cnt.most_common(10))
```

##### With gensim

```python
from gensim.corpora.dictionary import Dictionary

dictionary = Dictionary(articles)
# get id for token 'computer'
computer_id = dictionary.token2id.get("computer")

# corpus is collection of bag of words, one for each article
corpus = [dictionary.doc2bow(article) for article in articles]
```

##### With scikit-learn

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
















## Machine Learning

- process of processing free form text to create features for a machine learning algorithm


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
