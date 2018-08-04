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

- **tf-if (term frequency - inverse document frequency)**: used to determine most important words in each document in a corpus by assigning weights to individual tokens
    - identify common, shared words in a corpus that aren't just stopwords
    - should be down-weighted in importance so common words don't show up as key words
        - want words important within a document weighted high, not words important across whole corpus
    - e.g. "sky" for an astronomy corpus

    - `w(i, j) = tf(i, j) * log(N / df(i))`
        - w(i, j): weight of token i in document j
        - tf(i, j): # occurrences token i in document j / total # tokens in document j
        - df(i): # documents containing token i
        - N: total # documents
        - recall `ln(1) = 0` so the more documents contain token i, the closer `N / df(i)` is to 1 so smaller the weight
            - fewer documents containing token i, the larger `N / df(i)` is, the larger `ln(N / df(i)` is i.e. the heavier the weight

      ```python
      from gensim.models.tfidfmodel import TfidfModel

      # tfidf is a dictionary of documents in a corpus => list of (word id, tfidf weight)
      tfidf = TfidfModel(corpus)
      ```

##### With scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer

# split on whitespace
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# extracts only those tokens that match the given pattern
# each token is a feature for the model
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# learn vocabulary dictionary of all tokens
vec_alphanumeric.fit(df['some_free_text_column'])
# feature names are distinct tokens
vec_alphanumeric.get_feature_names()
```

- tf-idf


```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
print(tfidf_vectorizer.get_feature_names()[:10])
```

## Named Entity Recognition

##### With nltk

```python
import nltk

sentence = ...

tokenized_sent = nltk.word_tokenize(sentence)

# dict of tuples containing (token, part of speach)
tagged_sent = nltk.pos_tag(tokenized_sent)

# builds tree of tokens and their named entity type
# e.g. New York => GPE (geopolitical entity)
nltk.ne_chunk(tagged_sent)

# use nltk.ne_chunk_sents if you have many sentences
```

##### With spacy

- support for building NLP pipelines
- different entity types than `nltk`
- corpora for informal language
- growing quickly

```python
import spacy

# load pre-trained word vectors
nlp = spacy.load('en')

doc = nlp('some text...')

# list entities, each with .text and .label_ fields
doc.ents
```

## Machine Learning

- **Naive Bayes Model** is commonly used for testing NLP classificationp problems
    - answers: given a particular piece of data, how likely is an outcome?
    - relatively simple and effective
    - commonly used in domain since initial application in 1960s

```python
# multinomialnb good for multiple label classification
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

nb_clf = MultinomialNB()
nb_clf.fit(count_train, y_train)
pred = nb_clf.predict(count_test)
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
