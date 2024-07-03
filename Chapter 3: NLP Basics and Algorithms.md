### Chapter 3: NLP Basics and Algorithms

#### 1. Introduction to NLP
- **What is NLP?**
  - Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through natural language. The goal is to read, decipher, understand, and make sense of human languages in a valuable way.

#### 2. Text Preprocessing
- **Tokenization**
  - Splitting text into individual words or sentences.
  ```python
  from nltk.tokenize import word_tokenize, sent_tokenize
  
  text = "Hello, world! How are you today?"
  words = word_tokenize(text)
  sentences = sent_tokenize(text)
  
  print(words)      # Output: ['Hello', ',', 'world', '!', 'How', 'are', 'you', 'today', '?']
  print(sentences)  # Output: ['Hello, world!', 'How are you today?']
  ```

- **Stop Words Removal**
  - Removing common words that do not carry significant meaning.
  ```python
  from nltk.corpus import stopwords
  
  stop_words = set(stopwords.words('english'))
  filtered_words = [word for word in words if word.lower() not in stop_words]
  
  print(filtered_words)  # Output: ['Hello', ',', 'world', '!', 'today', '?']
  ```

- **Stemming and Lemmatization**
  - Reducing words to their root form.
  ```python
  from nltk.stem import PorterStemmer
  from nltk.stem import WordNetLemmatizer
  
  stemmer = PorterStemmer()
  lemmatizer = WordNetLemmatizer()
  
  stemmed = [stemmer.stem(word) for word in filtered_words]
  lemmatized = [lemmatizer.lemmatize(word) for word in filtered_words]
  
  print(stemmed)      # Output: ['Hello', ',', 'world', '!', 'today', '?']
  print(lemmatized)   # Output: ['Hello', ',', 'world', '!', 'today', '?']
  ```

#### 3. Bag of Words and TF-IDF
- **Bag of Words (BoW)**
  - Represents text as a collection of word frequencies.
  ```python
  from sklearn.feature_extraction.text import CountVectorizer
  
  corpus = [
      'This is the first document.',
      'This document is the second document.',
      'And this is the third one.',
      'Is this the first document?'
  ]
  vectorizer = CountVectorizer()
  X = vectorizer.fit_transform(corpus)
  
  print(vectorizer.get_feature_names_out())
  # Output: ['and' 'document' 'first' 'is' 'one' 'second' 'the' 'third' 'this']
  print(X.toarray())
  # Output: [[0 1 1 1 0 0 1 0 1]
  #          [0 2 0 1 0 1 1 0 1]
  #          [1 1 0 1 1 0 1 1 1]
  #          [0 1 1 1 0 0 1 0 1]]
  ```

- **TF-IDF (Term Frequency-Inverse Document Frequency)**
  - Weighs words based on their frequency and importance across documents.
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(corpus)
  
  print(vectorizer.get_feature_names_out())
  # Output: ['and' 'document' 'first' 'is' 'one' 'second' 'the' 'third' 'this']
  print(X.toarray())
  # Output: [[0.         0.46979138 0.58028582 0.38408524 0.         0.
  #           0.38408524 0.         0.46979138]
  #          [0.         0.6876236  0.         0.37602176 0.         0.52305744
  #           0.37602176 0.         0.37602176]
  #          [0.43877656 0.36388646 0.         0.27803808 0.43877656 0.
  #           0.27803808 0.43877656 0.36388646]
  #          [0.         0.46979138 0.58028582 0.38408524 0.         0.
  #           0.38408524 0.         0.46979138]]
  ```

#### 4. Word Embeddings
- **Word2Vec**
  - Word embeddings are dense vectors representing words in a continuous vector space.
  ```python
  from gensim.models import Word2Vec
  
  sentences = [
      ['this', 'is', 'the', 'first', 'document'],
      ['this', 'document', 'is', 'the', 'second', 'document'],
      ['and', 'this', 'is', 'the', 'third', 'one'],
      ['is', 'this', 'the', 'first', 'document']
  ]
  model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
  
  vector = model.wv['document']
  print(vector)  # Output: Word vector for 'document'
  ```

- **GloVe**
  - Global Vectors for Word Representation, another method to obtain word embeddings.
  ```python
  import gensim.downloader as api
  
  glove_vectors = api.load("glove-wiki-gigaword-100")
  
  vector = glove_vectors['document']
  print(vector)  # Output: Word vector for 'document'
  ```

#### 5. Basic NLP Algorithms
- **Text Classification**
  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.pipeline import make_pipeline
  
  model = make_pipeline(TfidfVectorizer(), MultinomialNB())
  model.fit(corpus, [0, 1, 2, 0])
  
  labels = model.predict(corpus)
  print(labels)  # Output: Predicted labels for the documents
  ```

- **Sentiment Analysis**
  ```python
  from textblob import TextBlob
  
  text = "I love this product! It's amazing."
  blob = TextBlob(text)
  sentiment = blob.sentiment
  
  print(sentiment)  # Output: Sentiment(polarity=0.75, subjectivity=0.8)
  ```

#### 6. NLP Libraries
- **NLTK (Natural Language Toolkit)**
  - Provides easy-to-use interfaces to over 50 corpora and lexical resources.
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  ```

- **spaCy**
  - Industrial-strength NLP library with pre-trained models for tokenization, POS tagging, named entity recognition, etc.
  ```python
  import spacy
  
  nlp = spacy.load("en_core_web_sm")
  doc = nlp("Hello, world! How are you today?")
  
  for token in doc:
      print(token.text, token.pos_, token.dep_)
  ```

---

Please review Chapter 3 in detail. Let me know when you're ready to proceed to the next chapter by saying "OK," "next," or "got it."
