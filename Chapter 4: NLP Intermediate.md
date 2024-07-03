### Chapter 4: NLP Intermediate

#### 1. Advanced Text Preprocessing
- **Named Entity Recognition (NER)**
  - Identifying and classifying named entities (e.g., people, organizations, locations) in text.
  ```python
  import spacy
  
  nlp = spacy.load("en_core_web_sm")
  doc = nlp("Apple is looking at buying U.K. startup for $1 billion.")
  
  for entity in doc.ents:
      print(entity.text, entity.label_)
  # Output:
  # Apple ORG
  # U.K. GPE
  # $1 billion MONEY
  ```

- **Part-of-Speech (POS) Tagging**
  - Identifying the grammatical parts of speech in text.
  ```python
  for token in doc:
      print(token.text, token.pos_, token.dep_)
  # Output:
  # Apple PROPN nsubj
  # is AUX aux
  # looking VERB ROOT
  # at ADP prep
  # buying VERB pcomp
  # U.K. PROPN compound
  # startup NOUN dobj
  # for ADP prep
  # $ SYM quantmod
  # 1 NUM compound
  # billion NUM nummod
  ```

#### 2. Word Embeddings: Deeper Dive
- **Word2Vec Models: Skip-Gram and CBOW**
  - **Skip-Gram**: Predicts context words from a target word.
  - **CBOW (Continuous Bag of Words)**: Predicts a target word from context words.
  ```python
  from gensim.models import Word2Vec
  
  sentences = [
      ['this', 'is', 'the', 'first', 'document'],
      ['this', 'document', 'is', 'the', 'second', 'document'],
      ['and', 'this', 'is', 'the', 'third', 'one'],
      ['is', 'this', 'the', 'first', 'document']
  ]
  model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)  # Skip-Gram
  model_cbow = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0)  # CBOW
  ```

- **GloVe: Global Vectors for Word Representation**
  - Utilizes the co-occurrence matrix of words in a corpus to create embeddings.
  ```python
  import gensim.downloader as api
  
  glove_vectors = api.load("glove-wiki-gigaword-100")
  
  vector = glove_vectors['document']
  print(vector)  # Output: Word vector for 'document'
  ```

#### 3. Advanced NLP Algorithms
- **Named Entity Recognition (NER)**
  - Identifying and classifying entities in text into predefined categories.
  ```python
  from nltk import ne_chunk, pos_tag, word_tokenize
  from nltk.tree import Tree
  
  sentence = "Mark Zuckerberg is the CEO of Facebook."
  chunked = ne_chunk(pos_tag(word_tokenize(sentence)))
  
  for subtree in chunked:
      if type(subtree) == Tree:
          print(' '.join([token for token, pos in subtree.leaves()]), subtree.label_)
  # Output:
  # Mark Zuckerberg PERSON
  # Facebook ORGANIZATION
  ```

- **Dependency Parsing**
  - Understanding the grammatical structure of a sentence by identifying relationships between "head" words and words which modify those heads.
  ```python
  for token in doc:
      print(f'{token.text} -> {token.head.text} ({token.dep_})')
  # Output:
  # Apple -> looking (nsubj)
  # is -> looking (aux)
  # looking -> looking (ROOT)
  # at -> looking (prep)
  # buying -> at (pcomp)
  # U.K. -> startup (compound)
  # startup -> buying (dobj)
  # for -> buying (prep)
  # $ -> billion (quantmod)
  # 1 -> billion (compound)
  # billion -> for (pobj)
  ```

#### 4. NLP Architectures
- **Recurrent Neural Networks (RNN)**
  - Neural networks with loops to allow information to persist.
  - Useful for sequence data like text.
  ```python
  from keras.models import Sequential
  from keras.layers import SimpleRNN, Dense, Embedding
  
  model = Sequential()
  model.add(Embedding(input_dim=10000, output_dim=128))
  model.add(SimpleRNN(units=128))
  model.add(Dense(units=1, activation='sigmoid'))
  
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  ```

- **Long Short-Term Memory (LSTM)**
  - A type of RNN capable of learning long-term dependencies.
  ```python
  from keras.layers import LSTM
  
  model = Sequential()
  model.add(Embedding(input_dim=10000, output_dim=128))
  model.add(LSTM(units=128))
  model.add(Dense(units=1, activation='sigmoid'))
  
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  ```

- **Convolutional Neural Networks (CNN) for Text**
  - CNNs can be applied to text for tasks like sentiment analysis.
  ```python
  from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
  
  model = Sequential()
  model.add(Embedding(input_dim=10000, output_dim=128))
  model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
  model.add(MaxPooling1D(pool_size=2))
  model.add(GlobalMaxPooling1D())
  model.add(Dense(units=1, activation='sigmoid'))
  
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  ```

#### 5. Sequence-to-Sequence Models
- **Encoder-Decoder Architecture**
  - Typically used in tasks like machine translation.
  ```python
  from keras.layers import Input, LSTM, Dense
  from keras.models import Model
  
  encoder_inputs = Input(shape=(None, num_encoder_tokens))
  encoder = LSTM(latent_dim, return_state=True)
  encoder_outputs, state_h, state_c = encoder(encoder_inputs)
  encoder_states = [state_h, state_c]
  
  decoder_inputs = Input(shape=(None, num_decoder_tokens))
  decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
  decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
  decoder_dense = Dense(num_decoder_tokens, activation='softmax')
  decoder_outputs = decoder_dense(decoder_outputs)
  
  model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
  ```

#### 6. Pretrained Models and Transfer Learning
- **Using Pretrained Embeddings**
  - Leveraging embeddings from models trained on large corpora.
  ```python
  glove_vectors = api.load("glove-wiki-gigaword-100")
  word_vector = glove_vectors['king']
  ```

- **Transfer Learning with Transformers**
  - Using models like BERT, GPT, etc., for various NLP tasks.
  ```python
  from transformers import pipeline
  
  summarizer = pipeline("summarization")
  text = "Long text to summarize."
  summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
  print(summary)
  ```

### Detailed Explanation of Important NLP Architectures

#### 1. Recurrent Neural Networks (RNN)
- **Overview**:
  - RNNs are designed to recognize patterns in sequences of data, such as time series or natural language.
  - They have loops in their architecture, allowing them to maintain a state that captures information about previous inputs in the sequence.
  
- **Architecture**:
  - Consists of an input layer, a hidden layer with recurrent connections, and an output layer.
  - The hidden state is updated at each time step using the current input and the previous hidden state.

```python
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128))
model.add(SimpleRNN(units=128))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

- **Interview Questions**:
  - **Q: What are the limitations of vanilla RNNs?**
    - **A**: Vanilla RNNs suffer from vanishing and exploding gradient problems, making it difficult to learn long-term dependencies in sequences.

#### 2. Long Short-Term Memory (LSTM)
- **Overview**:
  - LSTMs are a type of RNN designed to overcome the limitations of vanilla RNNs by introducing a more complex architecture that includes gates to control the flow of information.
  - They can capture long-term dependencies more effectively.

- **Architecture**:
  - Consists of a cell state and three types of gates: input gate, forget gate, and output gate.
  - These gates regulate the information that is added to or removed from the cell state.

```python
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128))
model.add(LSTM(units=128))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

- **Interview Questions**:
  - **Q: How do LSTMs address the vanishing gradient problem?**
    - **A**: LSTMs use cell states and gates to maintain and update the information over long sequences, allowing gradients to flow more effectively and mitigating the vanishing gradient problem.

#### 3. Gated Recurrent Unit (GRU)
- **Overview**:
  - GRUs are a simplified version of LSTMs that combine the input and forget gates into a single update gate and merge the cell state and hidden state.
  - They have fewer parameters than LSTMs and are faster to train while maintaining similar performance.

- **Architecture**:
  - Consists of update and reset gates that control the information flow.

```python
from keras.layers import GRU

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128))
model.add(GRU(units=128))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

- **Interview Questions**:
  - **Q: Compare LSTM and GRU in terms of architecture and performance.**
    - **A**: GRUs are simpler and faster to train than LSTMs as they have fewer gates and parameters. Both can effectively capture long-term dependencies, but LSTMs might be more suitable for more complex tasks requiring finer control over memory.

#### 4. Convolutional Neural Networks (CNN) for Text
- **Overview**:
  - CNNs can be used for text classification tasks by treating text as a sequence and applying convolutional filters to capture local features.
  - They are particularly effective for tasks like sentiment analysis.

- **Architecture**:
  - Consists of convolutional layers followed by pooling layers and dense layers.

```python
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(GlobalMaxPooling1D())
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

- **Interview Questions**:
  - **Q: Why use CNNs for text classification?**
    - **A**: CNNs are effective at capturing local dependencies in text through convolutional filters, making them suitable for tasks like sentiment analysis where local patterns (n-grams) are important.

#### 5. Sequence-to-Sequence Models
- **Overview**:
  - Sequence-to-sequence (seq2seq) models are used for tasks where the output is a sequence, such as machine translation, text summarization, and dialogue generation.
  - They consist of an encoder and a decoder, both of which are typically RNNs, LSTMs, or GRUs.

- **Architecture**:
  - The encoder processes the input sequence and converts it into a fixed-size context vector.
  - The decoder takes this context vector and generates the output sequence.

```python
from keras.layers import Input, LSTM, Dense
from keras.models import Model

# Define the encoder
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Define the decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
```

- **Interview Questions**:
  - **Q: How does a sequence-to-sequence model work for machine translation?**
    - **A**: In machine translation, the encoder processes the source language sentence and generates a context vector. The decoder then uses this context vector to generate the target language sentence, one word at a time.

#### 6. Transformer Models
- **Overview**:
  - Transformers are a type of model designed to handle sequential data with self-attention mechanisms. They excel at tasks such as translation, text summarization, and language modeling.
  - The transformer architecture eliminates recurrence and relies entirely on attention mechanisms to draw global dependencies between input and output.

- **Architecture**:
  - Consists of an encoder and a decoder, both composed of multiple layers of self-attention and feed-forward neural networks.

```python
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Hello, how are you?", return_tensors="tf")
outputs = model(inputs)

print(outputs)
```

- **Interview Questions**:
  - **Q: Explain the attention mechanism in transformers.**
    - **A**: The attention mechanism allows the model to weigh the importance of different words in a sentence when encoding or decoding each word, enabling it to focus on relevant parts of the input sequence for each output element.
  - **Q: What are the advantages of transformers over RNNs and LSTMs?**
    - **A**: Transformers can handle long-range dependencies more effectively, allow for parallel processing of data, and avoid issues like the vanishing gradient problem, making them faster and more scalable.

