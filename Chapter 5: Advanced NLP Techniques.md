### Chapter 5: Advanced NLP Techniques

#### 1. BERT (Bidirectional Encoder Representations from Transformers)
- **Overview**:
  - BERT is a transformer-based model designed to pretrain deep bidirectional representations by jointly conditioning on both left and right context in all layers.
  - It achieves state-of-the-art performance on various NLP tasks.

- **Architecture**:
  - BERT uses a multi-layer bidirectional Transformer encoder.
  - It is pretrained on two tasks: masked language modeling and next sentence prediction.

```python
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
outputs = model(inputs)

print(outputs)
```

- **Interview Questions**:
  - **Q: How does BERT handle bidirectional context?**
    - **A**: BERT uses a masked language model objective during pretraining, where some percentage of the input tokens are masked at random, and the model learns to predict those masked tokens based on the context provided by the surrounding tokens. This allows BERT to capture bidirectional context.

#### 2. GPT (Generative Pretrained Transformer)
- **Overview**:
  - GPT is a unidirectional transformer model designed to generate human-like text.
  - It is pretrained on a large corpus of text using a language modeling objective and fine-tuned on specific tasks.

- **Architecture**:
  - GPT uses a stack of transformer decoder layers with self-attention mechanisms.
  - It generates text by predicting the next word in a sequence.

```python
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2LMHeadModel.from_pretrained('gpt2')

input_ids = tokenizer.encode('Once upon a time', return_tensors='tf')
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

- **Interview Questions**:
  - **Q: What is the main difference between BERT and GPT?**
    - **A**: BERT is a bidirectional model designed for understanding and generating context-aware embeddings for NLP tasks, while GPT is a unidirectional model focused on text generation by predicting the next word in a sequence.

#### 3. XLNet
- **Overview**:
  - XLNet combines the best of both BERT and GPT by using a permutation-based training method, capturing bidirectional context while avoiding issues related to masked language modeling.

- **Architecture**:
  - XLNet uses a Transformer-XL architecture with a segment-level recurrence mechanism and relative positional encoding to capture longer dependencies.

```python
from transformers import XLNetTokenizer, TFXLNetForSequenceClassification

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = TFXLNetForSequenceClassification.from_pretrained('xlnet-base-cased')

inputs = tokenizer("This is a test", return_tensors="tf")
outputs = model(inputs)

print(outputs)
```

- **Interview Questions**:
  - **Q: How does XLNet improve over BERT?**
    - **A**: XLNet uses permutation language modeling to capture bidirectional context without corrupting input data with masked tokens. It also incorporates segment recurrence and relative positional encoding to model long-range dependencies more effectively.

#### 4. T5 (Text-to-Text Transfer Transformer)
- **Overview**:
  - T5 is a unified framework that converts all NLP tasks into a text-to-text format, where both inputs and outputs are text strings.
  - It simplifies the process of multitask learning and transfer learning.

- **Architecture**:
  - T5 uses an encoder-decoder transformer architecture, treating every NLP problem as a text generation task.

```python
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = TFT5ForConditionalGeneration.from_pretrained('t5-small')

input_text = "translate English to French: The house is wonderful."
inputs = tokenizer(input_text, return_tensors="tf")
outputs = model.generate(inputs['input_ids'], max_length=40)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

- **Interview Questions**:
  - **Q: What is the main advantage of T5's text-to-text framework?**
    - **A**: T5's text-to-text framework allows for a consistent approach to handling various NLP tasks, making it easier to apply transfer learning and multitask learning, as the model architecture and training process remain the same across different tasks.

#### 5. RoBERTa (A Robustly Optimized BERT Pretraining Approach)
- **Overview**:
  - RoBERTa builds on BERT by optimizing its pretraining methodology, using more data, larger batch sizes, and longer training times to improve performance.

- **Architecture**:
  - Similar to BERT, RoBERTa uses a multi-layer bidirectional transformer encoder but with improvements in training procedures.

```python
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = TFRobertaForSequenceClassification.from_pretrained('roberta-base')

inputs = tokenizer("This is another test", return_tensors="tf")
outputs = model(inputs)

print(outputs)
```

- **Interview Questions**:
  - **Q: How does RoBERTa improve upon BERT?**
    - **A**: RoBERTa improves upon BERT by using a larger dataset, removing the next sentence prediction objective, using dynamic masking, and training with larger batch sizes and longer sequences, leading to better performance on downstream tasks.
