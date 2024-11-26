# Vector-Databases

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project-directory>
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the example code, execute the Jupyter Notebook or integrate the following code snippets into your project:

```python
!pip install faiss-cpu
!pip install sentence-transformers

import pandas as pd
df = pd.read_csv('/content/Tag.csv')
df
``````python
import spacy
nlp = spacy.load('en_core_web_sm')
stop_words = nlp.Defaults.stop_words

def preprocess(text):
  doc = nlp(str(text))
  preprocessed_text = []
  for token in doc:
    if token.is_punct or token.like_num or token in stop_words or token.is_space:
      continue
    preprocessed_text.append(token.lemma_.lower().strip())
  return ' '.join(preprocessed_text)

df['Processed Text'] = df['Text'].apply(preprocess)

df
``````python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

df['Embedding'] = df['Processed Text'].apply(model.encode)

vector = model.encode(df['Processed Text'])
```
``````python
dim = vector.shape[1]

import faiss
index = faiss.IndexFlatL2(dim)

index.add(vector)
```
``````python
search_query = 'I like eating cauliflower'
test_pre = preprocess(search_query)
encode_pre = model.encode(test_pre)
encode_pre.shape

#FAISS expects 2d array, so next step we are converting encode_pre to a 2D array
import numpy as np
svec = np.array(encode_pre).reshape(1,-1)


#We will get euclidean distance and index of the 2 nearest neighbours
distance,pos = index.search(svec,k=2)

df.Text.iloc[pos[0]]
```

## Requirements

- Python 3.7+
- Required libraries are listed in `requirements.txt`.

## Description

This project demonstrates how to utilize FAISS (Facebook AI Similarity Search) for vector similarity computations efficiently.
