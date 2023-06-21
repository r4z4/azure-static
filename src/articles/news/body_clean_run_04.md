```python
import numpy as np
import json
import regex as re
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils
import seaborn as sns
import keras
import nltk
import random

from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from sklearn.naive_bayes import MultinomialNB
```

    2023-06-21 00:43:13.244196: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2023-06-21 00:43:13.868509: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2023-06-21 00:43:13.883636: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-06-21 00:43:16.466271: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



```python
# Load model
model_file = 'models/newsgroup_body_clean_model'
model = keras.models.load_model(model_file)
```

## Augmentation. Run 04 = Word Swap


```python
df = pd.read_pickle('../data/dataframes/newsgroup_body_cleaned_exploded.pkl')
```


```python
nan_rows = df[df['subject'].isnull()]
print(nan_rows.to_markdown(tablefmt="grid"))
```

    +-------------+-----------+
    | newsgroup   | subject   |
    +=============+===========+
    +-------------+-----------+



```python
df = df.dropna()
```


```python
def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

def swap_rejoin(x):
	if len(x) > 1:
		words = random_swap(x.split(), 1)
		sentence = ' '.join(words)
		return sentence
```


```python
df['subject'] = df['subject'].apply(lambda x: swap_rejoin(x))
print(df.sample(frac=1).reset_index(drop=True).head().to_markdown(tablefmt="grid"))
```

    +----+-------------+-------------------------------+
    |    | newsgroup   | subject                       |
    +====+=============+===============================+
    |  0 | politics    | unpatriot tax evas consid whi |
    +----+-------------+-------------------------------+
    |  1 | comp_elec   | triangul delaunay             |
    +----+-------------+-------------------------------+
    |  2 | comp_elec   | pb100 upgrad                  |
    +----+-------------+-------------------------------+
    |  3 | religion    | kick satan heaven biblic      |
    +----+-------------+-------------------------------+
    |  4 | seller      | mac forsal se                 |
    +----+-------------+-------------------------------+



```python
all_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
# We'll use all
target_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
```


```python
df = df.dropna()
```


```python
# container for sentences
X = np.array([s for s in df['subject']])
# container for sentences
y = np.array([n for n in df['newsgroup']])
```


```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(df['newsgroup'])
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.25)

classes = np.unique(y_train)
mapping = dict(zip(classes, target_categories))

len(X_train), len(X_test), classes, mapping
```




    (6167,
     2056,
     array([0, 1, 2, 3, 4, 5, 6]),
     {0: 'sport',
      1: 'autos',
      2: 'religion',
      3: 'comp_elec',
      4: 'sci_med',
      5: 'seller',
      6: 'politics'})




```python
# model parameters
vocab_size = 1200
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
```


```python
# tokenize sentences
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# convert train dataset to sequence and pad sequences
train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

# convert validation dataset to sequence and pad sequences
validation_sequences = tokenizer.texts_to_sequences(X_test)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)
```


```python
# fit model
num_epochs = 20
history = model.fit(train_padded, y_train, 
                    epochs=num_epochs, verbose=1,
                    validation_split=0.3)

# predict values
pred = model.predict(validation_padded)
```

    Epoch 1/20
      1/135 [..............................] - ETA: 1s - loss: 1.4896 - accuracy: 0.4062135/135 [==============================] - 1s 4ms/step - loss: 1.2952 - accuracy: 0.5324 - val_loss: 1.3645 - val_accuracy: 0.5116
    Epoch 2/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.2740 - accuracy: 0.5405 - val_loss: 1.3507 - val_accuracy: 0.5224
    Epoch 3/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.2549 - accuracy: 0.5521 - val_loss: 1.3376 - val_accuracy: 0.5267
    Epoch 4/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.2376 - accuracy: 0.5607 - val_loss: 1.3346 - val_accuracy: 0.5316
    Epoch 5/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.2158 - accuracy: 0.5660 - val_loss: 1.3172 - val_accuracy: 0.5419
    Epoch 6/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.1952 - accuracy: 0.5829 - val_loss: 1.3020 - val_accuracy: 0.5478
    Epoch 7/20
    135/135 [==============================] - 1s 6ms/step - loss: 1.1800 - accuracy: 0.5873 - val_loss: 1.2893 - val_accuracy: 0.5575
    Epoch 8/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.1578 - accuracy: 0.5973 - val_loss: 1.2743 - val_accuracy: 0.5673
    Epoch 9/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.1397 - accuracy: 0.5980 - val_loss: 1.2774 - val_accuracy: 0.5521
    Epoch 10/20
    135/135 [==============================] - 1s 5ms/step - loss: 1.1242 - accuracy: 0.6108 - val_loss: 1.2583 - val_accuracy: 0.5651
    Epoch 11/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.1049 - accuracy: 0.6193 - val_loss: 1.2447 - val_accuracy: 0.5732
    Epoch 12/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.0905 - accuracy: 0.6214 - val_loss: 1.2398 - val_accuracy: 0.5732
    Epoch 13/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.0743 - accuracy: 0.6284 - val_loss: 1.2272 - val_accuracy: 0.5829
    Epoch 14/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.0589 - accuracy: 0.6330 - val_loss: 1.2189 - val_accuracy: 0.5910
    Epoch 15/20
    135/135 [==============================] - 1s 4ms/step - loss: 1.0421 - accuracy: 0.6392 - val_loss: 1.2091 - val_accuracy: 0.5954
    Epoch 16/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.0315 - accuracy: 0.6425 - val_loss: 1.2039 - val_accuracy: 0.6002
    Epoch 17/20
    135/135 [==============================] - 0s 3ms/step - loss: 1.0166 - accuracy: 0.6485 - val_loss: 1.1987 - val_accuracy: 0.5986
    Epoch 18/20
    135/135 [==============================] - 1s 5ms/step - loss: 1.0047 - accuracy: 0.6541 - val_loss: 1.1930 - val_accuracy: 0.5954
    Epoch 19/20
    135/135 [==============================] - 1s 4ms/step - loss: 0.9912 - accuracy: 0.6571 - val_loss: 1.1879 - val_accuracy: 0.6002
    Epoch 20/20
    135/135 [==============================] - 0s 3ms/step - loss: 0.9769 - accuracy: 0.6601 - val_loss: 1.1897 - val_accuracy: 0.5981
    65/65 [==============================] - 0s 5ms/step



```python
import os

file_name = 'run_04'
plot_type = 'history'
model_name = 'newsgroup_body_clean'
#####
os.makedirs(f"images/{plot_type}", exist_ok=True)
os.makedirs(f"images/{plot_type}/{model_name}", exist_ok=True)
save_path = f'images/{plot_type}/{model_name}/{file_name}.png' 

utils.plot_history_and_save(history, save_path)
```


![png](clean_run_04_files/clean_run_04_16_0.png)


It does seem to be performing slightly better than our previous run, but this is still certainly nothing to write home about. Now, let's work with some real data and incorporate the body of these message into our next runs. We will save the model file incase we want to use it later in any form.


```python
# TensorFlow SavedModel format => .keras
model_file = 'models/newsgroup_body_clean_model'
model.save(model_file)
```

    WARNING:absl:Found untraced functions such as _update_step_xla while saving (showing 1 of 1). These functions will not be directly callable after loading.


    INFO:tensorflow:Assets written to: models/newsgroup_clean_model/assets


    INFO:tensorflow:Assets written to: models/newsgroup_clean_model/assets

