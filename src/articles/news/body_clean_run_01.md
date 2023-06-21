```python
import numpy as np
import regex as re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
```

    2023-06-20 21:06:54.379258: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2023-06-20 21:06:55.007708: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2023-06-20 21:06:55.009763: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-06-20 21:06:56.977580: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



```python
import os
dirpath = '../data/clean_pkl/to_use/'
directory = os.fsencode(dirpath)
df = pd.DataFrame(columns=['newsgroup', 'body'])
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     name, ext = filename.split('.')
     if filename.endswith(".pkl"): 
         df_ = pd.read_pickle(dirpath + filename) 
         df = pd.merge(
            df, df_, how="outer"
        )
         continue
     else:
         continue
```


```python
df.shape
```

    (40590, 2)


```python
df = pd.read_pickle('../data/dataframes/newsgroup_body_cleaned.pkl')
```


```python
df.shape
```


    (35794, 2)


```python
from itertools import islice
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

# This will output lists of words, at most 100 each
def long_split(s):
    if len(s) > 100:
        tuple_list = list(chunk(s.split(), 100))
        return [' '.join(tups) for tups in tuple_list]
    
```


```python
df['exploded_body'] = df['body'].apply(lambda x: long_split(x))

```


```python
df['exploded_body'][3]
```


    ['analges diuret sometim see otc prepar muscl achesback ach combin aspirin diuret idea seem reduc inflamm get rid fluid doe thi actual work thank -larri c']


```python
df = df.explode('exploded_body')
```


```python
df.shape
```




    (72250, 3)



## Recheck our corpus

I am still trying get a better sense of what is happening here in the explode function to cause is to get None in the exploded_body col. In cases like this when I just always like to make sure I look over the data to make sure it is still correct. We'll eventually find out when we get our eval metrics but it is pretty clean if you just sample a few rows and make sure you don't have Wayne Gretzky articles under religion.  He was good though.


```python
df[df['exploded_body'].apply(lambda x: not isinstance(x, str))]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>newsgroup</th>
      <th>body</th>
      <th>exploded_body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>59</th>
      <td>sci_med</td>
      <td>twitch eye thi one time attribut lack sleep si...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>85</th>
      <td>sci_med</td>
      <td>open letter hillari rodham clinton  post one r...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>260</th>
      <td>sci_med</td>
      <td>erythromycin erythromycin effect treat pneumon...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>556</th>
      <td>sci_med</td>
      <td>foreskin troubl done short circumcis adult mal...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>637</th>
      <td>sci_med</td>
      <td>food-rel seizur articl apr spdcccom dyer spdcc...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>71885</th>
      <td>comp_elec</td>
      <td>doe ani one know biggest rom present pleas r...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>72110</th>
      <td>comp_elec</td>
      <td>doe ani one know biggest rom present pleas r...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>72151</th>
      <td>comp_elec</td>
      <td>card phone help understand cardphon oper val...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>72156</th>
      <td>comp_elec</td>
      <td>hd-tv sound system would like get inform cur...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>72178</th>
      <td>comp_elec</td>
      <td>whi circuit board green</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>3073 rows × 3 columns</p>
</div>




```python
bad_indices = df[df['exploded_body'].apply(lambda x: not isinstance(x, str))].index
```


```python
df.drop(bad_indices, inplace = True)
```


```python
df.shape
```




    (69177, 3)




```python
all_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
# We'll use all
target_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
```


```python
# container for sentences
X = np.array([t for t in df['exploded_body']])
# container for sentences
y = np.array([n for n in df['newsgroup']])
```


```python
encoder = LabelEncoder()
y = encoder.fit_transform(df['newsgroup'])
```


```python
y.shape
```




    (69177,)




```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.25)

classes = np.unique(y_train)
mapping = dict(zip(classes, target_categories))

len(X_train), len(X_test), classes, mapping
```




    (51882,
     17295,
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
### Final layer must be same as y.shape output -> # categories
```


```python
# model initialization
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(7, activation='sigmoid')
])
model._name = "NewsgroupBodyCleanBOW"
# compile model
# categorical-cross-entropy requires labels one-hot-encoded. sparse = as ints. binary = t/f
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# model summary
print(model.summary())
```

    Model: "NewsgroupBodyCleanBOW"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding_1 (Embedding)     (None, 120, 16)           19200     
                                                                     
     global_average_pooling1d_1   (None, 16)               0         
     (GlobalAveragePooling1D)                                        
                                                                     
     dense_2 (Dense)             (None, 24)                408       
                                                                     
     dense_3 (Dense)             (None, 7)                 175       
                                                                     
    =================================================================
    Total params: 19,783
    Trainable params: 19,783
    Non-trainable params: 0
    _________________________________________________________________
    None



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
    1135/1135 [==============================] - 5s 3ms/step - loss: 1.3570 - accuracy: 0.4956 - val_loss: 0.9589 - val_accuracy: 0.6743
    Epoch 2/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.7806 - accuracy: 0.7425 - val_loss: 0.6945 - val_accuracy: 0.7727
    Epoch 3/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.6152 - accuracy: 0.8015 - val_loss: 0.5928 - val_accuracy: 0.8094
    Epoch 4/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.5274 - accuracy: 0.8303 - val_loss: 0.5361 - val_accuracy: 0.8285
    Epoch 5/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.4730 - accuracy: 0.8470 - val_loss: 0.4974 - val_accuracy: 0.8406
    Epoch 6/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.4379 - accuracy: 0.8571 - val_loss: 0.4825 - val_accuracy: 0.8468
    Epoch 7/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.4147 - accuracy: 0.8647 - val_loss: 0.4677 - val_accuracy: 0.8490
    Epoch 8/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3959 - accuracy: 0.8697 - val_loss: 0.4553 - val_accuracy: 0.8510
    Epoch 9/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3811 - accuracy: 0.8736 - val_loss: 0.4470 - val_accuracy: 0.8536
    Epoch 10/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3691 - accuracy: 0.8787 - val_loss: 0.4403 - val_accuracy: 0.8563
    Epoch 11/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3581 - accuracy: 0.8806 - val_loss: 0.4384 - val_accuracy: 0.8545
    Epoch 12/20
    1135/1135 [==============================] - 4s 4ms/step - loss: 0.3484 - accuracy: 0.8850 - val_loss: 0.4363 - val_accuracy: 0.8566
    Epoch 13/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3399 - accuracy: 0.8865 - val_loss: 0.4303 - val_accuracy: 0.8583
    Epoch 14/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3325 - accuracy: 0.8889 - val_loss: 0.4267 - val_accuracy: 0.8605
    Epoch 15/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3253 - accuracy: 0.8917 - val_loss: 0.4265 - val_accuracy: 0.8649
    Epoch 16/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3197 - accuracy: 0.8935 - val_loss: 0.4240 - val_accuracy: 0.8630
    Epoch 17/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3133 - accuracy: 0.8960 - val_loss: 0.4225 - val_accuracy: 0.8639
    Epoch 18/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3081 - accuracy: 0.8981 - val_loss: 0.4207 - val_accuracy: 0.8659
    Epoch 19/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.3029 - accuracy: 0.9004 - val_loss: 0.4205 - val_accuracy: 0.8666
    Epoch 20/20
    1135/1135 [==============================] - 4s 3ms/step - loss: 0.2981 - accuracy: 0.9015 - val_loss: 0.4228 - val_accuracy: 0.8671
    541/541 [==============================] - 1s 2ms/step



```python
import os

file_name = 'run_01'
plot_type = 'history'
model_name = 'newsgroup_body_clean'
#####
os.makedirs(f"images/{plot_type}", exist_ok=True)
os.makedirs(f"images/{plot_type}/{model_name}", exist_ok=True)
save_path = f'images/{plot_type}/{model_name}/{file_name}.png' 

utils.plot_history_and_save(history, save_path)
```


![png](clean_run_01_files/clean_run_01_26_0.png)



```python
# TensorFlow SavedModel format => .keras
model_file = 'models/newsgroup_body_clean_model'
model.save(model_file)
```

    WARNING:absl:Found untraced functions such as _update_step_xla while saving (showing 1 of 1). These functions will not be directly callable after loading.


    INFO:tensorflow:Assets written to: models/newsgroup_body_clean_model/assets


    INFO:tensorflow:Assets written to: models/newsgroup_body_clean_model/assets


That just warms me up a bit. Now, lets take a look at what some augmentation can do. I am not expecting too much of a change here to be honest, and am actually curious if we will experience any setbacks and what some of the initial runs are like in the epochs.
