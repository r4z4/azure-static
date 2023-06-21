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

    2023-06-21 00:47:39.297891: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2023-06-21 00:47:39.898066: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2023-06-21 00:47:39.899722: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-06-21 00:47:43.555617: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



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
nan_rows = df[df['exploded_body'].isnull()]
print(nan_rows.to_markdown())
```

    | newsgroup   | body   | exploded_body   |
    |-------------|--------|-----------------|



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
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
df['exploded_body'] = df['exploded_body'].apply(lambda x: swap_rejoin(x))
print(df.sample(frac=1).reset_index(drop=True).loc[:,['newsgroup', 'exploded_body']].head().to_markdown())
```


<style>.container { width:100% !important; }</style>


    |    | newsgroup   | exploded_body                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
    |---:|:------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    |  0 | religion    | search illeg weapon also arrest warrant illeg weapon found thi case no-knock warrant wa call difficult flush gun toilet atf could surround compound mark polic car could driven entranc uniform offic could knock serv rcander usual way thi happen koresh refus warrant drove cop gun point tpg folk would kept quiet -- rod anderson nnzo think unit state govern boulder co respons fact bunch warrant nyxcsduedu fanat decid kill themselv satellit nnzo ao- slick willi compassion                                                                                                                                              |
    |  1 | politics    | took nth degre idea move peopl around abus often kill sole becaus ethnic abhorr thing especi troublesom area peopl differ ethnic group live side side long togeth think stand think wrong talk today troubl region wa happi see violenc croat muslim bosnia subsid thi morn think make progress front go serbian ethnic cleans qualit differ conflict within former yugoslavia part region q mr presid ani count good week presid tragedi waco defeat stimulu bill standoff bosnia wrong go differ look thing reassess laughter presid nt stimulu believ situat bosnia -- good week world nt know administr could made differ realli |
    |  2 | politics    | armenia azerbaijan two view articl may seassmuedu pt seassmuedu paul thompson schreiber post pt armenia azerbaijan two view pt pt washington report middl east -- pt aprilmay vol xi pt pt -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- affair -- -- -- pt pt pt life blockad yerevan pt -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- pt nanci najarian ms najarian wrote person observ somebodi wish counter realiti                                                                                                                                                                |
    |  3 | religion    | latest branch davidian articl apr genevarutgersedu conditt tsdarlututexasedu paul conditt wrote think realli sad mani peopl put faith mere man even claim son later andor prophet christ said hi return say desert say wilder believ lightn flash east west shall come son man paraphras - think vers somewher john jon -- -- -- -- -- -- -- -- sig file broken pleas tri god -- -- -- -- -- -- -- --                                                                                                                                                                                                                                |
    |  4 | politics    | frankli never met woman worth kill anyway ar- chrome barrel worth kill - much thi ha ruin caus recoveri near futur find martial come arm one help danger think crimin thi fault doe thi pinhead know someth rest us nt worri feder martial come get gun govern ca nt seem keep violent crimin jail sinc nt enough prison space legal system over- burden anyway go put million gun-own wo nt fork weapon mayb like volunt servic humbl abod sinc obvious feel sooooo strongli thi nd amend dead accept thi feder anoth way argument ha render useless accept thi find anoth                                                          |



```python
df.shape
```




    (69177, 3)




```python
bad_indices = df[df['exploded_body'].apply(lambda x: not isinstance(x, str))].index
df.drop(bad_indices, inplace = True)
```


```python
df.shape
```




    (69021, 3)




```python
all_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
# We'll use all
target_categories = ['sport', 'autos', 'religion', 'comp_elec', 'sci_med', 'seller', 'politics']
```


```python
# container for sentences
X = np.array([s for s in df['exploded_body']])
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




    (51765,
     17256,
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
    1133/1133 [==============================] - 5s 4ms/step - loss: 1.9307 - accuracy: 0.4637 - val_loss: 1.2464 - val_accuracy: 0.5558
    Epoch 2/20
    1133/1133 [==============================] - 4s 4ms/step - loss: 1.0477 - accuracy: 0.6299 - val_loss: 0.9275 - val_accuracy: 0.6737
    Epoch 3/20
    1133/1133 [==============================] - 4s 3ms/step - loss: 0.8007 - accuracy: 0.7205 - val_loss: 0.7495 - val_accuracy: 0.7422
    Epoch 4/20
    1133/1133 [==============================] - 4s 3ms/step - loss: 0.6566 - accuracy: 0.7731 - val_loss: 0.6498 - val_accuracy: 0.7768
    Epoch 5/20
    1133/1133 [==============================] - 4s 3ms/step - loss: 0.5652 - accuracy: 0.8048 - val_loss: 0.5812 - val_accuracy: 0.8042
    Epoch 6/20
    1133/1133 [==============================] - 4s 3ms/step - loss: 0.5019 - accuracy: 0.8275 - val_loss: 0.5361 - val_accuracy: 0.8182
    Epoch 7/20
    1133/1133 [==============================] - 4s 3ms/step - loss: 0.4568 - accuracy: 0.8440 - val_loss: 0.5053 - val_accuracy: 0.8303
    Epoch 8/20
    1133/1133 [==============================] - 4s 3ms/step - loss: 0.4222 - accuracy: 0.8570 - val_loss: 0.4783 - val_accuracy: 0.8408
    Epoch 9/20
    1133/1133 [==============================] - 4s 4ms/step - loss: 0.3953 - accuracy: 0.8666 - val_loss: 0.4639 - val_accuracy: 0.8476
    Epoch 10/20
    1133/1133 [==============================] - 4s 3ms/step - loss: 0.3747 - accuracy: 0.8737 - val_loss: 0.4501 - val_accuracy: 0.8508
    Epoch 11/20
    1133/1133 [==============================] - 4s 3ms/step - loss: 0.3570 - accuracy: 0.8800 - val_loss: 0.4369 - val_accuracy: 0.8554
    Epoch 12/20
    1133/1133 [==============================] - 4s 3ms/step - loss: 0.3431 - accuracy: 0.8835 - val_loss: 0.4397 - val_accuracy: 0.8589
    Epoch 13/20
    1133/1133 [==============================] - 4s 3ms/step - loss: 0.3307 - accuracy: 0.8885 - val_loss: 0.4251 - val_accuracy: 0.8625
    Epoch 14/20
    1133/1133 [==============================] - 4s 3ms/step - loss: 0.3200 - accuracy: 0.8925 - val_loss: 0.4242 - val_accuracy: 0.8610
    Epoch 15/20
    1133/1133 [==============================] - 4s 4ms/step - loss: 0.3100 - accuracy: 0.8982 - val_loss: 0.4180 - val_accuracy: 0.8654
    Epoch 16/20
    1133/1133 [==============================] - 4s 4ms/step - loss: 0.3023 - accuracy: 0.8988 - val_loss: 0.4145 - val_accuracy: 0.8669
    Epoch 17/20
    1133/1133 [==============================] - 4s 4ms/step - loss: 0.2944 - accuracy: 0.9026 - val_loss: 0.4216 - val_accuracy: 0.8618
    Epoch 18/20
    1133/1133 [==============================] - 4s 4ms/step - loss: 0.2889 - accuracy: 0.9040 - val_loss: 0.4137 - val_accuracy: 0.8693
    Epoch 19/20
    1133/1133 [==============================] - 4s 4ms/step - loss: 0.2819 - accuracy: 0.9070 - val_loss: 0.4158 - val_accuracy: 0.8685
    Epoch 20/20
    1133/1133 [==============================] - 4s 3ms/step - loss: 0.2765 - accuracy: 0.9082 - val_loss: 0.4122 - val_accuracy: 0.8706
    540/540 [==============================] - 1s 2ms/step



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


![png](clean_run_04_files/clean_run_04_18_0.png)


It does seem to be performing slightly better than our previous run, but this is still certainly nothing to write home about. Now, let's work with some real data and incorporate the body of these message into our next runs. We will save the model file incase we want to use it later in any form.


```python
# TensorFlow SavedModel format => .keras
model_file = 'models/newsgroup_body_clean_model'
model.save(model_file)
```

    WARNING:absl:Found untraced functions such as _update_step_xla while saving (showing 1 of 1). These functions will not be directly callable after loading.


    INFO:tensorflow:Assets written to: models/newsgroup_body_clean_model/assets


    INFO:tensorflow:Assets written to: models/newsgroup_body_clean_model/assets

