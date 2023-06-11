```python
import pandas as pd
import numpy as np
import json
import re
```


```python
df = pd.read_json("trivia_data.json")
```


```python
## Gonna take a while to embed all the data (> 2hrs on CPU). Lets just use 20% of the data
# n = 20
# df = df.head(int(len(df)*(n/100)))

data = df['corrected_question']
data_array = np.array([e for e in df['corrected_question']])
```


```python
len(data)
```




    4000




```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(data, show_progress_bar=True)
```


    Batches:   0%|          | 0/125 [00:00<?, ?it/s]



```python
!pip install umap-learn
```

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    Found existing installation: umap 0.1.1
    Uninstalling umap-0.1.1:
      Successfully uninstalled umap-0.1.1
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mhuggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    Requirement already satisfied: umap-learn in /usr/local/lib/python3.8/site-packages (0.5.3)
    Requirement already satisfied: numpy>=1.17 in /root/.local/lib/python3.8/site-packages (from umap-learn) (1.23.5)
    Requirement already satisfied: scikit-learn>=0.22 in /root/.local/lib/python3.8/site-packages (from umap-learn) (1.2.2)
    Requirement already satisfied: scipy>=1.0 in /root/.local/lib/python3.8/site-packages (from umap-learn) (1.10.1)
    Requirement already satisfied: numba>=0.49 in /usr/local/lib/python3.8/site-packages (from umap-learn) (0.57.0)
    Requirement already satisfied: pynndescent>=0.5 in /usr/local/lib/python3.8/site-packages (from umap-learn) (0.5.10)
    Requirement already satisfied: tqdm in /root/.local/lib/python3.8/site-packages (from umap-learn) (4.65.0)
    Requirement already satisfied: llvmlite<0.41,>=0.40.0dev0 in /usr/local/lib/python3.8/site-packages (from numba>=0.49->umap-learn) (0.40.1rc1)
    Requirement already satisfied: importlib-metadata in /root/.local/lib/python3.8/site-packages (from numba>=0.49->umap-learn) (6.6.0)
    Requirement already satisfied: joblib>=0.11 in /root/.local/lib/python3.8/site-packages (from pynndescent>=0.5->umap-learn) (1.2.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /root/.local/lib/python3.8/site-packages (from scikit-learn>=0.22->umap-learn) (3.1.0)
    Requirement already satisfied: zipp>=0.5 in /root/.local/lib/python3.8/site-packages (from importlib-metadata->numba>=0.49->umap-learn) (3.15.0)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m


```python
import umap.umap_ as umap
umap_embeddings = umap.UMAP(n_neighbors=15, 
                            n_components=5, 
                            metric='cosine').fit_transform(embeddings)
```

    /usr/local/lib/python3.8/site-packages/umap/distances.py:1063: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
      @numba.jit()
    /usr/local/lib/python3.8/site-packages/umap/distances.py:1071: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
      @numba.jit()
    /usr/local/lib/python3.8/site-packages/umap/distances.py:1086: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
      @numba.jit()
    /usr/local/lib/python3.8/site-packages/umap/umap_.py:660: NumbaDeprecationWarning: The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.
      @numba.jit()



```python
!pip install hdbscan
```

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    Collecting hdbscan
      Downloading hdbscan-0.8.29.tar.gz (5.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.2/5.2 MB[0m [31m4.0 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25h  Installing build dependencies ... [?25ldone
    [?25h  Getting requirements to build wheel ... [?25ldone
    [?25h  Preparing metadata (pyproject.toml) ... [?25ldone
    [?25hCollecting cython>=0.27 (from hdbscan)
      Using cached Cython-0.29.35-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_24_x86_64.whl (2.0 MB)
    Requirement already satisfied: numpy>=1.20 in /root/.local/lib/python3.8/site-packages (from hdbscan) (1.23.5)
    Requirement already satisfied: scipy>=1.0 in /root/.local/lib/python3.8/site-packages (from hdbscan) (1.10.1)
    Requirement already satisfied: scikit-learn>=0.20 in /root/.local/lib/python3.8/site-packages (from hdbscan) (1.2.2)
    Requirement already satisfied: joblib>=1.0 in /root/.local/lib/python3.8/site-packages (from hdbscan) (1.2.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /root/.local/lib/python3.8/site-packages (from scikit-learn>=0.20->hdbscan) (3.1.0)
    Building wheels for collected packages: hdbscan
      Building wheel for hdbscan (pyproject.toml) ... [?25ldone
    [?25h  Created wheel for hdbscan: filename=hdbscan-0.8.29-cp38-cp38-linux_x86_64.whl size=3716696 sha256=22809fd8fed458ccf9855d5c1d69b02592fe79f1232dd0737c531ffcb73ce273
      Stored in directory: /root/.cache/pip/wheels/76/06/48/527e038689c581cc9e519c73840efdc7473805149e55bd7ffd
    Successfully built hdbscan
    Installing collected packages: cython, hdbscan
    Successfully installed cython-0.29.35 hdbscan-0.8.29
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m


```python
import hdbscan
cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(umap_embeddings)
```


```python
import matplotlib.pyplot as plt

# Prepare data
umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
result = pd.DataFrame(umap_data, columns=['x', 'y'])
result['labels'] = cluster.labels_

# Visualize clusters
fig, ax = plt.subplots(figsize=(20, 10))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
plt.colorbar()
```




    <matplotlib.colorbar.Colorbar at 0x7f386ad9a940>




![png](topic_modeling_json_files/topic_modeling_json_9_1.png)



```python
docs_df = pd.DataFrame(data_array, columns=["Doc"])
docs_df['Topic'] = cluster.labels_
docs_df['Doc_ID'] = range(len(docs_df))
docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
```


```python
docs_per_topic.head()
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
      <th>Topic</th>
      <th>Doc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>Name the mascot of Austin College ? What was t...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Name the F1 racer with relative as Ralf Schuma...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Which country's largest city is Lima? Which st...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>How many races have the horses bred by Jacques...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>Give me all tv shows which are based in boston...</td>
    </tr>
  </tbody>
</table>
</div>




```python
docs_df['Topic'].value_counts()
```




    -1     1338
     44     203
     17     203
     21     194
     29     137
     19     102
     38      96
     11      95
     39      86
     9       75
     33      73
     13      71
     45      70
     14      70
     35      67
     4       66
     10      64
     41      64
     7       64
     12      56
     42      54
     43      47
     36      47
     15      44
     23      42
     5       40
     26      38
     30      36
     8       34
     24      34
     37      33
     6       29
     3       28
     27      28
     0       27
     34      27
     2       26
     32      25
     18      21
     20      20
     28      19
     1       19
     22      19
     31      18
     40      18
     16      17
     25      16
    Name: Topic, dtype: int64




```python
docs_df.head()
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
      <th>Doc</th>
      <th>Topic</th>
      <th>Doc_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>How many movies did Stanley Kubrick direct?</td>
      <td>19</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Which city's foundeer is John Forbes?</td>
      <td>44</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What is the river whose mouth is in deadsea?</td>
      <td>38</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What is the allegiance of John Kotelawala ?</td>
      <td>44</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>How many races have the horses bred by Jacques...</td>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count
  
tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(data))
```


```python
def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes

top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)
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
      <th>Topic</th>
      <th>Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>1338</td>
    </tr>
    <tr>
      <th>45</th>
      <td>44</td>
      <td>203</td>
    </tr>
    <tr>
      <th>18</th>
      <td>17</td>
      <td>203</td>
    </tr>
    <tr>
      <th>22</th>
      <td>21</td>
      <td>194</td>
    </tr>
    <tr>
      <th>30</th>
      <td>29</td>
      <td>137</td>
    </tr>
    <tr>
      <th>20</th>
      <td>19</td>
      <td>102</td>
    </tr>
    <tr>
      <th>39</th>
      <td>38</td>
      <td>96</td>
    </tr>
    <tr>
      <th>12</th>
      <td>11</td>
      <td>95</td>
    </tr>
    <tr>
      <th>40</th>
      <td>39</td>
      <td>86</td>
    </tr>
    <tr>
      <th>10</th>
      <td>9</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>




```python
top_n_words[21][:10]
```




    [('television', 0.25199257520408436),
     ('tv', 0.2272489906622773),
     ('shows', 0.17306120187797722),
     ('network', 0.0945769606612006),
     ('theme', 0.08517609400440743),
     ('company', 0.08181918094364625),
     ('producer', 0.07445065395128529),
     ('executive', 0.05932842541421376),
     ('broadcast', 0.057600770775431076),
     ('series', 0.0476476709374204)]




```python
from sklearn.metrics.pairwise import cosine_similarity
for i in range(20):
    # Calculate cosine similarity
    similarities = cosine_similarity(tf_idf.T)
    np.fill_diagonal(similarities, 0)

    # Extract label to merge into and from where
    topic_sizes = docs_df.groupby(['Topic']).count().sort_values("Doc", ascending=False).reset_index()
    topic_to_merge = topic_sizes.iloc[-1].Topic
    topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1

    # Adjust topics
    docs_df.loc[docs_df.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
    old_topics = docs_df.sort_values("Topic").Topic.unique()
    map_topics = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}
    docs_df.Topic = docs_df.Topic.map(map_topics)
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})

    # Calculate new topic words
    m = len(data)
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m)
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)

topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)
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
      <th>Topic</th>
      <th>Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>18506</td>
    </tr>
    <tr>
      <th>47</th>
      <td>46</td>
      <td>1289</td>
    </tr>
    <tr>
      <th>52</th>
      <td>51</td>
      <td>624</td>
    </tr>
    <tr>
      <th>51</th>
      <td>50</td>
      <td>593</td>
    </tr>
    <tr>
      <th>48</th>
      <td>47</td>
      <td>361</td>
    </tr>
    <tr>
      <th>17</th>
      <td>16</td>
      <td>312</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>232</td>
    </tr>
    <tr>
      <th>12</th>
      <td>11</td>
      <td>223</td>
    </tr>
    <tr>
      <th>50</th>
      <td>49</td>
      <td>221</td>
    </tr>
    <tr>
      <th>28</th>
      <td>27</td>
      <td>216</td>
    </tr>
  </tbody>
</table>
</div>




```python
top_n_words[51][:10]
```




    [('trump', 0.10467902584887456),
     ('president', 0.06072602666892069),
     ('2020', 0.03411193922678639),
     ('america', 0.03269916901768407),
     ('democratic', 0.032277888078611434),
     ('donald', 0.029263497562009327),
     ('democrats', 0.0268655653148694),
     ('election', 0.02609372385412432),
     ('presidential', 0.025912575696792114),
     ('bernie', 0.025237479236590536)]




```python
top_n_words[50][:10]
```




    [('don', 0.03970939022232294),
     ('people', 0.03339189616992504),
     ('anxiety', 0.03049151218674598),
     ('life', 0.023705264451986674),
     ('mental', 0.023679815071311398),
     ('doesn', 0.02318471421412793),
     ('disorder', 0.02080708641397244),
     ('need', 0.01934262579411308),
     ('like', 0.01924398264657584),
     ('just', 0.019145351423775627)]


