```python
import numpy as np
import regex as re
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from textwrap import wrap
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils
```

## Load Raw Data

There were a couple of steps that I took that I will spare us all the details of, but just to give you an idea here is a snippet of the data in its original form:

    ...
    ENTY:animal What predators exist on Antarctica ?
    DESC:manner How is energy created ?
    NUM:other What is the quantity of American soldiers still unaccounted for from the Vietnam war ?
    LOC:mount What was the highest mountain on earth before Mount Everest was discovered ?
    HUM:gr What Polynesian people inhabit New Zealand ?
    ...

& so I needed to perform some initial cleaning on the text data to transform it into this form, where we can pick up below:

    ...
    ENTY@@animal@@What predators exist on Antarctica@@?
    DESC@@manner@@How is energy created@@?
    NUM@@other@@What is the quantity of American soldiers still unaccounted for from the Vietnam war@@?
    LOC@@mount@@What was the highest mountain on earth before Mount Everest was discovered@@?
    HU@@gr@@What Polynesian people inhabit New Zealand@@?
    ...

Don't ask why I chose the delimiter.


```python
df1 = pd.read_csv('data/clean/processed/train_1000.txt', sep='@@')
df2 = pd.read_csv('data/clean/processed/train_2000.txt', sep='@@')
df3 = pd.read_csv('data/clean/processed/train_3000.txt', sep='@@')
df4 = pd.read_csv('data/clean/processed/train_4000.txt', sep='@@')
df5 = pd.read_csv('data/clean/processed/train_5500.txt', sep='@@')
df_test = pd.read_csv('data/clean/processed/test_100.txt', sep='@@')
```


```python
frames = [df1, df2, df3, df4, df5]

df = pd.concat(frames)
df.shape
```

    (15452, 4)


```python
df.head()
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
      <th>entity</th>
      <th>definition</th>
      <th>question</th>
      <th>punctuation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DESC</td>
      <td>manner</td>
      <td>How did serfdom develop in and then leave Russia</td>
      <td>?</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ENTY</td>
      <td>cremat</td>
      <td>What films featured the character Popeye Doyle</td>
      <td>?</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DESC</td>
      <td>manner</td>
      <td>How can I find a list of celebrities ' real names</td>
      <td>?</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ENTY</td>
      <td>animal</td>
      <td>What fowl grabs the spotlight after the Chines...</td>
      <td>?</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABBR</td>
      <td>exp</td>
      <td>What is the full form of .com</td>
      <td>?</td>
    </tr>
  </tbody>
</table>
</div>



## Clean / Preprocess Data

---
There are several steps that we need to take here. Many of them will depend on the type of data that we have & our end goal, though, too. For certain types of text we may or may not be interested in numerical values, so we may strip those out with a function. Or maybe we need to keep any punctuation around. In most cases we will remove any punctuation, but the idea is to always be thinking about your data and how you may need to adapt it for your specific use case to get the most out of it.

---
Some other alterations that are included in this step but may seem a little different are the more advanced linguistic techiques of stemming and lemmatization. We won't get into particulars here but the same idea applies, in that if you are to use these methods it is always good to review just what they are doing and why they may or may not be needed for our case. With that in mind, let's take a look and see what we should do here.

First things first, we're keeping things simple and only interested in two columns.


```python
df = df.drop(['definition','punctuation'], axis='columns')
```

I have two functions in my utils.py that do some text cleaning using a combination of the methods mentioned above. Here is what each of those looks like and the corresponding output:

```python
def clean_text(text, ):

    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

    def remove_special_characters(text, characters=string.punctuation.replace('-', '')):
        tokens = tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))

    def stem_text(text, stemmer=default_stemmer):
        tokens = tokenize_text(text)
        return ' '.join([stemmer.stem(t) for t in tokens])

    def remove_stopwords(text, stop_words=default_stopwords):
        tokens = [w for w in tokenize_text(text) if w not in stop_words]
        return ' '.join(tokens)

    text = text.strip(' ') # strip whitespaces
    text = text.lower() # lowercase
    text = stem_text(text) # stemming
    text = remove_special_characters(text) # remove punctuation and symbols
    text = remove_stopwords(text) # remove stopwords
    #text.strip(' ') # strip whitespaces again?

    return text
```

```python
def normalize_text(s):
    s = s.lower()
    
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W\s',' ',s)
    
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+',' ',s)
    
    return s
```


```python
df['question_normalized'] = [utils.normalize_text(s) for s in df['question']]
```


```python
df['question_cleaned'] = [utils.clean_text(s) for s in df['question']]
```


```python
df.head(20)
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
      <th>entity</th>
      <th>question</th>
      <th>question_normalized</th>
      <th>question_cleaned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DESC</td>
      <td>How did serfdom develop in and then leave Russia</td>
      <td>how did serfdom develop in and then leave russia</td>
      <td>serfdom develop leav russia</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ENTY</td>
      <td>What films featured the character Popeye Doyle</td>
      <td>what films featured the character popeye doyle</td>
      <td>film featur charact popey doyl</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DESC</td>
      <td>How can I find a list of celebrities ' real names</td>
      <td>how can i find a list of celebrities real names</td>
      <td>find list celebr real name</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ENTY</td>
      <td>What fowl grabs the spotlight after the Chines...</td>
      <td>what fowl grabs the spotlight after the chines...</td>
      <td>fowl grab spotlight chines year monkey</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABBR</td>
      <td>What is the full form of .com</td>
      <td>what is the full form of com</td>
      <td>full form com</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HUM</td>
      <td>What contemptible scoundrel stole the cork fro...</td>
      <td>what contemptible scoundrel stole the cork fro...</td>
      <td>contempt scoundrel stole cork lunch</td>
    </tr>
    <tr>
      <th>6</th>
      <td>HUM</td>
      <td>What team did baseball 's St. Louis Browns become</td>
      <td>what team did baseball s st louis browns become</td>
      <td>team basebal st loui brown becom</td>
    </tr>
    <tr>
      <th>7</th>
      <td>HUM</td>
      <td>What is the oldest profession</td>
      <td>what is the oldest profession</td>
      <td>oldest profess</td>
    </tr>
    <tr>
      <th>8</th>
      <td>DESC</td>
      <td>What are liver enzymes</td>
      <td>what are liver enzymes</td>
      <td>liver enzym</td>
    </tr>
    <tr>
      <th>9</th>
      <td>HUM</td>
      <td>Name the scar-faced bounty hunter of The Old West</td>
      <td>name the scar-faced bounty hunter of the old west</td>
      <td>name scar-fac bounti hunter old west</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NUM</td>
      <td>When was Ozzy Osbourne born</td>
      <td>when was ozzy osbourne born</td>
      <td>wa ozzi osbourn born</td>
    </tr>
    <tr>
      <th>11</th>
      <td>DESC</td>
      <td>Why do heavier objects travel downhill faster</td>
      <td>why do heavier objects travel downhill faster</td>
      <td>whi heavier object travel downhil faster</td>
    </tr>
    <tr>
      <th>12</th>
      <td>HUM</td>
      <td>Who was The Pride of the Yankees</td>
      <td>who was the pride of the yankees</td>
      <td>wa pride yanke</td>
    </tr>
    <tr>
      <th>13</th>
      <td>HUM</td>
      <td>Who killed Gandhi</td>
      <td>who killed gandhi</td>
      <td>kill gandhi</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ENTY</td>
      <td>What is considered the costliest disaster the ...</td>
      <td>what is considered the costliest disaster the ...</td>
      <td>consid costliest disast insur industri ha ever...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>LOC</td>
      <td>What sprawling U.S. state boasts the most airp...</td>
      <td>what sprawling u.s state boasts the most airports</td>
      <td>sprawl us state boast airport</td>
    </tr>
    <tr>
      <th>16</th>
      <td>DESC</td>
      <td>What did the only repealed amendment to the U....</td>
      <td>what did the only repealed amendment to the u....</td>
      <td>onli repeal amend us constitut deal</td>
    </tr>
    <tr>
      <th>17</th>
      <td>NUM</td>
      <td>How many Jews were executed in concentration c...</td>
      <td>how many jews were executed in concentration c...</td>
      <td>mani jew execut concentr camp dure wwii</td>
    </tr>
    <tr>
      <th>18</th>
      <td>DESC</td>
      <td>What is `` Nine Inch Nails ''</td>
      <td>what is nine inch nails '</td>
      <td>nine inch nail</td>
    </tr>
    <tr>
      <th>19</th>
      <td>DESC</td>
      <td>What is an annotated bibliography</td>
      <td>what is an annotated bibliography</td>
      <td>annot bibliographi</td>
    </tr>
  </tbody>
</table>
</div>



#### Creating a Document Term Matrix


```python
df_grouped=df[['entity','question_cleaned']].groupby(by='entity').agg(lambda x:' '.join(x))
df_grouped.head()
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
      <th>question_cleaned</th>
    </tr>
    <tr>
      <th>entity</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ABBR</th>
      <td>full form com doe abbrevi aid stand doe inri s...</td>
    </tr>
    <tr>
      <th>DESC</th>
      <td>serfdom develop leav russia find list celebr r...</td>
    </tr>
    <tr>
      <th>ENTY</th>
      <td>film featur charact popey doyl fowl grab spotl...</td>
    </tr>
    <tr>
      <th>HUM</th>
      <td>contempt scoundrel stole cork lunch team baseb...</td>
    </tr>
    <tr>
      <th>LOC</th>
      <td>sprawl us state boast airport highest waterfal...</td>
    </tr>
  </tbody>
</table>
</div>




```python
cv=CountVectorizer(analyzer='word')
data=cv.fit_transform(df_grouped['question_cleaned'])
df_dtm = pd.DataFrame(data.toarray(), columns=cv.get_feature_names_out())
df_dtm.index=df_grouped.index
df_dtm.head(6)
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
      <th>000</th>
      <th>000th</th>
      <th>007</th>
      <th>10</th>
      <th>100</th>
      <th>103</th>
      <th>11</th>
      <th>111</th>
      <th>118</th>
      <th>11th</th>
      <th>...</th>
      <th>zipper</th>
      <th>zitoni</th>
      <th>zodiac</th>
      <th>zolotow</th>
      <th>zon</th>
      <th>zone</th>
      <th>zoo</th>
      <th>zoolog</th>
      <th>zoonos</th>
      <th>zorro</th>
    </tr>
    <tr>
      <th>entity</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ABBR</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>DESC</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ENTY</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>HUM</th>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>LOC</th>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>NUM</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 6889 columns</p>
</div>



Just another good example of why doing these sometimes tedious tasks has value. Might want to come back and examine what exactly items like 000 and 000th are doing in the dataset. Might be indicative of larger issues, or just a one off that we need to drop. Also, I just want to make sure that my instinct of why 007 is in there holds true.


```python
# Function for generating word clouds
def generate_wordcloud(data,title):
  wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2").generate_from_frequencies(data)
  plt.figure(figsize=(10,8))
  plt.imshow(wc, interpolation='bilinear')
  plt.axis("off")
  plt.title('\n'.join(wrap(title,60)),fontsize=13)
  plt.show()
  
# Transposing document term matrix
df_dtm=df_dtm.transpose()

# Plotting word cloud for each product
for index,product in enumerate(df_dtm.columns):
  generate_wordcloud(df_dtm[product].sort_values(ascending=False),product)
```


![png](trec_eda_files/trec_eda_18_0.png)



![png](trec_eda_files/trec_eda_18_1.png)



![png](trec_eda_files/trec_eda_18_2.png)



![png](trec_eda_files/trec_eda_18_3.png)



![png](trec_eda_files/trec_eda_18_4.png)



![png](trec_eda_files/trec_eda_18_5.png)



```python
doe_stems = [
    'Jim does like oranges',
    'Jim does not like oranges',
    "Jim doesn't like oranges", 
    'Jim doe like oranges'
    ]
results = [utils.clean_text(s) for s in doe_stems]
print(results)
```

    ['jim doe like orang', 'jim doe like orang', 'jim doe nt like orang', 'jim doe like orang']


I'll be honest I do not actually use WordClouds all that often in practice, but in this case I think it helps us quite a bit. We already know not to expect too much from this dataset, but at least so far we can see that it makes sense. If we were really digging in we would want to address the "doe" stem and maybe find a way to differentiate that between the two different sets. Or at least it might warrant just examining the data and seeing where it appears, and maybe a strategy will emerge from there.


```python
from textblob import TextBlob
df['polarity']=df['question_cleaned'].apply(lambda x:TextBlob(x).sentiment.polarity)
df.head()
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
      <th>entity</th>
      <th>question</th>
      <th>question_normalized</th>
      <th>question_cleaned</th>
      <th>polarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DESC</td>
      <td>How did serfdom develop in and then leave Russia</td>
      <td>how did serfdom develop in and then leave russia</td>
      <td>serfdom develop leav russia</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ENTY</td>
      <td>What films featured the character Popeye Doyle</td>
      <td>what films featured the character popeye doyle</td>
      <td>film featur charact popey doyl</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DESC</td>
      <td>How can I find a list of celebrities ' real names</td>
      <td>how can i find a list of celebrities real names</td>
      <td>find list celebr real name</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ENTY</td>
      <td>What fowl grabs the spotlight after the Chines...</td>
      <td>what fowl grabs the spotlight after the chines...</td>
      <td>fowl grab spotlight chines year monkey</td>
      <td>-0.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ABBR</td>
      <td>What is the full form of .com</td>
      <td>what is the full form of com</td>
      <td>full form com</td>
      <td>0.35</td>
    </tr>
  </tbody>
</table>
</div>




```python
question_polarity_sorted=pd.DataFrame(df.groupby('entity')['polarity'].mean().sort_values(ascending=True))

plt.figure(figsize=(16,8))
plt.xlabel('Polarity')
plt.ylabel('Entities')
plt.title('Polarity of Different Question Entities from TREC Dataset')
polarity_graph=plt.barh(np.arange(len(question_polarity_sorted.index)),question_polarity_sorted['polarity'],color='orange',)

# Writing product names on bar
for bar,product in zip(polarity_graph,question_polarity_sorted.index):
  plt.text(0.005,bar.get_y()+bar.get_width(),'{}'.format(product),va='center',fontsize=11,color='black')

# Writing polarity values on graph
for bar,polarity in zip(polarity_graph,question_polarity_sorted['polarity']):
  plt.text(bar.get_width()+0.001,bar.get_y()+bar.get_width(),'%.3f'%polarity,va='center',fontsize=11,color='black')
  
plt.yticks([])
plt.show()
```


![png](trec_eda_files/trec_eda_22_0.png)



