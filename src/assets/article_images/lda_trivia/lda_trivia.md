```python
import numpy as np
import regex as re
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```


```python
df = pd.read_pickle('../trivia_classification/data/dataframes/trivia_qs_normalized.pkl')
df.shape
```




    (34460, 2)




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
      <th>Questions</th>
      <th>category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>what hollywood actor portrayed casanova in the...</td>
      <td>religion-faith</td>
    </tr>
    <tr>
      <th>1</th>
      <td>which of these is not a son of adam and eve</td>
      <td>religion-faith</td>
    </tr>
    <tr>
      <th>2</th>
      <td>noah sent these two birds out of the ark to se...</td>
      <td>religion-faith</td>
    </tr>
    <tr>
      <th>3</th>
      <td>jacob had 12 sons which of his wives/maids bor...</td>
      <td>religion-faith</td>
    </tr>
    <tr>
      <th>4</th>
      <td>which of these animals did the israelites wors...</td>
      <td>religion-faith</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df['Questions'].values
y = df['category'].values
```


```python
print(X)
```

    ['what hollywood actor portrayed casanova in the 2005 movie entitled casanova based on the life of the popular adventurer '
     'which of these is not a son of adam and eve '
     'noah sent these two birds out of the ark to search for land ' ...
     'on what kind of surface is the sport called bandy practiced '
     'what type of sport is enduro '
     'elements of which sport does the game called pickleball include ']



```python
target_names = np.unique(y)
```


```python
from sklearn.preprocessing import OrdinalEncoder
enc=OrdinalEncoder() 

# Encode categorical values
df['category_enc']=enc.fit_transform(df[['category']])

# Check encoding results in a crosstab
pd.crosstab(df['category'], df['category_enc'], margins=False)
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
      <th>category_enc</th>
      <th>0.0</th>
      <th>1.0</th>
      <th>2.0</th>
      <th>3.0</th>
      <th>4.0</th>
      <th>5.0</th>
      <th>6.0</th>
      <th>7.0</th>
      <th>8.0</th>
      <th>9.0</th>
      <th>10.0</th>
      <th>11.0</th>
      <th>12.0</th>
    </tr>
    <tr>
      <th>category</th>
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
      <th>animals</th>
      <td>1368</td>
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
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>geography</th>
      <td>0</td>
      <td>842</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>history</th>
      <td>0</td>
      <td>0</td>
      <td>1642</td>
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
      <th>literature</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1290</td>
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
      <th>movies</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4301</td>
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
      <th>music</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5582</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>people</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2746</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>religion-faith</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>639</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>science-technology</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2487</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>sports</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2840</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>television</th>
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
      <td>5232</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>video-games</th>
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
      <td>0</td>
      <td>600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>world</th>
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
      <td>0</td>
      <td>0</td>
      <td>4891</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = df['category_enc'].values
```


```python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(max_features=50,
                      min_df=8,
                      max_df=0.7,
                      stop_words=stopwords.words("english"))
cleaned = vec.fit_transform(X).toarray()
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!



```python
print(cleaned)
len(cleaned[3])
```

    [[0.64115294 0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]
     ...
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]]





    50




```python
X_train, X_test, y_train, y_test = train_test_split(cleaned, y, test_size=0.2, random_state=0)
```


```python
len(X_train)
```




    27568




```python
len(y_train)
```




    27568




```python
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)
```




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearDiscriminantAnalysis()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">LinearDiscriminantAnalysis</label><div class="sk-toggleable__content"><pre>LinearDiscriminantAnalysis()</pre></div></div></div></div></div>




```python
y_hat_lda_model = model.predict(X_test)
actual_and_lda_model_preds = pd.DataFrame({"Actual Category": y_test,
                                           "Predicted Category": y_hat_lda_model})
actual_and_lda_model_preds
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
      <th>Actual Category</th>
      <th>Predicted Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6887</th>
      <td>10.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>6888</th>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>6889</th>
      <td>12.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>6890</th>
      <td>1.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>6891</th>
      <td>12.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
<p>6892 rows × 2 columns</p>
</div>




```python
from sklearn import metrics
lda_model_rpt = pd.DataFrame(metrics.classification_report(y_test, y_hat_lda_model, output_dict=True)).transpose()
lda_model_rpt
```

    /root/.local/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /root/.local/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /root/.local/lib/python3.8/site-packages/sklearn/metrics/_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))





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
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>0.500000</td>
      <td>0.003906</td>
      <td>0.007752</td>
      <td>256.000000</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>0.150602</td>
      <td>0.150602</td>
      <td>0.150602</td>
      <td>166.000000</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>0.445545</td>
      <td>0.150000</td>
      <td>0.224439</td>
      <td>300.000000</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>0.098039</td>
      <td>0.018182</td>
      <td>0.030675</td>
      <td>275.000000</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>0.820175</td>
      <td>0.638952</td>
      <td>0.718310</td>
      <td>878.000000</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>0.827320</td>
      <td>0.597209</td>
      <td>0.693679</td>
      <td>1075.000000</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>0.255250</td>
      <td>0.275742</td>
      <td>0.265101</td>
      <td>573.000000</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>117.000000</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>0.163763</td>
      <td>0.730097</td>
      <td>0.267520</td>
      <td>515.000000</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>0.384306</td>
      <td>0.323181</td>
      <td>0.351103</td>
      <td>591.000000</td>
    </tr>
    <tr>
      <th>10.0</th>
      <td>0.774559</td>
      <td>0.595930</td>
      <td>0.673604</td>
      <td>1032.000000</td>
    </tr>
    <tr>
      <th>11.0</th>
      <td>0.416667</td>
      <td>0.384615</td>
      <td>0.400000</td>
      <td>117.000000</td>
    </tr>
    <tr>
      <th>12.0</th>
      <td>0.365915</td>
      <td>0.292879</td>
      <td>0.325348</td>
      <td>997.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.428903</td>
      <td>0.428903</td>
      <td>0.428903</td>
      <td>0.428903</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.400165</td>
      <td>0.320100</td>
      <td>0.316010</td>
      <td>6892.000000</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.521436</td>
      <td>0.428903</td>
      <td>0.441474</td>
      <td>6892.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#checking for the model accuracy using score method
model.fit(X_train, y_train).score(X_train, y_train)
```




    0.4252756819500871




```python
y_pred = model.predict(X_test)
```


```python
model = LinearDiscriminantAnalysis()
data_plot = model.fit(X_train, y_train).transform(X_train)

#create LDA plot
plt.figure()
colors = ['red', 'green', 'blue', 'orange', 'purple', 'violet', 'teal', 'maroon', 'lime', 'grey', 'aqua', 'gold', 'yellow', 'black']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], target_names):
    plt.scatter(data_plot[y_train == i, 0], data_plot[y_train == i, 1], alpha=.8, color=color,
                label=target_name)

#add legend to plot
plt.legend(loc='best', shadow=False, scatterpoints=1)

#display LDA plot
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.show()
```


![png](lda_trivia_files/lda_trivia_18_0.png)



```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train, y_train)
```

As we can see, PCA selected the components which would result in the highest spread (retain the most information) and not necessarily the ones which maximize the separation between classes.


```python
pca.explained_variance_ratio_
```




    array([0.06503106, 0.05712134])




```python
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y_train,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)
```




    <matplotlib.collections.PathCollection at 0x7f325d536640>




![png](lda_trivia_files/lda_trivia_22_1.png)



```python
data_plot[1,1]
```




    0.11961195950953266




```python
print(data_plot)
```

    [[ 1.41130353  1.13511287 -0.58189461 ...  1.29220416 -0.35522076
       0.49487595]
     [-1.05597864  0.11961196  1.01331247 ... -1.17341956  1.1449998
       1.04536396]
     [-0.57571158  0.08735004  0.58077694 ... -0.2778546   0.42449414
      -0.22682093]
     ...
     [-0.57571158  0.08735004  0.58077694 ... -0.2778546   0.42449414
      -0.22682093]
     [-1.32424386  0.12582777  1.68527725 ... -2.42901658 -0.36184077
       1.8126642 ]
     [-0.51144126  0.18091163  0.57832353 ... -0.94078746  0.97648786
      -0.40167008]]

