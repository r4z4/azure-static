```python

```

Keeping with the theme of staying simple and conise, to augment our data - since we have a relatively very small dataset - we will turn to some simple techniques that were highlighted in a popular 2019 paper titld "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks". In the paper they introduce four simpe techniques to performing data augmentation, and we will utilize them all for our dataset.

One very important point to bring up is the attention paid to the issue of how much augmentation to apply. For our purposes here, we are mainly just exploring in order to get a sense of what the techniques do and how they can - in general - affect our data. If we were engagine with real data for real business solutions, it is important to test a variety of sample sizes and tune with various hyperparamters. There is a large section in the paper dedicated to the question of how many sentences or items (naug) to augment, and note that researchers promote trying several out.

"For smaller training sets, overfitting was more likely, so generating many augmented sentences yielded large performance boosts. For larger training sets, adding more than four augmented sentences per original sentence was unhelpful since models tend to generalize properly when large quantities of real data are available. (pg. 4)"

###### Table 3: Recommended usage parameters.

| Ntrain | α     | naug |
|--------|-------|------|
| 500    | 0.05  | 16   |
| 2,000  | 0.05  | 8    |
| 5,000  | 0.1   | 4    |
| More   | 0.1   | 4    |

##### Table 1: Sentences generated using EDA. SR: synonym replacement. RI: random insertion. RS: random swap. RD: random deletion.

| Operation | Sentence                                                                      |
|-----------|-------------------------------------------------------------------------------|
| None      | A sad, superior human comedy played out on the back roads of life.            |
| SR        | A lamentable, superior human comedy played out on the backward road of life.  |
| RI        | A sad, superior human comedy played out on funniness the back roads of life.  |
| RS        | A sad, superior human comedy played out on roads back the of life.            |
| RD        | A sad, superior human out on the roads of life.                               |
