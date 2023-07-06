# Embedding Generation Run Using Mojo 🔥

---

A fair warning to anyone hoping to just pick right up and start adding to some existing Python code ... pretty close but not quite that easy. If you simply want to execute Python code you can add ```%%python``` to the top of the cell, but just be aware of the lack of compatability with Mojo code. From what I can tell though things will be getting smoother as time goes on. Much of this exercise here is just to jump in like I think many others would - taking some simple existing module and converting it. At least, this is the first thing that I reached for when I heard of the infamous '35,000x speed of Python'.

---

```mojo
from PythonInterface import Python
# Specify path to `mypython.py` module
# Python.add_to_path("path/to/module")
# let mypython = Python.import_module("mypython")
```

#### Here is my Python import

```
from utils import text_preprocessing, create_unique_word_dict
```


```mojo
let utils = Python.add_to_path("./utils.py")
```


```mojo
let np = Python.import_module("numpy")
```


```mojo
let line1: StringLiteral = "One of my projects is nearing its last stretch. I’m super tired and I need a day or two away from computers before I 
can handle that final stretch. So tomorrow I’m going gift shopping!"
```


```mojo
## Actually a good opportunity here. Need to print the type. Define type in .py and import it.
Python.add_to_path("./types.py")
let types = Python.import_module("types")
types.print_str(line1)
```

    Error: An error occurred in Python.



```mojo
%%python
print(type(line1))
```

    Traceback (most recent call last):
      File "<string>", line 6, in <module>
      File "<string>", line 5, in <module>
    NameError: name 'line1' is not defined
    Error: The Python expression raised an exception



```mojo
let line2: StringLiteral = "I would’ve loved loved LOVED some vacation but alas the stars have not aligned and there’s no way I can take a vacation now. 
However, I really needed a change of scenery to avoid the rapidly-approaching burnout, so I’m gonna spend the next week working from Chania, my second 
home. I’m gonna visit some friends there and I can’t show up empty-handed, so I’m probably gonna go hunt for some small gifts tomorrow, too. Nothing 
fancy, we have this routine where we each show up with homemade jam/brandy/whatever, but last year I also brought some local cheese and it was a blast, 
so I’ll see if I can pick up something else that’s good."
```

## What we get when try to just use String. Going to need to get used to StringLiteral and StringRef
```
error: Expression [18]:17:16: use of unknown declaration 'String'
```


```mojo
let line3: StringLiteral = "Then I’m off to watch the Champions League final. I missed most of this season (tough year…) and I don’t watch that much football 
these days, but these things are still kind of special to me because nostalgia is a hell of a drug. I missed only one final in the last 24 years, and I had a 
pretty good excuse for it (I, uh, got married on that day)."
```


```mojo
let line_list: ListLiteral[StringLiteral, StringLiteral, StringLiteral] = [line1, line2, line3]
```


```python
print(len(line_list))
```

    3



```mojo

```


### Looking At Some Errors

```python
%%python
filename = 'random_text.txt'
lines = open(filename).read().split("\n")
```


```mojo
def filter_empties(lines):
    for x in lines:
        ix if x
```

    error: Expression 'object' does not implement the '__iter__' method
    for x in lines:
             ^
    error: Expression expecting an 'else' followed by an expression
    if x
    ^

    expression failed to parse (no further compiler diagnostics)


```mojo
let texts = [x for x in lines if x]
```

    error: Expression expected ']' in list expression
    let texts = [x for x in lines if x]

    expression failed to parse (no further compiler diagnostics)