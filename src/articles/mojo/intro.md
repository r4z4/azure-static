# Mojo 🔥 Intro

---

I recently came across the Mojo languages and pretty quickly was drawn to it for a couple of reasons. First, it has a lot of new ideas (often remixed from existing ideas) that are exciting. It also has a lot of promise, and from what I gather, a good direction and overall goal. Finally, yes, it uses a fire emoji for the file extension (you can also just use .mojo, but why would you). Go with .🔥

It is a very new platform and language that was recently lauched, and so currently the only way to give it a try is to use the Mojo Playground. I signed up for a trial and was able to start using it the next day. There is certainly a lot of material to go over and so much that will change and go away and be added, but there are few rather larger ideas and overarching things to keep in mind. Again, these are also probably just a short list of a what is actually out there, but it is just what I have been able to pick up on so far.

---

First, Mojo is the language, Modular is the infrastructure and company. Mojo is a superset of Python. There is a lot that goes into that, but overall it means it works with current Python code and adds to it, in varying ways that often require very careful and deliberate design choices. We'll take a look at some of those here and this will be the palce that I will return to to add when we encounter some new ones or find out an error that we have made.

---

It is also worth noting how to do the first thing that probably everyone will try.

```python
from PythonInterface import Python

# This is equivalent to Python's `import numpy as np`
let np = Python.import_module("numpy")

# Now use numpy as if writing in Python
a = np.array([1, 2, 3])
print(a)
```

It seemed like a good idea to at least have a place for me to return to to keep a more organized and structured set of notes for this thing. There is a lot going on here, so having this plus the ability to adapt Mojo as needed will be a nice way to ease into it.
The beauty of the project and this approach is that you can really start to focus on the areas where you feel the most comfortable and have the most experience. For the sake of learning the overall convepts though I like to start on the other end and explore ideas
that I am not as familar with.  I have only done a little bit of C++ in my life and very little compiler work in general, and so that is where I will be starting.

---

```python
def example(inout a: Int, b: Int, c):
    # b and c use value semantics so they're mutable in the function
    ...

fn example(inout a: Int, b_in: Int, c_in: Object):
    # b_in and c_in are immutable references, so we make mutable shadow copies
    var b = b_in
    var c = c_in
    ...
```

### But, we actually DO need to get something going though and make sure  we understand the basics before we can even do that, though.


---

#### One other quick note is the concept of mutability & mutable arguments (inout)
---

If you define an fn function and want an argument to be mutable (so that changes to the argument inside the function are visible outside the function), you must declare the argument as mutable with the inout keyword.

---
