## POS tagging

### This is the code I tried:
```python
text = \
"""
This course will cover the major techniques for mining and analyzing text data to discover interesting patterns, extract useful knowledge, and support decision making, with an emphasis on statistical approaches t
hat can be generally applied to arbitrary text data in any natural language with no or minimum human effort.

Detailed analysis of text data requires understanding of natural language text, which is known to be a difficult task for computers. However, a number of statistical approaches have been shown to work well for th
e "shallow" but robust analysis of text data for pattern finding and knowledge discovery. You will learn the basic concepts, principles, and major algorithms in text mining and their potential applications.
"""

from nltk.tag import pos_tag

tagged = pos_tag(text.split())
```

With freshly installed `nltk` (Ubuntu 14.04) I was presented with this error:
```
Traceback (most recent call last):
  File "./pos.py", line 15, in <module>
    tagged = pos_tag(text.split())
  File "/usr/lib/python2.7/dist-packages/nltk/tag/__init__.py", line 63, in pos_tag
    tagger = nltk.data.load(_POS_TAGGER)
  File "/usr/lib/python2.7/dist-packages/nltk/data.py", line 594, in load
    resource_val = pickle.load(_open(resource_url))
  File "/usr/lib/python2.7/dist-packages/nltk/data.py", line 673, in _open
    return find(path).open()
  File "/usr/lib/python2.7/dist-packages/nltk/data.py", line 455, in find
    raise LookupError(resource_not_found)
LookupError: 
**********************************************************************
  Resource 'taggers/maxent_treebank_pos_tagger/english.pickle' not
  found.  Please use the NLTK Downloader to obtain the resource:
  >>> nltk.download().
  Searched in:
    - '/home/wmigda/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/local/lib/nltk_data'
**********************************************************************
```

Great, it tells me that I am missing `taggers/maxent_treebank_pos_tagger/english.pickle` and I should use `nltk.download()` to obtain it.
So, I did this:
```
import nltk
nltk.download()
```
only to be presented with another error:
```
urllib2.HTTPError: HTTP Error 404: Not Found
```
Quick investigation revealed that the dead download link in the downloader GUI windows is to blame. Google gave me the replacement url: `http://www.nltk.org/nltk_data/index.xml`.
After the downloader pulled in the index of resources I selected `models->maxent_treebank_pos_tagger`.
Once it was saved on my disk the code I first tried worked.
So what did it do?
```
print tagged
```
gave this:
```
[('This', 'DT'), ('course', 'NN'), ('will', 'MD'), ('cover', 'VB'), ('the', 'DT'), ('major', 'JJ'), ('techniques', 'NNS'), ('for', 'IN'), ('mining', 'VBG'), ('and', 'CC'), ('analyzing', 'VBG'), ('text', 'NN'), ('data', 'NNS'), ('to', 'TO'), ('discover', 'VB'), ('interesting', 'JJ'), ('patterns,', 'NN'), ('extract', 'NN'), ('useful', 'NN'), ('knowledge,', 'NN'), ('and', 'CC'), ('support', 'NN'), ('decision', 'NN'), ('making,', 'NN'), ('with', 'IN'), ('an', 'DT'), ('emphasis', 'NN'), ('on', 'IN'), ('statistical', 'JJ'), ('approaches', 'NNS'), ('that', 'WDT'), ('can', 'MD'), ('be', 'VB'), ('generally', 'RB'), ('applied', 'VBN'), ('to', 'TO'), ('arbitrary', 'JJ'), ('text', 'NN'), ('data', 'NNS'), ('in', 'IN'), ('any', 'DT'), ('natural', 'JJ'), ('language', 'NN'), ('with', 'IN'), ('no', 'DT'), ('or', 'CC'), ('minimum', 'JJ'), ('human', 'NN'), ('effort.', 'NNP'), ('Detailed', 'NNP'), ('analysis', 'NN'), ('of', 'IN'), ('text', 'NN'), ('data', 'NNS'), ('requires', 'VBZ'), ('understanding', 'VBG'), ('of', 'IN'), ('natural', 'JJ'), ('language', 'NN'), ('text,', 'NN'), ('which', 'WDT'), ('is', 'VBZ'), ('known', 'VBN'), ('to', 'TO'), ('be', 'VB'), ('a', 'DT'), ('difficult', 'JJ'), ('task', 'NN'), ('for', 'IN'), ('computers.', 'NNP'), ('However,', 'NNP'), ('a', 'DT'), ('number', 'NN'), ('of', 'IN'), ('statistical', 'JJ'), ('approaches', 'NNS'), ('have', 'VBP'), ('been', 'VBN'), ('shown', 'VBN'), ('to', 'TO'), ('work', 'VB'), ('well', 'RB'), ('for', 'IN'), ('the', 'DT'), ('"shallow"', 'NN'), ('but', 'CC'), ('robust', 'RB'), ('analysis', 'VBZ'), ('of', 'IN'), ('text', 'NN'), ('data', 'NNS'), ('for', 'IN'), ('pattern', 'NN'), ('finding', 'VBG'), ('and', 'CC'), ('knowledge', 'NN'), ('discovery.', 'NNP'), ('You', 'NNP'), ('will', 'MD'), ('learn', 'VB'), ('the', 'DT'), ('basic', 'JJ'), ('concepts,', 'NN'), ('principles,', 'NN'), ('and', 'CC'), ('major', 'JJ'), ('algorithms', 'NNS'), ('in', 'IN'), ('text', 'NN'), ('mining', 'NN'), ('and', 'CC'), ('their', 'PRP$'), ('potential', 'JJ'), ('applications.', 'NNP')]
```
This is for starters - we applied POS to the text just tokenized by whitespace. We maybe would like to preprocess is a bit differently.

### Sentence tokenize
First, you need to dowload the `punkt` tokenizer with `nltk.download()` (`models->punkt`).
Then, the text can be broken into sentences with:
```
from nltk import sent_tokenize
sentences = sent_tokenize(text)
```

### ICU tokenize
TODO

### PTB tokenize
Penn Treebank tokenizer is provided by `nltk`. We can invoke it like this:
```
from nltk.tokenize import TreebankWordTokenizer
TreebankWordTokenizer().tokenize(sentence)
```

### Summary
```python
text = \
"""
This course will cover the major techniques for mining and analyzing text data to discover interesting patterns, extract useful knowledge, and support decision making, with an emphasis on statistical approaches that can be generally applied to arbitrary text data in any natural language with no or minimum human effort.

Detailed analysis of text data requires understanding of natural language text, which is known to be a difficult task for computers. However, a number of statistical approaches have been shown to work well for the "shallow" but robust analysis of text data for pattern finding and knowledge discovery. You will learn the basic concepts, principles, and major algorithms in text mining and their potential applications.
"""

print "Input text:", text, "\n"

from nltk import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.tag import pos_tag

tagged = [pos_tag(TreebankWordTokenizer().tokenize(sent)) for sent in sent_tokenize(text)]
print tagged
```
