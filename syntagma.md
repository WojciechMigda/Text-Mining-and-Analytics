### Syntagmatic relations / word associations

In this problem we have a collection of documents, each document represented by a single line of text.
Hence, the number of documents equals the number of lines.
```python
def read_doc():
    from sys import stdin
    return stdin.readlines()

text = read_doc()
```

The text might contain escape sequences, so we will get rid of those:
```python
def unescape(s):
    """
    http://stackoverflow.com/a/24519338/2003487
    """
    import re
    import codecs
    ESCAPE_SEQUENCE_RE = re.compile(r'''
    ( \\U........      # 8-digit hex escapes
    | \\u....          # 4-digit hex escapes
    | \\x..            # 2-digit hex escapes
    | \\[0-7]{1,3}     # Octal escapes
    | \\N\{[^}]+\}     # Unicode characters by name
    | \\[\\'"abfnrtv]  # Single-character escapes
    )''', re.UNICODE | re.VERBOSE)
    def decode_match(match):
        return codecs.decode(match.group(0), 'unicode-escape')
        pass

    return ESCAPE_SEQUENCE_RE.sub(decode_match, s)

text = [unescape(sent) for sent in text]
```
Then, we'll break each document by whitespace, also skipping empty documents
```python
from nltk.tokenize.regexp import WhitespaceTokenizer
text = [WhitespaceTokenizer().tokenize(sent) for sent in text if len(sent) > 0]
```
convert everything to lowercase
```python
text = [[token.lower() for token in sent] for sent in text]
```

and clean all words from characters which are not letters (except apostrophes)
```python
text = [[''.join(ch for ch in token if ch.isalpha() or ch == '\'') for token in sent] for sent in text]
```

remove all words which are shorter than 2 or longer than 35 characters:
```python
text = [[token for token in sent if len(token) >= 2 and len(token) <= 35] for sent in text]
```

Now, it's time to get rid of all stopwords. Conveniently, `nltk` itself provides such:
```python
from nltk.corpus import stopwords
text = [[token for token in sent if not token in set(stopwords.words('english'))] for sent in text]
```

Finally, we will apply Porter2 stemmer:
```python
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
text = [[stemmer.stem(token) for token in sent] for sent in text]
```

Having the preprocessed document data in hand we can gather the word and bigram frequencies.
For that I am going to use the `CountVectorizer` from `scikit-learn`.
However, `CountVectorizer` by itself will attempt to tokenize the input, so to skip that we will pass it a stub analyzer.
```python
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=1, analyzer=lambda x:x)
X = vect.fit_transform(text)
```
`X` is now a sparse [`csr_matrix`](http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.sparse.csr_matrix.html)
where columns correspond to words and rows to documents,
which holds corresponding word counts.
We can have a peek at the matrix contents and the word/column mappings:
```python
print X.toarray()
feature_names = vect.get_feature_names()
print feature_names
```

We will store the document-per-word and document-per-bigram counts in `FreqDist` provided by `nltk`
```python
from nltk.probability import FreqDist
wfd = FreqDist()
bfd = FreqDist()
```

Retrieving of document-per-word counts is simple with
[`getnnz()`](http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.sparse.csr_matrix.getnnz.html)
method of the csr matrix:
```python
from itertools import izip
for index, count in izip(range(X.shape[1]), X.getnnz(0)):
    wfd[index] = count
```

document-per-bigram counts are only slightly more difficult to collect:
```python
from itertools import combinations
for segment in X:
    for w1, w2 in combinations(segment.tocoo().col, 2):
        bfd[w1, w2] += 1
```
I've used a `tocoo` trick described [here](http://stackoverflow.com/a/4319087/2003487) but I haven't made any
measurements for this case myself.

The mutual information scoring function (with smoothing) looks like this:
```python
import math
def mutinf(Nab, (Na, Nb), N):
    log2 = lambda x: math.log(x, 2)
    PXa1 = (Na + 0.5) / (N + 1)
    PXb1 = (Nb + 0.5) / (N + 1)
    PXa0 = 1. - PXa1
    PXb0 = 1. - PXb1
    PXab11 = (Nab + 0.25) / (N + 1)

    PXab01 = PXb1 - PXab11
    PXab10 = PXa1 - PXab11
    PXab00 = PXa0 - PXab01
    return \
        PXab00 * log2(PXab00 / (PXa0 * PXb0)) + \
        PXab01 * log2(PXab01 / (PXa0 * PXb1)) + \
        PXab10 * log2(PXab10 / (PXa1 * PXb0)) + \
        PXab11 * log2(PXab11 / (PXa1 * PXb1))
```

Scores will be calculated by iterating the `bfd` counts:
```python
N_seg = len(text)
scores = [(tup, mutinf(bfd[tup], (wfd[tup[0]], wfd[tup[1]]), N_seg)) for tup in bfd]
```

Finally, we can print the top-20
```python
print [(tup[1], feature_names[tup[0][0]], feature_names[tup[0][1]]) for tup in sorted(scores, key=lambda t: (-t[1], t[0]))[:20]]
```
