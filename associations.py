# -*- coding: utf-8 -*-

def read_doc():
    from sys import stdin
    return stdin.readlines()

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

    return ESCAPE_SEQUENCE_RE.sub(decode_match, s)

import math
def mutinf(Nab, Na, Nb, N):
    PXa1 = (Na + 0.5) / (N + 1)
    PXb1 = (Nb + 0.5) / (N + 1)
    PXa0 = 1. - PXa1
    PXb0 = 1. - PXb1
    PXab11 = (Nab + 0.25) / (N + 1)

    PXab01 = PXb1 - PXab11
    PXab10 = PXa1 - PXab11
    PXab00 = PXa0 - PXab01
    return \
        PXab00 * math.log(PXab00 / (PXa0 * PXb0), 2) + \
        PXab01 * math.log(PXab01 / (PXa0 * PXb1), 2) + \
        PXab10 * math.log(PXab10 / (PXa1 * PXb0), 2) + \
        PXab11 * math.log(PXab11 / (PXa1 * PXb1), 2)


def main():
    text = read_doc()

    text = [unescape(sent) for sent in text]

    from nltk.tokenize.regexp import WhitespaceTokenizer
    ws_tokenizer = WhitespaceTokenizer()
    text = [ws_tokenizer.tokenize(sent) for sent in text if len(sent) > 0]

    text = [[token.lower() for token in sent] for sent in text]

    text = [[''.join(ch for ch in token if ch.isalpha() or ch == '\'') for token in sent] for sent in text]

    text = [[token for token in sent if len(token) >= 2 and len(token) <= 35] for sent in text]

    from nltk.corpus import stopwords
    stopwords = set(stopwords.words('english'))
    text = [[token for token in sent if not token in stopwords] for sent in text]

    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    text = [[stemmer.stem(token) for token in sent] for sent in text]

    from sklearn.feature_extraction.text import CountVectorizer
    vect = CountVectorizer(min_df=20, analyzer=lambda x:x)
    X = vect.fit_transform(text)

    #print(X.toarray())
    feature_names = vect.get_feature_names()
    #print(feature_names)

    from collections import Counter
    try:
        # Python 2
        from itertools import izip
    except ImportError:
        # Python 3
        izip = zip
    wfd = Counter({key: value for (key, value) in izip(range(X.shape[1]), X.getnnz(0))})

    from itertools import combinations, chain
    bfd = Counter(chain.from_iterable([combinations(sorted(segment.tocoo().col), 2) for segment in X]))

    N_seg = len(text)
    scores = [(mutinf(bfd[tup], wfd[tup[0]], wfd[tup[1]], N_seg), tup) for tup in bfd]

    print([(tup[0], feature_names[tup[1][0]], feature_names[tup[1][1]]) for tup in sorted(scores, reverse=True)[:20]])

    pass

if __name__ == "__main__":
    main()
    pass
