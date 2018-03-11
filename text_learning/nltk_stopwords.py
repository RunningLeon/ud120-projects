#!/usr/bin/python
###import nltk package and find how many stopwords
from nltk.corpus import stopwords
# import nltk
# nltk.download()

sw = stopwords.words("english")
print len(sw)
# for s in sw :
#     print s
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer( "english")
print stemmer.stem( "responsiveness" )
print stemmer.stem("germany")
