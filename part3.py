# -*- coding: utf-8 -*-
from __future__ import print_function
import re
import string
import os
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import TwitterCorpusReader
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans

cachedStopWords = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")
emoji_re = re.compile(u'('
                      u'\ud83c[\udf00-\udfff]|'
                      u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
                      u'[\u2600-\u26FF\u2700-\u27BF])+',
                      re.UNICODE)


def clean_up_tweets(text):
    t = TweetTokenizer()
    tokens = t.tokenize(text)
    clean_tokens = []
    for token in tokens:
        if not token.startswith('https') \
                and not token.startswith('@') \
                and token not in cachedStopWords \
                and token not in set(string.punctuation) \
                and not re.match(emoji_re, token) \
                and re.match('[a-zA-Z]', token) \
                and token != 'rt':
            if token.startswith('#'):
                token = token[1:]
            clean_tokens.append(token)
    stems = [stemmer.stem(s) for s in clean_tokens]
    return stems


def main():
    twitter_docs = TwitterCorpusReader(os.getcwd(), '.*\.json').docs()
    tweets = [d["text"].lower() for d in twitter_docs if d["lang"] == "en"]
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in tweets:
        allwords_stemmed = clean_up_tweets(i)
        totalvocab_stemmed.extend(allwords_stemmed)
        allwords_tokenized = clean_up_tweets(i)
        totalvocab_tokenized.extend(allwords_tokenized)
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)
    print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                       min_df=0.002, stop_words='english',
                                       use_idf=True, tokenizer=clean_up_tweets, ngram_range=(1, 1))
    tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)
    terms = tfidf_vectorizer.get_feature_names()
    num_clusters = 3
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    tweet_data = {'id': [d["id_str"] for d in twitter_docs if d["lang"] == "en"], 'tweets': tweets, 'cluster': clusters}
    frame = pd.DataFrame(tweet_data, index=[clusters], columns=['id', 'tweets', 'cluster'])
    print(frame['cluster'].value_counts())
    print("Top terms per cluster:")
    print()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    for i in range(num_clusters):
        print("Cluster %d words:" % i, end='')
        for ind in order_centroids[i, :20]:
            print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
        print()
        print()
        print("Cluster %d tweets:" % i, end='')
        print()
        tw = list(set([u" ".join(c for c in clean_up_tweets(tt)) for tt in frame.ix[i]['tweets']]))
        for tweet in tw:
            print(tweet.encode('utf-8'))
            print()
        print()
        print()

    print()
    print()


if __name__ == '__main__':
    main()
