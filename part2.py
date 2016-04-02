import os, prettytable, re, string
from textblob import TextBlob
from nltk.tokenize import TweetTokenizer
from nltk.corpus import TwitterCorpusReader


class Part2():
    def __init__(self):
        self.subj_scores = {}  # map from tweet text to the tuple (subjectivity, polarity_score)

    def add_score(self, tweet_id, scores):
        self.subj_scores[tweet_id] = scores


def main():
    t = TweetTokenizer()
    twitter_docs = TwitterCorpusReader(os.getcwd(), '.*\.json').docs()
    part2 = Part2()
    for tweet in twitter_docs:
        text = TextBlob(tweet["text"])
        if tweet["lang"] != "en":
            try:
                text = TextBlob(tweet["text"]).translate(from_lang=tweet["lang"], to='en')
            except:
                pass
        part2.add_score(unicode(text), text.sentiment)
    for key, value in part2.subj_scores.items():
        pol = ''
        if value.polarity == 0:
            pol = u"neutral"
        if value.polarity > 0:
            pol = u"positive"
        if value.polarity < 0:
            pol = u"negative"
        print key + u": \n has polarity: " + pol + ", with subj_score = " + unicode(value.subjectivity)


if __name__ == '__main__':
    main()
