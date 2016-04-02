import collections, os, prettytable
from nltk import ngrams
from nltk.tokenize import TweetTokenizer
from nltk.corpus import TwitterCorpusReader


class Part1():
    def __init__(self, tweets):
        self.tweets = tweets
        self.bigram_to_ids = {}  # map from bigram to list of twitter ids
        self.trigram_to_ids = {}  # map from trigram to list of twitter ids
        self.bicount = collections.Counter()
        self.tricount = collections.Counter()

    def update_freqs(self, doc_text, id_str):
        for bigram in list(ngrams(doc_text, 2)):
            k = bigram[0] + u"_" + bigram[1]
            self.bicount.update([k])
            self.bigrams[k] = self.bigrams.get(k, []) + [id_str]
        for trigram in list(ngrams(doc_text, 3)):
            k = trigram[0] + u"_" + trigram[1] + u"_" + trigram[2]
            self.tricount.update([k])
            self.trigrams[k] = self.trigrams.get(k, []) + [id_str]


def main():
    t = TweetTokenizer()
    twitter_docs = TwitterCorpusReader(os.getcwd(), '.*\.json').docs()
    part1_data = Part1(twitter_docs)
    for doc in twitter_docs:
        part1_data.update_freqs(t.tokenize(doc["text"]), doc["id_str"])
    pt1 = prettytable.PrettyTable(['Bigram', 'Frequency', 'ID List Size'])
    for row in part1_data.bicount:
        pt1.add_row(row + (len(part1_data.bigram_to_ids[row[0]]),))
    pt2 = prettytable.PrettyTable(['Trigram', 'Frequency', 'ID List Size'])
    for row in part1_data.tricount:
        pt2.add_row(row + (len(part1_data.trigram_to_ids[row[0]]),))
    print pt1
    print pt2


if __name__ == '__main__':
    main()
