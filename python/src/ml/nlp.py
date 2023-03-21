import re

import nltk
from nltk import PorterStemmer, WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS

sentiment_analyzer = VS()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)



################################# Modifed the code from Char to Word level from Pinkesh paper ################################################

import sys
import re
from string import punctuation
# from preprocess_twitter import tokenize as tokenizer_g
from gensim.parsing.preprocessing import STOPWORDS

FLAGS = re.MULTILINE | re.DOTALL

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = u"<hashtag> {} <allcaps>".format(hashtag_body)
    else:
        result = " ".join(["<hashtag>"] + re.split("(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


def tokenizer_g(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text.lower()

def glove_tokenize(text):
    text = tokenizer_g(text)
    text = ''.join([c for c in text if c not in punctuation])
    words = text.split()
    # words = [word for word in words if word not in STOPWORDS]
    return words

################################################################################################################


# stem_or_lemma: 0 - apply porter's stemming; 1: apply lemmatization; 2: neither
# -set to 0 to reproduce Davidson. However, note that because a different stemmer is used, results could be
# sightly different
# -set to 2 will do 'basic_tokenize' as in Davidson
def tokenize(tweet, stem_or_lemma=0):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and normalizes tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    if stem_or_lemma==0:
        tokens = [stemmer.stem(t) for t in tweet.split()]
    elif stem_or_lemma==1:
        tokens=[lemmatizer.lemmatize(t) for t in tweet.split()]
    else:
        tokens = [t for t in tweet.split()] #this is basic_tokenize in TD's original code
    return tokens


# tweets should have been preprocessed to the clean/right format before passing to this method
def get_pos_tags(tweets):
    """Takes a list of strings (tweets) and
    returns a list of strings of (POS tags).
    """
    tweet_tags = []
    for t in tweets:
        tokens = tokenize(t, 2)
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        #for i in range(0, len(tokens)):
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    return tweet_tags
