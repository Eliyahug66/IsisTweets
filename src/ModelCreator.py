# import nltk
# nltk.download('punkt')
import codecs
import math

from nltk import NaiveBayesClassifier, classify, SklearnClassifier
from nltk.tokenize import word_tokenize
import random
import scipy
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression


# probabiliy for the tweet be isis
def isis_probability(classifier, tweet):
    return math.exp(classify.log_likelihood(classifier, [(tweet_features(tweet), 'isis')]))


def tweet_features(line):
    if (len(line) > 1):
        if line[0] == '"' and line[-1] == '"':  # Quotations is a future feature, unless in encloses entire tweet
            line = line.strip('"')
    return {word: '1' for word in word_tokenize(line)}


def train(isis_path, general_path, out_path, for_production=True):
    # Load data
    isis_tweets = tuple(codecs.open(isis_path, 'r', 'utf-8-sig'))
    general_tweets = tuple(codecs.open(general_path, 'r', 'utf-8-sig'))
    # Done

    # Build datasets
    #   Label & shuffle lines
    labeled_lines = ([(line, 'isis') for line in isis_tweets] + [(line, 'general') for line in general_tweets])
    random.shuffle(labeled_lines)
    #   Tokenize into words
    entire_set = [(tweet_features(n), tweet_class) for (n, tweet_class) in labeled_lines]
    cls = SklearnClassifier(LogisticRegression())
    train_set = test_set = entire_set
    if not for_production:
        train_set = entire_set[500:]
        test_set = entire_set[:500]
    cls.train(train_set)
    print("accuracy on training set: " + str(classify.accuracy(cls, test_set)))
    joblib.dump(cls, out_path)


def load_model(model_path):
    return joblib.load(model_path)


def predict_tweet():
    return


# =================== MAIN ========================================================
def main():
    model_path = '../Model/model.pkl'
    isis_path = '../input/tweets_isis_all.txt'
    general_path = '../input/tweets_random_all.txt'
    for_production = True  # if false, separates into training set and test set (to estimate accuracy in production).
                            # Otherwise, takes advantage of the entire dataset
    train(isis_path, general_path, model_path, for_production)
    classifier = load_model(model_path)
    print(isis_probability(classifier,
                           "ENGLISH TRANSLATION: 'A MESSAGE TO THE TRUTHFUL IN SYRIA - SHEIKH ABU MUHAMMED AL MAQDISI: http://t.co/73xFszsjvr http://t.co/x8BZcscXzq"))
    print("Completed!")
