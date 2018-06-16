from ModelCreator import load_model, isis_probability
from TweetsRetriever import tweet_text

model_path = '../Model/model.pkl'
classifier = load_model(model_path)


def get_isis_probability(id):
    return isis_probability(classifier, tweet_text(id))/26000.0 # factor the estimated ratio between our isis
                                                                # proportion to actual.  See Readme.md.
# Example:
print(get_isis_probability('1007246279016091648'))

