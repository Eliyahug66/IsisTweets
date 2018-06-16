import tweepy

ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""
CONSUMER_KEY = ""
CONSUMER_SECRET = ""

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)


# example: id = '1007246279016091648'
def tweet_text(id):
    return api.get_status(id, tweet_mode='extended')._json['full_text']
