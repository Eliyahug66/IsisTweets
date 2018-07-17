# ISIS Detector
## Python (2.7) code for determining the probability for a tweet being in support of ISIS
### Folders:
1.	*src* contains the Python source code
		*ModelCreator.py* - Training and prediction
		*TweetsRetriever.py* - Retrieving text of a tweet by its id
		*Predictor.py* - Given an id, retrieves tweet text using *TweetsRetriever.py* and determines the probability for being in support of ISIS using *ModelCreator.py*.  When initialized, loads the model.pkl file (see below)
2.	*Input* contains the input data for building the model (original files: https://www.kaggle.com/fifthtribe/how-isis-uses-twitter/data)
		*tweets_random_all.txt* contains 200k tweets randomly selected from Twitter, one per line
		*tweets_isis_all.txt* contains > 17k tweets identified as supporting isis
		
3.	*Model* contains *model.pkl* - Logistic Regresion model dumped by the *joblib.dump* method

### How to run prediction
1.	Download the folders structure (input files are not needed for that)
2.	Create a Twitter account to get the keys required and modify *TweetsRtriever.py* accordingly
3.	Run Predictor.py.  Example: *print(get_isis_probability('1007246279016091648'))*


### Algorithm used
*sklearn logistic regression*, which is the basic algorithm for text classification with probability (unlike *Naive Bayes*)
I used it in the most basic way, taking words as features (tokenized by nltk), ignoring re-occurrence of a word within a tweet.

### Error rate
Separating into training and testing sets we yields 2.2% error rate.

Let's estimate the actual error rate when this predictor is fed with random tweets:
Supposing number of isis related tweets on the first third of 2016 were about 5 times higher - i.e. 100k.  From http://uk.businessinsider.com/tweets-on-twitter-is-in-serious-decline-2016-2 we can roughly estimate about 250M tweets daily in that period, which amounts totally to 30G tweets, which brings us to an actual ratio isis/random = 0.00033% which is 1/26k of our training data.

This means that **our predictions would consist almost entirely on false positives (i.e.: detecting non-isis as isis).**

### How to improve this?
**Features:** instead of using just words as is,
* consider using stoplists and other pre-processing techniques (tf-idf, ngrams, pos tagging, ner tagging...)
* Some substrings, especially in names seem to be indicative.
* Tendency to quote could also be indicative.
* Urls within tweet:
*	their total number in the tweet
*	if not to another tweet, consider suffix and prefix of links (after converting from short format) as a feature
*	resource type (image, video, other tweet)
*	topic or classification of linked resource or site (using Similar Web service or alike)
*	is linked content removed?
* 	structural considerations: who else links to that content?  Create influencers map and consider PageRank (i.e.: https://en.wikipedia.org/wiki/PageRank) scoring

* Consider special handling of hashtags (clustering?)
* Use tweet metadata - age of account (small is more likely to follow a removal) and other
* For users who have enough text - consider using authorship attribution (e.g.: Moshe Koppel's work) to realize if same person under different username
* Sentiment analysis to realize if atrocities are expressed with joy
* Time of the day, cluster - much research is required to decide upon goal (user type in the isis fabric? content topic? sentiment?) and relevant features.  Otherwise you get nonsense.

**Dataset issues**
We might want to consider having one model for making distinction between random tweets to those come from random muslim users and another for differenciating between the second to isis supporters.  Otherwise, almost every single classifier would contain too many FP.  Use lightweight classifier first, and more accurate, which demands heavier processing next.

**Algorithm**
First, tweak the LGR parameters used.  Then, considering much larger repositories, we may need to find a platform for running the training distributively, probably with different algorithms, maybe in several phases: LDA, SGD, RNN (deep learning), Poisson logistic...
