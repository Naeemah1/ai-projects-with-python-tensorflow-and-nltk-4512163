import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import random
nltk.download('movie_reviews')

# Prepare the dataset
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
random.shuffle(documents) # reduce bias

# Define the feature extractor
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# Train the classifier
featuresets = [(document_features(d), c) for (d,c) in documents]
#each feature set is a tuple containing dict of features and a category
train_set, test_set = featuresets[100:], featuresets[:100]
#split into train and test set
classifier = NaiveBayesClassifier.train(train_set)
#probabilistic classifier that applies bayes
#works by having strong assumptions on different independencies of features we created


# Test the classifier
print(accuracy(classifier, test_set))

# Show the most important features
classifier.show_most_informative_features(5)