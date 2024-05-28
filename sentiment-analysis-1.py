# Importing necessary libraries
import nltk
#symbolic and statistical language natural language processing
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the vader_lexicon package
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    # Initialize the VADER sentiment intensity analyzer
    sia = SentimentIntensityAnalyzer()

    # Compute and print the sentiment scores
    sentiment = sia.polarity_scores(text)
    #gives positive and negative weightings between 0-1 and an overall compound score betweeen -1 and 1
    print(sentiment)

# Test the function with a sample text
analyze_sentiment("Lamees is a beautiful stunning amazing genius angel")
analyze_sentiment("angiotensin is a vasoconstrictor")
analyze_sentiment("luna is not very nice")