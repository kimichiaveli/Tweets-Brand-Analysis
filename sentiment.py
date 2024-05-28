import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def sentiment_analyzer(df,text_column):
    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Define a function to apply sentiment analysis while handling NaN values
    def apply_sentiment_analysis(x):
        if isinstance(x, str):  # Check if x is a string
            return sid.polarity_scores(x)['compound']
        else:
            return float('nan')  # Return NaN for non-string values
        
    # Define threshold values for sentiment classification
    positive_threshold = 0.2
    negative_threshold = -0.2

    # Apply VADER sentiment analyzer
    df['sentiment_score'] = df[text_column].apply(apply_sentiment_analysis)

    # Apply sentiment labeling using lambda functions
    df['sentiment_label'] = df['sentiment_score'].apply(lambda x: 'positive' if x > positive_threshold else ('negative' if x < negative_threshold else 'neutral'))

