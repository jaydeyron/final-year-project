import os # type: ignore
import tweepy # type: ignore
import pandas as pd # type: ignore
import re # type: ignore
from dotenv import load_dotenv #type: ignore
from textblob import TextBlob #type: ignore

load_dotenv()

API_KEY = os.getenv('TWITTER_API_KEY')
API_SECRET_KEY = os.getenv('TWITTER_API_SECRET_KEY')
ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

try:
    api.verify_credentials()
    print("Authentication successful")
except:
    print("Authentication failed")

search_query = 'stock market'

try:
    tweets = api.get_tweets(query = 'Donald Trump', count = 200)
except tweepy.TweepyException as e:
    print(f"Error fetching tweets: {e}")

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

data = []
for tweet in tweets:
    text = preprocess_text(tweet.text)
    sentiment = analyze_sentiment(text)
    data.append({'text': tweet.text, 'processed_text': text, 'sentiment': sentiment})

for item in data:
    print(f"Original Tweet: {item['text']}")
    print(f"Processed Text: {item['processed_text']}")
    print(f"Sentiment Score: {item['sentiment']}")
    print("-" * 50)