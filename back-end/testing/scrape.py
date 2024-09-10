import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta

def fetch_tweets(query, num_tweets=10):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)
    
    formatted_query = f"{query} since:{start_time.strftime('%Y-%m-%d')} until:{end_time.strftime('%Y-%m-%d')}"
    
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(formatted_query).get_items():
        if len(tweets) >= num_tweets:
            break
        tweets.append(tweet)
    
    return tweets

def main():
    queries = ['HDFC', 'HDFC Bank']
    
    for query in queries:
        tweets = fetch_tweets(query)
        print(f"Tweets for '{query}':")
        for tweet in tweets:
            print(f"Tweet: {tweet.content}")
            print(f"Author: {tweet.user.username}")
            print(f"Date: {tweet.date}\n")

if __name__ == "__main__":
    main()
