import snscrape.modules.twitter as sntwitter

# Create a query
query = "(#python) since:2023-11-16 until:2023-11-17"

# Scrape tweets
tweets = sntwitter.TwitterSearchScraper(query).get_items()

for tweet in tweets:
    print(f"{tweet.date} - {tweet.user.username}: {tweet.content}")