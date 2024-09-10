import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "python"
since = "2023-01-01"
until = "2024-09-11"

tweets = []

for tweet in sntwitter.TwitterSearch(query, since=since, until=until):
    tweets.append([tweet.date, tweet.username, tweet.content, tweet.url])

df = pd.DataFrame(tweets, columns=["Date", "Username", "Content", "URL"])

print(df)