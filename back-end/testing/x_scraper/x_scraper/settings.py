# settings.py

BOT_NAME = 'x_scraper'

SPIDER_MODULES = ['x_scraper.spiders']
NEWSPIDER_MODULE = 'x_scraper.spiders'

# User-agent string to avoid being blocked
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Configure maximum concurrent requests (default: 16)
CONCURRENT_REQUESTS = 5

# Output format and location
FEED_FORMAT = 'json'
FEED_URI = 'nifty_tweets.json'

# Configure a delay for requests (to avoid hitting Twitter too fast)
DOWNLOAD_DELAY = 1

# Enable or disable spider middlewares
SPIDER_MIDDLEWARES = {
   'scrapy.downloadermiddlewares.retry.RetryMiddleware': None,
}

# Enable or disable downloader middlewares
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
}

# Respect robots.txt (set to False if you want to scrape even if Twitter's robots.txt disallows it)
ROBOTSTXT_OBEY = False
