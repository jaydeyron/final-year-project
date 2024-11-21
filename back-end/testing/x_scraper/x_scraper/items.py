import scrapy

class XScraperItem(scrapy.Item):
    tweet_number = scrapy.Field()
    content = scrapy.Field()
