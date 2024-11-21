import scrapy

class NiftySpider(scrapy.Spider):
    name = "nifty"
    allowed_domains = ["x.com"]
    start_urls = ["https://x.com/search?q=Nifty&src=typed_query&f=live"]

    def parse(self, response):
        # Extract the full HTML content
        html_content = response.body.decode('utf-8')
        
        # Limit the number of tweets to 10 for simplicity (or remove the limit to save all)
        yield {
            'url': response.url,
            'html': html_content  # Storing the entire HTML content
        }
