from scrapy import Spider
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

class TwitterTopicSpider(Spider):
    name = "twitter_topic"
    allowed_domains = ["twitter.com"]
    start_urls = [
        "https://twitter.com/search?q=Python&src=typed_query&f=live"  # Replace 'Python' with your search topic
    ]

    def __init__(self, *args, **kwargs):
        super(TwitterTopicSpider, self).__init__(*args, **kwargs)
        # Initialize Selenium WebDriver
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

    def parse(self, response):
        self.driver.get(self.start_urls[0])
        time.sleep(5)  # Wait for page to load

        # Scroll to load more tweets
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        tweets = []
        while len(tweets) < 100:  # Limit to 100 tweets
            # Extract tweet texts
            elements = self.driver.find_elements(By.CSS_SELECTOR, 'div[lang]')  # Tweets have a 'lang' attribute
            for element in elements:
                tweets.append(element.text)

            # Scroll down
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break  # Stop if no more content is loaded
            last_height = new_height

        # Yield scraped tweets
        for tweet in tweets:
            yield {"tweet": tweet}

        # Close the Selenium WebDriver
        self.driver.quit()
