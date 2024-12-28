from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys  # For keyboard inputs
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os  # For secure credentials
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Initialize the WebDriver
driver = webdriver.Chrome()
driver.get("https://x.com/i/flow/login")

# Retrieve credentials from environment variables
usern = str(os.getenv("TWITTER_USERNAME"))
passw = str(os.getenv("TWITTER_PASSWORD"))

# Automate login with explicit waits
try:
    username = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "text"))
    )
    username.send_keys(usern + Keys.RETURN)  # Use environment variable for security
    time.sleep(3)
    password = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "password"))
    )
    password.send_keys(passw + Keys.RETURN)
except Exception as e:
    print(f"Error during login: {e}")
    driver.quit()
    exit()

time.sleep(5)
# Navigate to search page
try:
    driver.get("https://x.com/search?q=%23Nifty&f=live")
    tweets = set()  # Use a set to store unique tweets
    
    for _ in range(5):  # Adjust the range to control how many scrolls to perform
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='tweetText']"))
        )
        
        # Scrape tweets currently loaded
        tweet_elements = driver.find_elements(By.CSS_SELECTOR, "[data-testid='tweetText']")
        for tweet in tweet_elements:
            tweets.add(tweet.text)
        
        # Scroll down the page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)  # Wait for new tweets to load
    
    # Print all unique tweets and analyze sentiment
    sentiments = []
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}

    def analyze_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        sentiment = torch.argmax(logits, dim=1).item()
        return sentiment_map[sentiment]

    for tweet in tweets:
        sentiment = analyze_sentiment(tweet)
        sentiments.append((tweet, sentiment))
    
    # Display results
    for tweet, sentiment in sentiments:
        print(f"Tweet: {tweet}\nSentiment: {sentiment}\n")
    print(f"Total tweets analyzed: {len(tweets)}")

except Exception as e:
    print(f"Error fetching tweets: {e}")

# Quit the driver
driver.quit()
