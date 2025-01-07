from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import time
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Set up Chrome options
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Run Chrome in headless mode
chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
chrome_options.add_argument("--window-size=1920x1080")  # Set window size
chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems

# Initialize the Chrome driver with options
search_query = input("Enter the search query: ")
driver = webdriver.Chrome(options=chrome_options)
driver.get("https://x.com/i/flow/login")

# Retrieve credentials from environment variables
usern = str(os.getenv("TWITTER_USERNAME"))
passw = str(os.getenv("TWITTER_PASSWORD"))

# Automate login with explicit waits
try:
    username = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.NAME, "text"))
    )
    print("Username field located")
    username.send_keys(usern + Keys.RETURN)  # Use environment variable for security
    time.sleep(5)  # Increase sleep time to ensure the next page loads

    password = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.NAME, "password"))
    )
    print("Password field located")
    password.send_keys(passw + Keys.RETURN)
except Exception as e:
    print(f"Error during login: {e}")
    driver.quit()
    exit()

time.sleep(10)

# Navigate to search page
try:
    driver.get(f"https://x.com/search?q=%23{search_query}&f=live")
    tweets = set()  # Use a set to store unique tweets
    
    for _ in tqdm(range(5), desc="Scrolling through tweets"):  # Adjust the range to control how many scrolls to perform
        WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='tweetText']"))
        )
        
        # Scrape tweets currently loaded
        tweet_elements = driver.find_elements(By.CSS_SELECTOR, "[data-testid='tweetText']")
        for tweet in tweet_elements:
            tweets.add(tweet.text)
        
        # Scroll down the page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)  # Wait for new tweets to load
    
    # Print all unique tweets and analyze sentiment
    sentiments = []
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}

    def analyze_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, clean_up_tokenization_spaces=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        sentiment = torch.argmax(logits, dim=1).item()
        return sentiment_map[sentiment]

    for tweet in tweets:
        sentiment = analyze_sentiment(tweet)
        sentiments.append((tweet, sentiment))
    
    # Display results
    for idx, (tweet, sentiment) in enumerate(sentiments, start=1):
        print(f"{idx}. Tweet: {tweet}\nSentiment: {sentiment}\n")
    print(f"Total tweets analyzed: {len(tweets)}")

except Exception as e:
    print(f"Error fetching tweets: {e}")

# Quit the driver
driver.quit()
