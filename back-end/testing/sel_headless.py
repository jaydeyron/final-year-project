from playwright.sync_api import Playwright, sync_playwright
import time

def get_nifty_tweets(username, password):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to Twitter's login page
        page.goto("https://x.com/login")

        # Fill in the username and password fields
        page.fill("input[name='text']", username)
        page.fill("input[name='password']", password)

        # Click the login button
        page.click("button[type='submit']")

        # Wait for login to complete
        page.wait_for_timeout(5000)

        # Navigate to the Nifty search page
        page.goto("https://x.com/search?q=Nifty&src=typed_query&f=live")

        # Wait for the page to load
        page.wait_for_selector(".tweet-wrapper")

        # Find the tweet elements and extract information
        tweets = page.query_selector_all(".tweet-wrapper")
        for tweet in tweets:
            username = tweet.query_selector(".tweet-header .username").inner_text()
            content = tweet.query_selector(".tweet-content").inner_text()

            print(f"Username: {username}")
            print(f"Content: {content}")
            print("--------------------")

        browser.close()

# Replace 'your_username' and 'your_password' with your actual credentials
username = ""
password = ""
get_nifty_tweets(username, password)