import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables from .env file
load_dotenv()

def get_articles_from_gnews(api_key, keywords, page_size=10):
    # Combine multiple keywords into a single query string
    query = ' OR '.join(keywords)
    
    # Calculate the date and time 7 days ago
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)
    start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    url = (f'https://gnews.io/api/v4/search?q={query}&token={api_key}'
           f'&lang=en&country=IN&max={page_size}&from={start_time_str}')
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('articles', [])
    else:
        print(f"Error fetching data: {response.status_code}")
        return []

def main():
    # Retrieve API key from environment variables
    api_key = os.getenv('GNEWS_API_KEY')
    if not api_key:
        print("API key not found. Please set it in the .env file.")
        return

    # Define your search keywords
    keywords = ['HDFC', 'HDFC Bank', 'Bank']  # Add more keywords as needed
    articles = get_articles_from_gnews(api_key, keywords)
    
    print(f"Retrieved {len(articles)} articles.")
    for article in articles:
        print(f"Title: {article['title']}")
        print(f"Description: {article['description']}")
        print(f"URL: {article['url']}\n")

if __name__ == "__main__":
    main()