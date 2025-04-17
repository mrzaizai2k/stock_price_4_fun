import sys
sys.path.append("")
import requests
import os
from src.Utils.utils import config_parser
import binascii

# Read configuration
data = config_parser(data_config_path='config/config.yaml')

# Base URL for the API (read from config or default to localhost:8000)
BASE_URL = f"http://{data.get('api_host', 'localhost')}:{data.get('api_port', 8000)}"

def test_paybacktime(symbol: str = "ACB"):
    """Test the /stocks/paybacktime endpoint"""
    endpoint = f"{BASE_URL}/stocks/paybacktime"
    payload = {"symbol": symbol}
    try:
        response = requests.post(endpoint, json=payload)
        print(f"Payback Time Test for {symbol}:")
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error testing paybacktime for {symbol}: {e}")

def test_support_resistance(symbol: str = "ACB"):
    """Test the /stocks/support-resistance endpoint"""
    endpoint = f"{BASE_URL}/stocks/support-resistance"
    payload = {"symbol": symbol}
    try:
        response = requests.post(endpoint, json=payload)
        
        print(f"\nSupport Resistance Test for {symbol}:")
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error testing support-resistance for {symbol}: {e}")

def test_find_paybacktime_stocks(symbol: str = "ACB"):
    """Test the /stocks/find-paybacktime-stocks endpoint"""
    endpoint = f"{BASE_URL}/stocks/find-paybacktime-stocks"
    try:
        response = requests.get(endpoint)
        
        print(f"\nFind Payback Time Stocks Test (symbol {symbol} not used in this endpoint):")
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error testing find-paybacktime-stocks: {e}")

def test_summary_news_url(news_url: str = "https://www.example.com/news"):
    """Test the /stocks/summary-news-url endpoint"""
    endpoint = f"{BASE_URL}/stocks/summary-news-url"
    payload = {"url": news_url}
    try:
        response = requests.post(endpoint, json=payload)
        
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error testing summary-news-url: {e}")

def test_pattern(symbol: str = "ACB", start_date: str = "2023-01-01"):
    """Test the /stocks/pattern endpoint (returns image)"""
    endpoint = f"{BASE_URL}/stocks/pattern"
    payload = {
        "symbol": symbol,
        "start_date": start_date
    }
    try:
        response = requests.post(endpoint, json=payload)
        
        result = response.json()
        print(f"\nPattern Test for {symbol}:")
        print(f"Message: {result['message']}")
        # Save image to file for verification
        image_data = binascii.unhexlify(result['image'])
        with open(f"test_pattern_image_{symbol}.png", "wb") as f:
            f.write(image_data)
        print(f"Image saved as test_pattern_image_{symbol}.png")
    except requests.exceptions.RequestException as e:
        print(f"Error testing pattern for {symbol}: {e}")
    except binascii.Error as e:
        print(f"Error decoding image data for {symbol}: {e}")

if __name__ == "__main__":

    symbol="ACBC"
    news_url = "https://vnexpress.net/tong-bi-thu-viet-nam-du-suc-vuot-qua-thach-thuc-4875205.html"
    print("Starting API tests...")
    # test_paybacktime(symbol=symbol)
    # test_support_resistance(symbol=symbol)
    # test_find_paybacktime_stocks(symbol=symbol)  # Symbol not used in this endpoint
    test_summary_news_url(news_url=news_url)
    # test_pattern(symbol=symbol, start_date="2023-01-01")
    print("\nAll tests completed.")