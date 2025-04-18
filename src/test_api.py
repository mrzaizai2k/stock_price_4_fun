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

def test_remote():
    """Test the /mcp/remote endpoint (opens VS Code tunnel)"""
    endpoint = f"{BASE_URL}/mcp/remote"
    try:
        response = requests.post(endpoint)
        response.raise_for_status()
        result = response.json()
        print("\nRemote Test:")
        print(f"Message: {result['message']}")
    except requests.exceptions.RequestException as e:
        print(f"Error testing remote: {e.response.json() if e.response else e}")

def test_log():
    """Test the /mcp/log endpoint (returns log file)"""
    endpoint = f"{BASE_URL}/mcp/log"
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        result = response.json()
        print("\nLog Test:")
        # Save log file for verification
        log_data = binascii.unhexlify(result['log'])
        with open("test_log_file.log", "wb") as f:
            f.write(log_data)
        print("Log file saved as test_log_file.log")
    except requests.exceptions.RequestException as e:
        print(f"Error testing log: {e.response.json() if e.response else e}")
    except binascii.Error as e:
        print(f"Error decoding log data: {e}")

def test_scrape():
    """Test the /mcp/scrape endpoint (scrapes trading data and news)"""
    endpoint = f"{BASE_URL}/mcp/scrape"
    payload = {}
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        result = response.json()
        print("\nScrape Test:")
        print(f"Message: {result['message']}")
    except requests.exceptions.RequestException as e:
        print(f"Error testing scrape: {e.response.json() if e.response else e}")

def test_masterquest(query: str = "Who is Karger?"):
    """Test the /mcp/masterquest endpoint (queries LLM and RAG system)"""
    endpoint = f"{BASE_URL}/mcp/masterquest"
    payload = {
        "query": query,
    }
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        result = response.json()
        print("\nMasterquest Test:")
        print(f"Model Type: {result['model_type']}")
        print(f"Result: {result['result']}")
    except requests.exceptions.RequestException as e:
        print(f"Error testing masterquest: {e.response.json() if e.response else e}")

def test_update_vectordb():
    """Test the /mcp/update-vectordb endpoint (updates vector database)"""
    endpoint = f"{BASE_URL}/mcp/update-vectordb"
    try:
        response = requests.post(endpoint)
        response.raise_for_status()
        result = response.json()
        print("\nUpdate VectorDB Test:")
        print(f"Message: {result['message']}")
    except requests.exceptions.RequestException as e:
        print(f"Error testing update-vectordb: {e.response.json() if e.response else e}")

if __name__ == "__main__":

    symbol="ACBC"
    news_url = "https://cafef.vn//mot-cong-ty-bds-khu-cong-nghiep-bao-lai-rong-quy-1-2025-tang-106-ky-moi-3-mou-gan-10ha-188250416140034188.chn"
    print("Starting API tests...")
    # test_paybacktime(symbol=symbol)
    # test_support_resistance(symbol=symbol)
    # test_find_paybacktime_stocks(symbol=symbol)  # Symbol not used in this endpoint
    test_summary_news_url(news_url=news_url)
    # test_pattern(symbol=symbol, start_date="2023-01-01")

    VALID_USER_ID = os.getenv("MRZAIZAI2K_ID", "123456")  # Set to actual MRZAIZAI2K_ID
    INVALID_USER_ID = "999999"

    test_remote()
    test_log()
    test_scrape()
    test_masterquest()
    # test_update_vectordb()

    print("\nAll tests completed.")