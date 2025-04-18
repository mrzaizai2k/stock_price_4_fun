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



def test_remote(user_id: str = "12345"):
    """Test the /mcp/remote endpoint (opens VS Code tunnel)"""
    endpoint = f"{BASE_URL}/mcp/remote"
    headers = {"user_id": str(user_id)}
    try:
        response = requests.post(endpoint, headers=headers)
        response.raise_for_status()
        result = response.json()
        print(f"\nRemote Test (user_id={user_id}):")
        print(f"Message: {result['message']}")
    except requests.exceptions.RequestException as e:
        print(f"Error testing remote for user_id={user_id}: {e.response.json() if e.response else e}")

def test_log(user_id: str = "12345"):
    """Test the /mcp/log endpoint (returns log file)"""
    endpoint = f"{BASE_URL}/mcp/log"
    headers = {"user_id": str(user_id)}
    try:
        response = requests.get(endpoint, headers=headers)
        response.raise_for_status()
        result = response.json()
        print(f"\nLog Test (user_id={user_id}):")
        # Save log file for verification
        log_data = binascii.unhexlify(result['log'])
        with open(f"test_log_file_{user_id}.log", "wb") as f:
            f.write(log_data)
        print(f"Log file saved as test_log_file_{user_id}.log")
    except requests.exceptions.RequestException as e:
        print(f"Error testing log for user_id={user_id}: {e.response.json() if e.response else e}")
    except binascii.Error as e:
        print(f"Error decoding log data for user_id={user_id}: {e}")

def test_scrape(user_id: str = "12345"):
    """Test the /mcp/scrape endpoint (scrapes trading data and news)"""
    endpoint = f"{BASE_URL}/mcp/scrape"
    payload = {"user_id": int(user_id)}
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        result = response.json()
        print(f"\nScrape Test (user_id={user_id}):")
        print(f"Message: {result['message']}")
    except requests.exceptions.RequestException as e:
        print(f"Error testing scrape for user_id={user_id}: {e.response.json() if e.response else e}")

def test_masterquest(query: str = "What is the stock market?", user_id: str = "12345"):
    """Test the /mcp/masterquest endpoint (queries LLM and RAG system)"""
    endpoint = f"{BASE_URL}/mcp/masterquest"
    payload = {
        "query": query,
        "user_id": int(user_id)
    }
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        result = response.json()
        print(f"\nMasterquest Test (user_id={user_id}):")
        print(f"Model Type: {result['model_type']}")
        print(f"Result: {result['result']}")
    except requests.exceptions.RequestException as e:
        print(f"Error testing masterquest for user_id={user_id}: {e.response.json() if e.response else e}")

def test_update_vectordb(user_id: str = "12345"):
    """Test the /mcp/update-vectordb endpoint (updates vector database)"""
    endpoint = f"{BASE_URL}/mcp/update-vectordb"
    headers = {"user_id": str(user_id)}
    try:
        response = requests.post(endpoint, headers=headers)
        response.raise_for_status()
        result = response.json()
        print(f"\nUpdate VectorDB Test (user_id={user_id}):")
        print(f"Message: {result['message']}")
    except requests.exceptions.RequestException as e:
        print(f"Error testing update-vectordb for user_id={user_id}: {e.response.json() if e.response else e}")

def test_unauthorized_access():
    """Test all /mcp endpoints with an invalid user_id to verify 403 errors"""
    endpoints = [
        (f"{BASE_URL}/mcp/remote", "POST", {"user_id": INVALID_USER_ID}, None),
        (f"{BASE_URL}/mcp/log", "GET", {"user_id": INVALID_USER_ID}, None),
        (f"{BASE_URL}/mcp/scrape", "POST", None, {"user_id": int(INVALID_USER_ID)}),
        (f"{BASE_URL}/mcp/masterquest", "POST", None, {"query": "Test query", "user_id": int(INVALID_USER_ID)}),
        (f"{BASE_URL}/mcp/update-vectordb", "POST", {"user_id": INVALID_USER_ID}, None)
    ]
    for endpoint, method, headers, payload in endpoints:
        try:
            if method == "POST":
                response = requests.post(endpoint, headers=headers, json=payload)
            else:
                response = requests.get(endpoint, headers=headers)
            print(f"\nUnauthorized Access Test ({endpoint}):")
            if response.status_code == 403:
                print(f"Expected 403 error: {response.json()['detail']}")
            else:
                print(f"Unexpected status code {response.status_code}: {response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"Error testing unauthorized access for {endpoint}: {e.response.json() if e.response else e}")


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

    # New tests for /mcp endpoints
    test_remote(user_id=VALID_USER_ID)
    test_log(user_id=VALID_USER_ID)
    test_scrape(user_id=VALID_USER_ID)
    test_masterquest(query="What is the stock market?", user_id=VALID_USER_ID)
    test_update_vectordb(user_id=VALID_USER_ID)

    # Test unauthorized access for all endpoints
    test_unauthorized_access()
    print("\nAll tests completed.")