import sys
sys.path.append("")

import flask
from dotenv import load_dotenv
load_dotenv()

import requests

# URL for the API endpoint
api_url = 'http://localhost:8083/query'

# Input query
query = input("Enter your query: ")

# Send POST request to the API
response = requests.post(api_url, json={'query': query})

# Print the response
print(response.json())
