import sys
sys.path.append("")

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import requests
from src.trading_record import scrape_trading_data
from src.Utils.utils import config_parser
from src.Utils.logger import create_logger


router = APIRouter(prefix="/mcp", tags=["Miscellaneous Control Panel"])

data = config_parser(data_config_path='config/config.yaml')
logger = create_logger(logfile="logging/mcp_apis.log")

class MasterQuestRequest(BaseModel):
    query: str

class ScrapeRequest(BaseModel):
    pass


@router.get(
    "/log",
    operation_id="get_log_file",
)
async def get_log_file():
    """Retrieves the content of the application's log file"""
    log_file_path = data.get('log_file_path')
    try:
        with open(log_file_path, 'rb') as log_file:
            log_content = log_file.read()
        return {"log": log_content.hex()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Log file not found")
    except Exception as e:
        logger.debug(msg=f"Error reading log file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")

@router.post(
    "/scrape",
    operation_id="scrape_trading_data_and_news",
)
async def scrape_data(request: ScrapeRequest):
    """Scrapes trading data and updates news using provided credentials"""
    try:
        TRADE_USER = os.getenv('TRADE_USER')
        TRADE_PASS = os.getenv('TRADE_PASS')
        scrape_trading_data(user_name=TRADE_USER, password=TRADE_PASS)
        logger.info(msg="Done scraping trading data and updating news")
        return {"message": "Done scraping trading data and updating news"}
    except Exception as e:
        logger.debug(msg=f"Error scraping data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scraping data: {str(e)}")

@router.post(
    "/masterquest",
    operation_id="masterquest_query",
)
async def masterquest(request: MasterQuestRequest):
    """Processes a query through the MasterQuest LLM and RAG system"""
    masterquest_url = data.get('masterquest_url')
    try:
        response = requests.post(masterquest_url, json={'query': request.query})
        response_data = response.json()
        logger.debug(msg=f"Masterquest result: {response_data}")
        return {
            "model_type": response_data['model_type'],
            "result": response_data['result']
        }
    except Exception as e:
        logger.debug(msg=f"Error on LLM and RAG system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error on LLM and RAG system: {str(e)}")

@router.post(
    "/update-vectordb",
    operation_id="update_vector_database",
)
async def update_vector_db():
    """Updates the vector database for improved query performance"""
    updatevectordb_url = data.get('updatevectordb_url')
    try:
        response = requests.post(updatevectordb_url)
        if response.status_code == 200:
            logger.debug(msg="Update Vector DB Successful")
            return {"message": "Update was successful"}
        else:
            logger.debug(msg=f"Update Vector DB Failed: {response.status_code} - {response.json()['message']}")
            raise HTTPException(status_code=response.status_code, detail=response.json()['message'])
    except Exception as e:
        logger.debug(msg=f"Error updating vector DB: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating vector DB: {str(e)}")