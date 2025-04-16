import sys
sys.path.append("")

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import os
import requests
from src.trading_record import scrape_trading_data
from src.summarize_text import StockNewsDatabase
from src.Utils.bot_utils import run_vscode_tunnel
from src.Utils.utils import config_parser
from src.Utils.logger import create_logger
from typing import Dict, Any

router = APIRouter(prefix="/mcp", tags=["Miscellaneous Control Panel"])

data = config_parser(data_config_path='config/config.yaml')
logger = create_logger(logfile="logging/mcp_apis.log")

# Mock user validation (replace with actual implementation)
def validate_mrzaizai2k_user(user_id: int) -> bool:
    # Replace with actual validation logic
    return user_id == int(os.getenv('YOUR_TELEGRAM_ID', '0'))

class MasterQuestRequest(BaseModel):
    query: str

def get_user_id() -> int:
    # Mock user ID for API context; in practice, use proper auth
    return int(os.getenv('YOUR_TELEGRAM_ID', '0'))

@router.post("/remote", operation_id="open_vscode_tunnel")
async def open_vscode_tunnel(user_id: int = Depends(get_user_id)):
    if not validate_mrzaizai2k_user(user_id):
        raise HTTPException(status_code=403, detail="This endpoint can only be used by the owner (mrzaizai2k)")
    try:
        # Note: run_vscode_tunnel is threaded in the original; here we assume it returns a result
        result = run_vscode_tunnel()  # Adjust based on actual implementation
        return {"message": "VS Code remote tunnel opened"}
    except Exception as e:
        logger.debug(msg=f"Error opening VS Code tunnel: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error opening VS Code tunnel: {str(e)}")

@router.get("/log", operation_id="get_log_file")
async def get_log_file(user_id: int = Depends(get_user_id)):
    if not validate_mrzaizai2k_user(user_id):
        raise HTTPException(status_code=403, detail="This endpoint can only be used by the owner (mrzaizai2k)")
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

# @router.post("/scrape", operation_id="scrape_trading_data_and_news")
# async def scrape_data(user_id: int = Depends(get_user_id)):
#     if not validate_mrzaizai2k_user(user_id):
#         raise HTTPException(status_code=403, detail="This endpoint can only be used by the owner (mrzaizai2k)")
#     try:
#         TRADE_USER = os.getenv('TRADE_USER')
#         TRADE_PASS = os.getenv('TRADE_PASS')
#         scrape_trading_data(user_name=TRADE_USER, password=TRADE_PASS)
#         summary_news_daily()
#         logger.info(msg="Done scraping trading data and updating news")
#         return {"message": "Done scraping trading data and updating news"}
#     except Exception as e:
#         logger.debug(msg=f"Error scraping data: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error scraping data: {str(e)}")

@router.post("/masterquest", operation_id="masterquest_query")
async def masterquest(request: MasterQuestRequest, user_id: int = Depends(get_user_id)):
    if not validate_mrzaizai2k_user(user_id):
        raise HTTPException(status_code=403, detail="This endpoint can only be used by the owner (mrzaizai2k)")
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

@router.post("/update-vectordb", operation_id="update_vector_database")
async def update_vector_db(user_id: int = Depends(get_user_id)):
    if not validate_mrzaizai2k_user(user_id):
        raise HTTPException(status_code=403, detail="This endpoint can only be used by the owner (mrzaizai2k)")
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