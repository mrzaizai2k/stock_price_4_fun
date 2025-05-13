import sys
sys.path.append("")

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import os
import requests
from tavily import TavilyClient
from datetime import datetime


from src.trading_record import scrape_trading_data
from src.Utils.utils import config_parser
from src.Utils.logger import create_logger
from src.Microsofttodo import MicrosoftToDo

router = APIRouter(prefix="/mcp", tags=["Miscellaneous Control Panel"])

data = config_parser(data_config_path='config/config.yaml')
logger = create_logger(logfile="logging/mcp_apis.log")

class MasterQuestRequest(BaseModel):
    query: str

class ScrapeRequest(BaseModel):
    pass

class TavilySearchRequest(BaseModel):
    query: str

def get_todo_client():
    return MicrosoftToDo()

# Pydantic model for create task request
class CreateTaskRequest(BaseModel):
    task_name: str
    importance: Optional[bool] = False
    due_date_time: Optional[str] = None  # Format: YYYY-MM-DD:HH:MM:SS
    body: Optional[dict] = None
    reminder_date_time: Optional[str] = None  # Format: YYYY-MM-DD:HH:MM:SS


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

@router.post(
    "/tavily-search",
    operation_id="tavily_search",
)
async def tavily_search(request: TavilySearchRequest):
    """Performs a search using the Tavily API"""
    try:
        TAVILY_KEY = os.getenv('TAVILY_KEY')
        if not TAVILY_KEY:
            raise HTTPException(status_code=500, detail="Tavily API key not configured")
        
        client = TavilyClient(TAVILY_KEY)
        response = client.search(query=request.query)
        logger.debug(msg=f"Tavily search result: {response}")
        return {"results": response}
    except Exception as e:
        logger.debug(msg=f"Error performing Tavily search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error performing Tavily search: {str(e)}")
    


# Endpoint to get tasks from "Tasks" list
@router.get(
    "/tasks",
    operation_id="get_tasks",
)
async def get_tasks(
    num_tasks: int = 100,
    get_completed: bool = False,
    todo: MicrosoftToDo = Depends(get_todo_client)
):
    """Retrieve tasks from the 'Tasks' list in Microsoft ToDo."""
    try:
        tasks = todo.get_tasks(
            list_name="Tasks",
            num_tasks=num_tasks,
            get_completed=get_completed
        )
        logger.debug(f"Retrieved {len(tasks)} tasks from 'Tasks' list")
        return {"tasks": tasks}
    except Exception as e:
        # Check if the error is related to the list not being found
        error_message = str(e)
        if "404" in error_message or "not found" in error_message.lower():
            logger.error(f"Tasks list not found: {error_message}")
            raise HTTPException(status_code=404, detail="Tasks list not found")
        logger.error(f"Error retrieving tasks: {error_message}")
        raise HTTPException(status_code=500, detail=f"Error retrieving tasks: {error_message}")

# Endpoint to create a task in "Tasks" list
@router.post(
    "/tasks",
    operation_id="create_task",
)
async def create_task(
    request: CreateTaskRequest,
    todo: MicrosoftToDo = Depends(get_todo_client)
):
    """Create a new task in the 'Tasks' list in Microsoft ToDo."""
    try:

        # Create the task
        result = todo.create_task(
            task_name=request.task_name,
            list_name="Tasks",
            importance=request.importance,
            dueDateTime=request.due_date_time,
            body=request.body,
            reminder_datetime=request.reminder_date_time
        )
        if not result:
            raise HTTPException(status_code=500, detail="Failed to create task")
        logger.debug(f"Created task '{request.task_name}' in 'Tasks' list")
        return {"status": "success", "message": f"Task '{request.task_name}' created successfully"}
    except Exception as e:
        # Check for specific error cases
        error_message = str(e)
        if "404" in error_message or "not found" in error_message.lower():
            logger.error(f"Tasks list not found: {error_message}")
            raise HTTPException(status_code=404, detail="Tasks list not found")
        if "400" in error_message or "invalid" in error_message.lower():
            logger.error(f"Invalid request: {error_message}")
            raise HTTPException(status_code=400, detail=f"Invalid request: {error_message}")
        logger.error(f"Error creating task: {error_message}")
        raise HTTPException(status_code=500, detail=f"Error creating task: {error_message}")