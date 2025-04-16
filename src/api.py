import sys
sys.path.append("")

from fastapi import FastAPI
import uvicorn
from fastapi_mcp import FastApiMCP
from src.apis.stock_apis import router as stocks_router
from src.apis.mcp_apis import router as mcp_router
from src.Utils.utils import config_parser

app = FastAPI(title="Stock Bot API", description="API for stock analysis and control panel operations")

# Include routers
app.include_router(stocks_router)
app.include_router(mcp_router)

# Read configuration
data = config_parser(data_config_path='config/config.yaml')

# Add MCP server to the FastAPI app
mcp = FastApiMCP(
    app,
    name="Item API MCP",
    description="MCP server for the Item API",
    base_url="http://localhost:8668",
)

mcp.mount()

mcp.setup_server()

if __name__ == "__main__":
    host = data.get('api_host', '0.0.0.0')
    port = data.get('api_port', 8000)
    uvicorn.run(app, host=host, port=port)