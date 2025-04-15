import sys
sys.path.append("")


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi_mcp import FastApiMCP
from typing import Literal, List
from src.PayBackTime import PayBackTime, find_PBT_stocks
from src.Indicators import MACD, PricevsMA, BigDayWarning

app = FastAPI(title="Stock Analysis API", description="API for stock analysis using PayBackTime and technical indicators")

# Pydantic models for request validation
class SymbolRequest(BaseModel):
    symbol: str
    report_range: Literal['yearly', 'quarterly'] = 'yearly'
    window_size: int = 10

class MACDRequest(BaseModel):
    symbol: str
    short_window: int = 12
    long_window: int = 26
    signal_window: int = 9
    offset: int = 3

class PricevsMARequest(BaseModel):
    symbol: str
    window_size: List[int] = [12, 26, 50]
    offset: int = 1

class BigDayRequest(BaseModel):
    symbol: str
    window_size: int = 20
    percent_diff: float = 3

# PayBackTime endpoints
@app.post("/paybacktime/", response_model=dict)
async def get_paybacktime_report(request: SymbolRequest):
    try:
        pbt = PayBackTime(
            symbol=request.symbol,
            report_range=request.report_range,
            window_size=request.window_size
        )
        report = pbt.get_report()
        return {"symbol": request.symbol, "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing {request.symbol}: {str(e)}")

@app.get("/pbt_stocks", response_model=List[str])
async def get_pbt_stocks():
    try:
        stocks = find_PBT_stocks()
        return stocks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding PBT stocks: {str(e)}")

# Technical Indicator endpoints
@app.post("/macd/", response_model=dict)
async def get_macd_signals(request: MACDRequest):
    try:
        macd = MACD(
            symbol=request.symbol,
            short_window=request.short_window,
            long_window=request.long_window,
            signal_window=request.signal_window
        )
        cross_up = macd.is_cross_up(offset=request.offset)
        cross_down = macd.is_cross_down(offset=request.offset)
        return {
            "symbol": request.symbol,
            "cross_up": cross_up,
            "cross_down": cross_down
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing MACD for {request.symbol}: {str(e)}")

@app.post("/pricevsma/", response_model=dict)
async def get_pricevsma_signals(request: PricevsMARequest):
    try:
        pvma = PricevsMA(
            symbol=request.symbol,
            window_size=request.window_size
        )
        cross_up = pvma.is_cross_up(offset=request.offset)
        cross_down = pvma.is_cross_down(offset=request.offset)
        return {
            "symbol": request.symbol,
            "cross_up": cross_up,
            "cross_down": cross_down
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PricevsMA for {request.symbol}: {str(e)}")

@app.post("/bigday/", response_model=dict)
async def get_bigday_signals(request: BigDayRequest):
    try:
        bigday = BigDayWarning(
            symbol=request.symbol,
            window_size=request.window_size,
            percent_diff=request.percent_diff
        )
        big_increase = bigday.is_big_increase()
        big_decrease = bigday.is_big_decrease()
        return {
            "symbol": request.symbol,
            "big_increase": big_increase,
            "big_decrease": big_decrease
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing BigDayWarning for {request.symbol}: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Add MCP server to the FastAPI app
mcp = FastApiMCP(
    app,
    name="Item API MCP",
    description="MCP server for the Item API",
    base_url="http://localhost:8668",
)


# MCP server
mcp.mount()

mcp.setup_server()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8668)