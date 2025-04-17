import sys
sys.path.append("")

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from src.PayBackTime import PayBackTime, find_PBT_stocks
from src.stock_class import Stock
from src.motif import MotifMatching, BestMarketMotifSearch
from src.support_resist import SupportResistFinding
from src.trading_record import BuySellAnalyzer, WinLossAnalyzer, AssetAnalyzer
from src.summarize_text import NewsSummarizer, NewsScraper
from src.Utils.utils import filter_stocks, general_rating, config_parser, validate_symbol
from functools import wraps

router = APIRouter(prefix="/stocks", tags=["Stock Operations"])

data = config_parser(data_config_path='config/config.yaml')

class SymbolRequest(BaseModel):
    symbol: str

class PatternRequest(BaseModel):
    symbol: str
    start_date: str  # Format: YYYY-mm-dd

class NewsSummaryRequest(BaseModel):
    url: str

def validate_symbol_decorator(func):
    @wraps(func)
    async def wrapper(request: SymbolRequest | PatternRequest, *args, **kwargs):
        if not validate_symbol(request.symbol.upper()):
            raise HTTPException(status_code=400, detail=f"Invalid symbol: {request.symbol}")
        return await func(request, *args, **kwargs)
    return wrapper

@router.post("/paybacktime")
@validate_symbol_decorator
async def get_paybacktime(request: SymbolRequest):
    try:
        pbt_params = data.get('pbt_params')
        pbt_generator = PayBackTime(symbol=request.symbol.upper(), report_range=pbt_params[0], window_size=pbt_params[1])
        report = pbt_generator.get_report()
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating payback time: {str(e)}")

@router.post("/support-resistance")
@validate_symbol_decorator
async def get_support_resistance(request: SymbolRequest):
    try:
        sr_finding = SupportResistFinding(symbol=request.symbol.upper())
        result = sr_finding.find_closest_support_resist(current_price=sr_finding.get_current_price())
        report = (
            f"The current price for {request.symbol.upper()} is {sr_finding.get_current_price()}\n"
            f"- The closest support is {round(result[0], 2)}\n"
            f"- The closest resistance is {round(result[1], 2)}\n"
        )
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating support/resistance: {str(e)}")

@router.get("/find-paybacktime-stocks")
async def find_paybacktime_stocks():
    try:
        pass_ticker = find_PBT_stocks(file_path="memory/paybacktime.csv")
        return {"stocks": pass_ticker}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding payback time stocks: {str(e)}")

@router.get("/find-favorite-stocks")
async def find_favorite_stocks():
    try:
        my_params = {
            "exchangeName": "HOSE,HNX",
            "marketCap": (1_000, 200_000),
            "roe": (10, 100),
            "pe": (10, 20),
            "priceNearRealtime": (10, 100),
            "macdHistogram": "macdHistLT0Increase",
            "strongBuyPercentage": (20, 100),
            "relativeStrength3Day": (50, 100),
        }
        pass_ticker = filter_stocks(param=my_params)
        pass_ticker_string = ", ".join(pass_ticker.ticker.unique())
        return {"stocks": pass_ticker_string}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding favorite stocks: {str(e)}")

@router.post("/risk")
@validate_symbol_decorator
async def calculate_risk(request: SymbolRequest):
    try:
        stock_generator = Stock(symbol=request.symbol.upper())
        stock_price = stock_generator.get_current_price()
        asset_data_path = data.get('asset_data_path')
        data_frame_reader = AssetAnalyzer(file_path=asset_data_path)
        try:
            capital_value = data_frame_reader.read_capital_value()
        except:
            capital_value = 100_000_000

        from src.Utils.utils import calculate_stocks_to_buy
        num_stocks = calculate_stocks_to_buy(stock_price, capital=capital_value)
        mess = (
            f"You can buy {num_stocks} stocks at the price of {stock_price} each\n"
            f"Your Capital: {capital_value/1_000_000:.2f} (triệu VND)\n"
            f"Total price: {stock_price*num_stocks/1_000_000:.2f} (triệu VND)\n"
            f"The fee is: 0.188% -> {((0.188/100) * stock_price*num_stocks)/1_000} (nghìn VND)\n"
        )
        return {"report": mess}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating risk: {str(e)}")

@router.post("/rate")
@validate_symbol_decorator
async def rate_stock(request: SymbolRequest):
    try:
        symbol = request.symbol.upper()
        if len(symbol) > 3:
            raise HTTPException(status_code=400, detail="This function can only be used for stocks, not indices")
        rating = general_rating(symbol)
        report = f"The general rating for {symbol}:\n"
        report += "\n".join([f"{col}: {rating[col].values[0]}" for col in rating.columns])
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rating stock: {str(e)}")

@router.post("/multi-pattern")
@validate_symbol_decorator
async def find_multi_pattern(request: PatternRequest):
    try:
        motif_matching = MotifMatching(symbol=request.symbol.upper(), start_date=request.start_date)
        image_path = motif_matching.plot_and_save_find_matching_series_multi_dim_with_date(save_fig=True)
        with open(image_path, 'rb') as photo:
            image_data = photo.read()
        os.remove(image_path)
        return {"image": image_data.hex(), "message": f"Multi-dimension pattern for {request.symbol.upper()}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding multi-dimension pattern: {str(e)}")

@router.post("/pattern")
@validate_symbol_decorator
async def find_pattern(request: PatternRequest):
    try:
        motif_matching = MotifMatching(symbol=request.symbol.upper(), start_date=request.start_date)
        image_path = motif_matching.plot_and_save_top_pattern(save_fig=True)
        with open(image_path, 'rb') as photo:
            image_data = photo.read()
        os.remove(image_path)
        return {"image": image_data.hex(), "message": f"Top 3 pattern for {request.symbol.upper()}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding pattern: {str(e)}")

@router.get("/find-best-motif")
async def find_best_motif():
    try:
        market_motif_search = BestMarketMotifSearch(motif_data_path=data.get('motif_data_path'))
        result_dict = market_motif_search.find_best_motifs()
        report = ""
        for stock, values in result_dict.items():
            start_date, end_date, distance = values
            report += f"Stock: {stock}\n"
            f"- Date: {start_date} to {end_date}\n"
            f"- Distance: {distance:.3f}\n\n"
        report += f"Use /mulpattern to see the pattern of each stock with date: {market_motif_search.start_date}"
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding best motif: {str(e)}")

@router.post("/buy-sell-analyze")
@validate_symbol_decorator
async def buy_sell_analyze(request: SymbolRequest):
    try:
        buy_sell_df_path = data.get('buy_sell_df_path', None)
        buysell_analyzer = BuySellAnalyzer(buy_sell_df_path=buy_sell_df_path)
        image_path = buysell_analyzer.plot_and_save_buy_sell_of_stock(symbol=request.symbol.upper())
        with open(image_path, 'rb') as photo:
            image_data = photo.read()
        os.remove(image_path)
        return {"image": image_data.hex(), "message": f"Buy/Sell analysis for {request.symbol.upper()}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing buy/sell: {str(e)}")

@router.get("/win-loss-analyze")
async def win_loss_analyze():
    try:
        win_loss_df_path = data.get('win_loss_df_path', None)
        winloss_analyzer = WinLossAnalyzer(win_loss_df_path=win_loss_df_path)
        report = winloss_analyzer.get_report()
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing win/loss: {str(e)}")

@router.post("/summary-news-url")
async def summary_news_url(request: NewsSummaryRequest):
    try:
        news_scraper = NewsScraper()
        new_summarizer = NewsSummarizer()
        news = news_scraper.take_text_from_link(news_url=request.url)
        sum_text = new_summarizer.summary_news(news=news)
        return {"summary": sum_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing news: {str(e)}")