import sys
sys.path.append("")
import pandas as pd
from vnstock import *

def filter_stocks(param):
    df = stock_screening_insights(param, size=1700, drop_lang='vi')
    if len(df)!= 0:
        print(f"Pre filter stocks: {df.ticker.unique()}")
    return df

def calculate_stocks_to_buy(stock_price, capital = 200_000_000, 
                            pct_loss_per_trade:float = 0.06, pct_total_loss:float = 0.02) -> int:

    max_total_loss = pct_total_loss * capital  # 2% of total capital as maximum total loss
    num_stocks_to_buy = 0
    
    while (pct_loss_per_trade * stock_price * num_stocks_to_buy) < max_total_loss:
        num_stocks_to_buy += 100
    return num_stocks_to_buy - 100

def convert_data_type(df, time_cols=[], float_cols=[], cat_cols=[]):
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], yearfirst=True)
    for col in float_cols:
        df[col] = df[col].astype(float)
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df

def validate_symbol(symbol):
    return (symbol in listing_companies(live=True).ticker.tolist()) or (symbol in ['VNINDEX','VN30'])