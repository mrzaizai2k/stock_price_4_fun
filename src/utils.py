import sys
sys.path.append("")
import pandas as pd
import os
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


def memoization(func):
    def wrapper(file_path, *args, **kwargs):
        # Check if file or folder exists, create if not
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


        # Check if the CSV file has the current date
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Date", "Value"])

        current_date = datetime.now().strftime("%Y-%m-%d")

        if current_date in df["Date"].values:
            # If the current date exists, return the value
            value = df.loc[df["Date"] == current_date, "Value"].iloc[0]
            # Split the value if it's a string
            value = value.split(',')
        else:
            # If the current date doesn't exist, call the original function
            value = func(file_path, *args, **kwargs)
            df = df.iloc[1:].copy()
            # Update CSV file with the new value and date
            new_row = {"Date": current_date, "Value": ','.join(value)}
            df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
            df.to_csv(file_path, index=False)
            print(f"File updated with pass_ticker for {current_date} in {file_path}")

        return value

    return wrapper