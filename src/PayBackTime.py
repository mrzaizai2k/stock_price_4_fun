import sys
sys.path.append("")
import os
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib as plt
from vnstock import *
import math
from typing import Literal
from datetime import datetime,timedelta
from src.utils import filter_stocks
from src.stock_class import Stock
from src.utils import memoization




class PayBackTime(Stock):
    def __init__(self, symbol: str = 'GEX',
                 report_range: Literal['yearly', 'quarterly'] = 'yearly',
                 is_all: bool = True,
                 window_size:int = 10,):
        super().__init__(symbol)
        self.window_size = window_size
        self.report_range = report_range
        self.is_all = is_all
        self.sticker_price, self.MOS_price = None,None

        # Initialize indicator_df, income_df, and balance_df using class variables
        self.indicator_df = financial_ratio(symbol=self.symbol, report_range=self.report_range, is_all=self.is_all).T.reset_index() 
        self.income_df = financial_flow(symbol=self.symbol, report_type='incomestatement', report_range=self.report_range).reset_index()
        self.balance_df = financial_flow(symbol=self.symbol, report_type='balancesheet', report_range=self.report_range).reset_index()
        self.interest_cols = ['roe', 'earningPerShare', 'bookValuePerShare', 'revenue', 'grossProfit', 'capital']

    def evaluate_interest_rate(self, interest, threshold:float = 10.0):
        if interest is None:
            return 'Neutral'
        if interest > threshold:
            status = 'Good'
        elif interest < 0:
            status = 'Really Bad'
        else:
            status = 'Bad'
        return status

    def calculate_interest_rate(self, df, column_name: str):
        # Consider the first 10 rows using iloc
        selected_rows = df.copy()
        time_length = selected_rows[column_name].count()
        # Calculate interest using the formula: (current_value - initial_value) / initial_value * 100
        initial_value = selected_rows[column_name].iloc[time_length-1]
        if initial_value is None:
            return None
        current_value = selected_rows[column_name].iloc[0]
        # interest = ((current_value - initial_value) / initial_value) * 100
        interest =  ((current_value / initial_value) ** (1 / time_length) - 1) * 100
        return interest
    
    def estimate_mean_interest(self):
        return self.create_interest_dataframe()['Interest'].mean()
    
    def create_interest_dataframe(self):
        data = []
        for col in self.interest_cols:
            df = self.get_dataframe_by_column(col).head(self.window_size)
            interest = self.calculate_interest_rate(df, column_name=col)
            status = self.evaluate_interest_rate(interest)
            time_length = df[col].count()
            data.append([col, interest, status, time_length])

        # Create a DataFrame with columns 'Feature', 'Interest', 'Status', and 'Time Length'
        interest_df = pd.DataFrame(data, columns=['Feature', 'Interest', 'Status', 'Time_Length'])
        return interest_df
    
    def get_dataframe_by_column(self, column_name: str):
        """
        Get the DataFrame that contains the specified column.
        """
        dataframes = [self.indicator_df, self.income_df, self.balance_df]
        for df in dataframes:
            if column_name in df.columns:
                return df
        return None
    
    def check_debt(self):
        debt_col = 'debt'
        profit_col = 'grossProfit'
        debt = self.balance_df[debt_col].iloc[0]
        profit = self.income_df[profit_col].iloc[:2].sum()
        
        print(f'Debt: {debt} - Profit: {profit}')
        
        if debt > profit:
            print('Debt is more than profit.')
        elif debt < profit:
            print('Profit is more than debt.')
        else:
            print('Debt and profit are equal.')

    def get_current_eps(self):
        indicator_df_temp = financial_ratio(symbol=self.symbol, report_range='quarterly', is_all=False).T.reset_index().head(1) 
        return indicator_df_temp['earningPerShare'].iloc[0]
    
    def get_future_pe(self):
        return self.indicator_df['priceToEarning'].mean()



    def calculate_price(self, current_eps=None, future_pe=None, future_growth_rate=None,
                        MARR: int = 15, NoY: int = 10, MOS: int = 50):
        if current_eps is None:
            current_eps = self.get_current_eps()
        if future_pe is None:
            future_pe = self.get_future_pe()
        if future_growth_rate is None:
            future_growth_rate = self.estimate_mean_interest()

        # Convert the percentage rates to decimals
        MARR /= 100
        MOS /= 100
        future_growth_rate /= 100

        # Calculate the future EPS using the compound annual growth rate formula
        future_eps = current_eps * (1 + future_growth_rate) ** NoY
        # Calculate the future price using the future PE and future EPS
        future_price = future_pe * future_eps
        # Calculate the present value using the discounted cash flow formula
        self.sticker_price = future_price / (1 + MARR) ** NoY
        self.MOS_price = self.sticker_price*MOS
        return self.sticker_price, self.MOS_price

    def calculate_payback_time(self, price = None , growth:int = None):
        # Initialize the cumulative earnings and the year
        eps = self.get_current_eps()
        if eps < 0:
            return None
        
        if growth is None:
            growth = self.estimate_mean_interest()
        if price is None:
            price  = self.get_current_price()
        
        growth /= 100
        cumulative_earnings = 0
        year = 0
        # Loop until the cumulative earnings exceed the price
        while cumulative_earnings < price:
            # Increment the year by 1
            year += 1
            # Update the cumulative earnings by adding the EPS times the growth factor
            cumulative_earnings += eps * (1 + growth) ** year
        # Return the year as the payback time
        return year


    def get_report(self)->str:
        # Construct the report as a text string
        report = ""
        report += f"\nCurrent EPS: {self.get_current_eps()}\n"
        report += f"Future PE: {self.get_future_pe():.2f}\n"
        report += f"Future Growth Rate: {self.estimate_mean_interest():.2f}%\n"
        self.check_debt()
        interest_df = self.create_interest_dataframe()
                 
        for kind, interest, status, time_length  in zip(interest_df.Feature, interest_df.Interest, interest_df.Status, interest_df.Time_Length):
            report += f"\n{kind}: {interest:.2f}, {status}, {time_length}"

        if self.sticker_price is None:
            self.sticker_price, self.MOS_price = self.calculate_price()
        report += f"\n{self.symbol} - Sticker price: {self.sticker_price/1000:.2f} - MOS price: {(self.MOS_price)/1000:.2f}\n"
        report += f'Current price: {self.get_current_price()/1000:.2f}\n'
        report += f"The payback time for {self.symbol} is {self.calculate_payback_time()} years.\n"
        report += "-------------------------\n"
        
        print (report)
        # Return the report as a text string
        return report


def pbt_pre_filter():
    # https://www.youtube.com/watch?v=4YPyxfXML0A&t=1825s
    paybacktime_params = {
        "exchangeName": "HOSE",
        "marketCap": (1_000, 200_000),
        'revenueGrowth5Year': (15,100),
        'priceNearRealtime': (15,80),
        "lastQuarterProfitGrowth": (2, 100),  # Minimum last quarter profit growth
        # 'pe':(10,20),
        # "epsGrowth5Year": (10, 100),  # Minimum 1-year EPS growth
        # "roe": (15, 100),  # Minimum Return on Equity (ROE)
        

        # "avgTradingValue20Day": (100, 2000),  # Minimum 20-day average trading value
    #     "breakout": 'BULLISH',  # Only buy stocks when the market is in an uptrend
    }
    df = filter_stocks(paybacktime_params)      
    pbt_stocks = df.ticker.to_list()
    return pbt_stocks


@memoization
def find_PBT_stocks(file_path="memory/paybacktime.csv"):
    pbt_stocks = pbt_pre_filter()
    pass_ticker = []
    for stock in pbt_stocks:
        pbt_generator = pbt_generator = PayBackTime(symbol=stock, report_range='yearly', window_size=10)
        # pbt_generator.calculate_price()
        pbt_years = pbt_generator.calculate_payback_time()
        if pbt_years is not None and pbt_years <= 5:
            pass_ticker.append(pbt_generator.symbol)
    print(f"PBT stocks: {pass_ticker}")
    return pass_ticker
