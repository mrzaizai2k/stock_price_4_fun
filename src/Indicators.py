import sys
sys.path.append("")
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from vnstock import *

from datetime import datetime,timedelta

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import urllib3

urllib3.disable_warnings()

from src.stock_class import Stock
from src.Utils.utils import *

class MACD(Stock):
    def __init__(self, symbol:str = 'MWG', short_window=12, long_window=26, signal_window=9,
                 start_date:str =None, end_date:str = None):
        super().__init__(symbol)
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
        self.end_date = end_date
        if self.end_date is None:
            self.end_date = datetime.now().strftime("%Y-%m-%d")

        self.start_date = start_date
        if self.start_date is None:
            self.start_date = (
                datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=60)
            ).strftime("%Y-%m-%d")

        self.dataframe = self.load_data()
        self.dataframe = self.calculate_macd()


        
    def load_data(self):
        data = stock_historical_data(self.symbol, self.start_date ,self.end_date, "1D", type = self.type)
        data = convert_data_type(data, [self.time_col], self.float_cols, self.cat_cols)
        return data

    def calculate_macd(self):  
        if 'macd_hist' in self.dataframe:
            return self.dataframe
        self.dataframe = self.dataframe.copy()
        self.dataframe['short_ema'] = self.dataframe['close'].ewm(span=self.short_window, adjust=False).mean()
        self.dataframe['long_ema'] = self.dataframe['close'].ewm(span=self.long_window, adjust=False).mean()
        self.dataframe['macd'] = self.dataframe['short_ema'] - self.dataframe['long_ema']
        self.dataframe['signal_line'] = self.dataframe['macd'].ewm(span=self.signal_window, adjust=False).mean()
        self.dataframe['macd_hist'] = self.dataframe['macd'] - self.dataframe['signal_line']
        return self.dataframe


    def is_cross_up(self, offset = 3):
        # Check if there is a crossover from negative to positive in MACD
        past_macd = self.dataframe['macd_hist'].iloc[-1-offset]
        today_macd = self.dataframe['macd_hist'].iloc[-1]
        
        return (past_macd < 0) & (today_macd > 0)

    def is_cross_down(self, offset = 3):
        # Check if there is a crossover from positive to negative in MACD
        past_macd = self.dataframe['macd_hist'].iloc[-1-offset]
        today_macd = self.dataframe['macd_hist'].iloc[-1]
        
        return (past_macd > 0) & (today_macd < 0)


    def plot_macd(self, figsize=(1500, 800)):

        # Create a Plotly Figure object
        fig = go.Figure()

        # Plot MACD Line and Signal Line
        fig.add_trace(go.Scatter(x=self.dataframe['time'], y=self.dataframe['macd'], mode='lines', name='MACD Line', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=self.dataframe['time'], y=self.dataframe['signal_line'], mode='lines', name='Signal Line', line=dict(color='orange')))

        # Plot MACD Histogram with color
        color_scale = [[0, 'red'], [0.5, 'red'], [0.5, 'green'], [1, 'green']]
        fig.add_trace(go.Bar(x=self.dataframe['time'],
                             y=self.dataframe['macd_hist'],
                             name='MACD Histogram',
                             marker=dict(colorscale=color_scale, color=(self.dataframe['macd_hist']))))

        # Plot Stock Data using Candlestick
        fig.add_trace(go.Candlestick(x=self.dataframe['time'],
                                     open=self.dataframe['open'],
                                     high=self.dataframe['high'],
                                     low=self.dataframe['low'],
                                     close=self.dataframe['close'],
                                     name='Stock Data'))

        # Update Layout
        fig.update_layout(title='MACD and Signal Line with Histogram and Stock Data',
                          xaxis_title='Time',
                          yaxis_title='Values',
                          width=figsize[0],
                          height=figsize[1],
                          showlegend=True,  # Show legend
                          coloraxis_colorbar=dict(title='MACD Histogram Color Scale'))  # Color scale information

        # Show the plot
        fig.show()

class PricevsMA(Stock):
    def __init__(self, symbol:str = 'MWG', 
                 window_size:list[int] = [12,26,50],
                 start_date:str = None,
                 end_date:str = None,
                 ):
        super().__init__(symbol)
        self.window_size = window_size

        self.end_date = end_date
        if self.end_date is None:
            self.end_date = datetime.now().strftime("%Y-%m-%d")

        self.start_date = start_date
        if self.start_date is None:
            self.start_date = (
                datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=max(window_size))
            ).strftime("%Y-%m-%d")

        self.dataframe = self.load_data()
        self.calculate_ma()
        
        
    def load_data(self):
        data = stock_historical_data(self.symbol, self.start_date, self.end_date, "1D", type = self.type)
        data = convert_data_type(data, [self.time_col], self.float_cols, self.cat_cols)
        return data

    def calculate_ma(self): 
        for window in self.window_size:
            new_col_name = f'{window}_ema'
            self.dataframe[new_col_name] = self.dataframe['close'].ewm(span=window, adjust=False).mean()

    def is_cross_up(self, offset = 1):

        current_price = self.get_current_price()
        past_price = self.dataframe['close'].iloc[-1 - offset]

        # Loop through each window size
        for window in self.window_size:
            # Get the current and past EMA values for the current window size
            current_ema = self.dataframe[f'{window}_ema'].iloc[-1]
            past_ema = self.dataframe[f'{window}_ema'].iloc[-1 - offset]

            # Check if the price is crossing up the current EMA
            if (past_price < past_ema) and (current_price > current_ema):
                return True  # Return True if the price crosses any window size

        return False  
    
    def is_cross_down(self, offset = 1):

        current_price = self.get_current_price()
        past_price = self.dataframe['close'].iloc[-1 - offset]

        # Loop through each window size
        for window in self.window_size:
            # Get the current and past EMA values for the current window size
            current_ema = self.dataframe[f'{window}_ema'].iloc[-1]
            past_ema = self.dataframe[f'{window}_ema'].iloc[-1 - offset]

            # Check if the price is crossing up the current EMA
            if (past_price > past_ema) and (current_price < current_ema):
                return True  # Return True if the price crosses any window size

        return False  # Return False if the price doesn't cross any window size


class BigDayWarning(Stock):
    '''Consider if that day has powerful Increase of Decrease'''
    def __init__(self, symbol:str = 'MWG', 
                 window_size:int = 20,
                 percent_diff = 3,
                 start_date:str = None,
                 end_date:str = None,
                 ):
        super().__init__(symbol)
        
        self.end_date = end_date
        if self.end_date is None:
            self.end_date = datetime.now().strftime("%Y-%m-%d")

        self.start_date = start_date
        if self.start_date is None:
            self.start_date = (
                datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=window_size+1)
            ).strftime("%Y-%m-%d")

        self.dataframe = self.load_data()
        self.window_size = window_size
        self.percent_diff = percent_diff

        
    def load_data(self):
        data = stock_historical_data(self.symbol, self.start_date ,self.end_date, "1D", type = self.type)
        data = convert_data_type(data, [self.time_col], self.float_cols, self.cat_cols)
        return data
    
    def calculate_avg_volume(self):
        return  self.dataframe['volume'].iloc[-1 - self.window_size: -1].mean()
    
    def is_big_increase(self):
        current_vol = self.get_current_volume()
        current_price = self.get_current_price()
        
        if current_vol >= self.calculate_avg_volume() and current_price > (1 + (self.percent_diff/100)) * self.dataframe.close.iloc[-2]:
            return True
        return False

    def is_big_decrease(self):
        current_vol = self.get_current_volume()
        current_price = self.get_current_price()

        # Assuming your class has a dataframe attribute named 'dataframe'
        yesterday_close_price = self.dataframe.close.iloc[-2]

        # Calculate the threshold for a significant decrease
        threshold_price = (1 - (self.percent_diff / 100)) * yesterday_close_price

        # Check if both volume and price have decreased significantly
        if current_vol >= self.calculate_avg_volume() and current_price < threshold_price:
            return True
        return False


def main():
    symbol = "SSI"
    macd = MACD(symbol)
    a = macd.calculate_macd()
    pvma = PricevsMA(symbol)
    print(pvma.is_cross_up(offset=3))
    bigday = BigDayWarning(symbol)
    print(bigday.is_big_decrease())

if __name__ =="__main__":
    main()