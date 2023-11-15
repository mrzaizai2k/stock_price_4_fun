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
from src.utils import *

class MACD(Stock):
    def __init__(self, symbol:str = 'MWG', short_window=12, long_window=26, signal_window=9):
        super().__init__(symbol)
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
        self.dataframe = self.load_data()
        self.calculate_macd()
        
        
    def load_data(self):
        curent_date = datetime.now().strftime("%Y-%m-%d")
        data = stock_historical_data(self.symbol, "2007-01-01", curent_date, "1D", type = self.type)
        data = convert_data_type(data, [self.time_col], self.float_cols, self.cat_cols)
        return data

    def calculate_macd(self):  
        self.dataframe['short_ema'] = self.dataframe['close'].ewm(span=self.short_window, adjust=False).mean()
        self.dataframe['long_ema'] = self.dataframe['close'].ewm(span=self.long_window, adjust=False).mean()
        self.dataframe['macd'] = self.dataframe['short_ema'] - self.dataframe['long_ema']
        self.dataframe['signal_line'] = self.dataframe['macd'].ewm(span=self.signal_window, adjust=False).mean()
        self.dataframe['macd_hist'] = self.dataframe['macd'] - self.dataframe['signal_line']


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
                 ):
        super().__init__(symbol)
        self.window_size = window_size
        self.dataframe = self.load_data()
        self.calculate_ma()
        
        
    def load_data(self):
        curent_date = datetime.now().strftime("%Y-%m-%d")
        data = stock_historical_data(self.symbol, "2007-01-01", curent_date, "1D", type = self.type)
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



    def is_cross_down(self, offset = 1):
        # Check if there is a crossover from positive to negative in MACD
        past_macd = self.dataframe['macd_hist'].iloc[-1-offset]
        today_macd = self.dataframe['macd_hist'].iloc[-1]
        
        return (past_macd > 0) & (today_macd < 0)

class FTDWarning(Stock):
    def __init__(self, symbol:str = 'MWG', 
                 window_size:int = 20,
                 percent_diff = 3,
                 ):
        super().__init__(symbol)
        self.dataframe = self.load_data()
        self.window_size = window_size
        self.percent_diff = percent_diff
        
        
    def load_data(self):
        curent_date = datetime.now().strftime("%Y-%m-%d")
        data = stock_historical_data(self.symbol, "2007-01-01", curent_date, "1D", type = self.type)
        data = convert_data_type(data, [self.time_col], self.float_cols, self.cat_cols)
        return data
    
    def calculate_avg_volume(self):
        return  self.dataframe['volume'].iloc[-1 - self.window_size: -1].mean()
    
    def is_FTD(self):
        current_vol = self.get_current_volume()
        current_price = self.get_current_price()
        
        if current_vol >= self.calculate_avg_volume() and current_price > (1 + (self.percent_diff/100)) * self.dataframe.close.iloc[-2]:
            return True
        return False


