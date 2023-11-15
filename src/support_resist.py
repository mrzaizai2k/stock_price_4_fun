import sys
sys.path.append("")
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import stumpy
from vnstock import *

from datetime import datetime,timedelta

import matplotlib.pyplot as plt
import urllib3

urllib3.disable_warnings()

# Data Visualization
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.io as pio
from IPython.display import display
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

# Statistics & Mathematics
import scipy.stats as stats
import statsmodels as sm
from scipy.stats import shapiro, skew

from typing import Literal
from src.utils import *
from src.stock_class import Stock

class SupportResistFinding(Stock):
    def __init__(self, symbol:str = 'MWG',
                 start_date = None, end_date = None,
                 accuracy:float = 5.0,
                 percent_threshold:float = 6.0,
                   ):
        '''
        Find support and resisitance of a stock
        T = The window_size (default: 20 is 1 month trading)
        
        '''
        super().__init__(symbol)

        self.end_date = end_date
        if self.end_date is None:
            self.end_date = datetime.now().strftime("%Y-%m-%d")

        self.start_date = start_date
        if self.start_date is None:
            self.start_date = (
                datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=365)
            ).strftime("%Y-%m-%d")

        self.accuracy = accuracy
        self.percent_threshold = percent_threshold
        self.dataframe = self.load_data().set_index(self.time_col)

        
    def load_data(self):
        data = stock_historical_data(self.symbol, self.start_date ,self.end_date, "1D", type = self.type)
        data = convert_data_type(data, [self.time_col], self.float_cols, self.cat_cols)
        return data
    
    def calculate_support_resistance_band(self):
        
        supports = self.dataframe[self.dataframe.low == self.dataframe.low.rolling(5, center = True).min()].low
        resistances = self.dataframe[self.dataframe.high == self.dataframe.high.rolling(5, center = True).min()].high
        levels = pd.concat([supports, resistances])
        levels = levels[abs(levels.diff()) > self.filter_levels()].tolist()
        levels.sort()
        levels = self.combine_close_numbers(levels)
        return levels
    
    def filter_levels(self):
        '''Make levels more accurate'''
        diff_val = (max(self.dataframe.close) - min(self.dataframe.close)) / self.accuracy
        return diff_val


    def combine_close_numbers(self, levels_list:list, ):
        """
        Combine close numbers in a list into their average.

        Parameters:
        - numbers (list): A list of numbers.
        - percent_threshold (float): The threshold percentage to consider numbers close.

        Returns:
        - list: List of combined numbers.
        """
        sorted_numbers = np.array(levels_list)
        diff_percent = np.abs(np.diff(sorted_numbers) / sorted_numbers[:-1]) * 100

        combined_numbers = [sorted_numbers[0]]
        current_group = [sorted_numbers[0]]

        for i, diff in enumerate(diff_percent):
            if diff <= self.percent_threshold:
                current_group.append(sorted_numbers[i + 1])
            else:
                # Combine numbers in the current group into their average
                average_number = np.mean(current_group)
                combined_numbers.append(average_number)

                # Start a new group with the current number
                current_group = [sorted_numbers[i + 1]]

        # Combine the last group
        if current_group:
            average_number = np.mean(current_group)
            combined_numbers.append(average_number)

        return combined_numbers


    def plot_candlestick_with_levels(self, fig_size=(1000, 800)):

        levels = self.calculate_support_resistance_band()

        # Create Candlestick chart
        candlestick = go.Candlestick(x=self.dataframe.index,
                                     open=self.dataframe['open'],
                                     high=self.dataframe['high'],
                                     low=self.dataframe['low'],
                                     close=self.dataframe['close'],
                                     increasing=dict(line=dict(color='green')),
                                     decreasing=dict(line=dict(color='red')))


        # Create layout
        layout = go.Layout(title='Candlestick Chart with Supports and Resistances',
                           xaxis=dict(title='Date'),
                           yaxis=dict(title='Price'),
                           template = 'seaborn',
                            plot_bgcolor = '#F6F5F5',
                            paper_bgcolor = '#F6F5F5',)

        # Create figure
        fig = go.Figure(data=[candlestick], layout=layout)
        
        # Add support and resistance lines
        for line_value in levels:
            fig.add_hline(y=line_value, line_width=2, line_color='blue')


        # Update figure size
        fig.update_layout(width=fig_size[0], height=fig_size[1])

        # Show the plot
        fig.show()
    
    def find_closest_support_resist(self, current_price:float) -> list:
        price_list = self.calculate_support_resistance_band()
        # Find the closest support (equal to or less than the current price)
        supports = [price for price in price_list if price <= current_price]
        closest_support = max(supports, default=None)

        # Find the closest resistance (greater than the current price)
        resistance_candidates = [price for price in price_list if price > current_price]
        closest_resistance = min(resistance_candidates, default=None)

        return [closest_support, closest_resistance]
