import sys
sys.path.append("")
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import stumpy
from vnstock import *

from datetime import datetime,timedelta
from src.Utils.utils import *
import matplotlib.pyplot as plt
import urllib3

urllib3.disable_warnings()


class Stock:
    def __init__(self, symbol:str = 'MWG',
                   ):
        
        self.time_col = "time"
        self.float_cols = ["open", "high", "low", "close", "volume"]
        self.cat_cols = ["ticker"]

        self.symbol = symbol.upper()
        if len(self.symbol) > 3:
            self.type = "index"
        elif len(self.symbol) == 3:
            self.type = 'stock'

    def load_full_data(self, start_date:str = None, end_date:str = None, resolution:str ='1D'):
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if start_date is None:
            start_date = "2017-01-01"

        df =  stock_historical_data(symbol=self.symbol, start_date = start_date,
                    end_date=end_date, resolution=resolution, type=self.type, beautify=True)
        df = convert_data_type(df, [self.time_col], self.float_cols, self.cat_cols)
        return df


    def load_current_data(self):
        current_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=10)).strftime('%Y-%m-%d')
        df =  stock_historical_data(symbol=self.symbol, start_date = start_date,
                            end_date=current_date, resolution='1H', type=self.type, beautify=True)
        return df
    
    def get_current_price(self):
        df = self.load_current_data()
        return df.close.iloc[-1]
    
    def get_current_volume(self):
        df = self.load_current_data()
        daily_volume_sum = df.groupby(df['time'].dt.date)['volume'].sum().reset_index()
        return daily_volume_sum.volume.iloc[-1]
    
if __name__ == "__main__":
    stock = Stock('MWG')
    print(stock.get_current_price())
