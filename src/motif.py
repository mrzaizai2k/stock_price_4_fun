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

from src.utils import *
from src.stock_class import Stock


class MotifMatching(Stock):
    def __init__(self, symbol:str = 'MWG',
                 start_date = None, end_date = None,
                   ):
        super().__init__(symbol)

        self.end_date = end_date
        if self.end_date is None:
            self.end_date = datetime.now().strftime("%Y-%m-%d")

        self.start_date = start_date
        if self.start_date is None:
            self.start_date = (
                datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=30)
            ).strftime("%Y-%m-%d")

        
        self.dataframe = self.load_data()
        self.filtered_data = self.filter_data()
        self.m = len(self.filtered_data)
        
    def load_data(self):
        data = stock_historical_data(self.symbol, "2007-01-01" ,self.end_date, "1D", type = self.type)
        data = convert_data_type(data, [self.time_col], self.float_cols, self.cat_cols)
        return data
    
    def filter_data(self):
        filtered_data = self.dataframe[
            (self.dataframe[self.time_col] >= self.start_date) & (self.dataframe[self.time_col] <= self.end_date)
        ]
        return filtered_data

    def format_date_from_idx(self, dataframe, index: int):
        """
        Format date from start index of dataframe and window size
        """
        date_timestamp = dataframe.iloc[index]
        # Format start date and end date in the same format as input
        formatted_date = date_timestamp.strftime("%Y-%m-%d")
        return formatted_date


    def find_similar_subseries_with_date(self, col: str = "close", plot = False):
        matrix_profile = stumpy.stump(self.dataframe[col], m=self.m)
        start_date_index = np.where(self.dataframe[self.time_col] == pd.to_datetime(self.start_date))[0][0]
        # Find the index of the most similar subsequence
        motif_index = matrix_profile[start_date_index, 1]
        distance_score = matrix_profile[start_date_index, 0]

        formatted_start_date = self.format_date_from_idx(dataframe = self.dataframe[self.time_col], index = motif_index)
        formatted_end_date = self.format_date_from_idx(dataframe = self.dataframe[self.time_col], index = motif_index + self.m - 1)

        print(f"Similar Subsequence from {self.start_date} to {self.end_date} (Indices: {formatted_start_date} to {formatted_end_date})")
        print(f'm: {self.m}')
        print(f"The distance score is {distance_score:.2f}")
        if plot:
            self.plot_time_series(self.start_date, self.end_date,title = "The original")
            self.plot_time_series(formatted_start_date, formatted_end_date,title = "The similar subseries")
        return formatted_start_date, formatted_end_date, distance_score
    
    def plot_time_series(self,
            start_date: str = "2023-06-30",
            end_date: str = "2023-10-15",
            col: str = "close",
            title: str = "The original",
            fig_size = (12, 6),
            ):


        # Extract data for plotting
        dates = self.filtered_data[self.time_col]
        close_prices = self.filtered_data[col]

        # Plot the time series
        plt.figure(figsize=fig_size)
        plt.plot(dates, close_prices, marker="o", linestyle="-", color="b")
        plt.title(f"Time Series from {start_date} to {end_date}")
        plt.xlabel(self.time_col)
        plt.ylabel(col)
        plt.grid(True)
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def find_top_pattern(self,
        col="close",
        top_k=3,
        ):
        Q_df = self.filtered_data[col]
        T_df = self.dataframe[col]
        distance_profile = stumpy.match(Q_df, T_df)
        k_index = distance_profile[:top_k, 1]

        print(f"Top {len(k_index)} similar periods")
        for i, idx in enumerate(k_index):
            
            formatted_start_date = self.format_date_from_idx(dataframe = self.dataframe[self.time_col], index = idx)
            formatted_end_date = self.format_date_from_idx(dataframe = self.dataframe[self.time_col], index = idx + self.m -1)
            distance = distance_profile[i, 0]
            print(f"From {formatted_start_date} to {formatted_end_date}: Distance = {distance}")

        plt.figure(figsize=(12, 6))
        plt.plot(T_df.values)
        for idx in k_index:
            plt.plot(
                range(idx, idx + len(Q_df)),
                T_df.values[idx : idx + len(Q_df)],
                lw=2,
                label=f"Series at index {idx}",
            )

        plt.xlabel("Index")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()

    
    def find_matching_series_multi_dim_with_date(self,
                                              dimension_cols:list=['close', 'volume'],
                                             nn_idx_threshold:int=None, distance_threshold:float = None,
                                             plot:bool=False):
        

        start_date_index = len(self.dataframe) - self.m
        df = self.dataframe[dimension_cols].reset_index(drop = True)
        mps, indices = stumpy.mstump(df, self.m)
        start_date_index = np.repeat(start_date_index, len(df.columns))
        nn_idx = indices[np.arange(len(df.columns)), start_date_index]
        # Check if the absolute difference between element 1 and element 2 is less than threshold
        if nn_idx_threshold is not None and np.abs(nn_idx[1] - nn_idx[0]) > nn_idx_threshold:
            return None, None
        
        distance = []
        for k in range(len(df.columns)):
            distance.append(mps[k, nn_idx[k]])
        mean_distance = np.mean(distance)

        if distance_threshold is not None and mean_distance > distance_threshold:
            return None, None
        
        result  = np.concatenate((nn_idx, np.array([mean_distance])))

        print(f'm: {self.m}')
        print(f'Ticker: {self.dataframe.iloc[nn_idx[0]].ticker}')
        print(f'From {self.dataframe.iloc[nn_idx[0]].time} to {self.dataframe.iloc[nn_idx[0] + self.m].time}')
        print(f'nn_idx: {nn_idx}')

        if plot:
            fig, axs = plt.subplots(mps.shape[0] * 2, sharex=True, gridspec_kw={'hspace': 0}, figsize=(14, 3 * mps.shape[0] * 2))
            for k, dim_name in enumerate(df.columns):
                axs[k].set_ylabel(dim_name, fontsize='20')
                axs[k].plot(df[dim_name])
                axs[k].set_xlabel('Time', fontsize ='20')

                axs[k + mps.shape[0]].set_ylabel(dim_name.replace('T', 'P'), fontsize='20')
                axs[k + mps.shape[0]].plot(mps[k], c='orange')
                axs[k + mps.shape[0]].set_xlabel('Time', fontsize ='20')

                axs[k].axvline(x=start_date_index[1], linestyle="dashed", c='black')
                axs[k].axvline(x=nn_idx[1], linestyle="dashed", c='black')
                axs[k + mps.shape[0]].axvline(x=start_date_index[1], linestyle="dashed", c='black')
                axs[k + mps.shape[0]].axvline(x=nn_idx[1], linestyle="dashed", c='black')

                axs[k].plot(range(start_date_index[k], start_date_index[k] + self.m), df[dim_name].iloc[start_date_index[k] : start_date_index[k] + self.m], c='red', linewidth=4)
                axs[k].plot(range(nn_idx[k], nn_idx[k] + self.m), df[dim_name].iloc[nn_idx[k] : nn_idx[k] + self.m], c='red', linewidth=4)
                axs[k + mps.shape[0]].plot(start_date_index[k], mps[k, start_date_index[k]] + 1, marker="v", markersize=10, color='red')
                axs[k + mps.shape[0]].plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", markersize=10, color='red')

            plt.tight_layout()
            plt.show()
        return self.dataframe.iloc[nn_idx[0]].ticker, result 



def motif_pre_filter()->list[str]:
    # https://www.youtube.com/watch?v=4YPyxfXML0A&t=1825s
    motif_params = {
        "exchangeName": "HOSE",
        "marketCap": (1_000, 200_000),
        # "avgTradingValue20Day": (100, 20000),  # Minimum 20-day average trading value
        # 'revenueGrowth5Year': (15,100),
        # 'pe':(10,20),
        # "epsGrowth5Year": (10, 100),  # Minimum 1-year EPS growth
        # "roe": (15, 100),  # Minimum Return on Equity (ROE)
        'priceNearRealtime': (15,80),
        "lastQuarterProfitGrowth": (2, 100),  # Minimum last quarter profit growth
        }
    df = filter_stocks(motif_params)
    return df['ticker'].to_list()


def find_best_motifs(start_date = None, end_date = None,
                    dimension_cols:list=['close', 'volume'],
                    nn_idx_threshold:int=5, 
                    distance_threshold:float = 5, 
                    plot:bool = False):
    unique_tickers = motif_pre_filter()
    results_dict = {}  # Initialize an empty dictionary to store results

    for ticker in unique_tickers:
        motif = MotifMatching(symbol = ticker, start_date=start_date, end_date=end_date)
        stock_name, result = motif.find_matching_series_multi_dim_with_date(
                                                                            dimension_cols=dimension_cols,
                                                                            nn_idx_threshold=nn_idx_threshold, 
                                                                            distance_threshold = distance_threshold, 
                                                                            plot = plot)
        
        # Check if the stock_name is not None
        if stock_name is not None:
            # Add the result to the dictionary with stock_name as the key
            pattern_start_date = motif.dataframe.iloc[int(result[0])].time.strftime('%Y-%m-%d')
            pattern_end_date= motif.dataframe.iloc[int(result[0]) + motif.m].time.strftime('%Y-%m-%d')
            distance = result[2]
            results_dict[stock_name] = [pattern_start_date, pattern_end_date, distance]

    return results_dict
