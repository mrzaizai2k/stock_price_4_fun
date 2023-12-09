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

from src.Utils.utils import *
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
        return k_index

    def plot_and_save_top_pattern(self, 
                        col="close",
                        figsize=(12, 6),
                        save_fig:bool = False,):
        
        k_index = self.find_top_pattern(col="close",top_k=3,)

        Q_df = self.filtered_data[col]
        T_df = self.dataframe[col]

        plt.figure(figsize=figsize)
        plt.plot(T_df.values)
        for idx in k_index:
            plt.plot(
                range(idx, idx + len(Q_df)),
                T_df.values[idx : idx + len(Q_df)],
                lw=2,
                label=f"Series at index {idx}",
            )

        plt.xlabel("Index")
        plt.ylabel(col)
        plt.legend()
        plt.tight_layout()
        if save_fig:
            check_path('data')
            save_path = f'data/{self.symbol}_top_pattern.png'
            plt.savefig(save_path)
            print(f'Plot saved in {save_path}')
            return save_path
        else:
            plt.show()
            return None

    def find_matching_series_multi_dim_with_date(self,
                                              dimension_cols:list=['close', 'volume'],
                                             nn_idx_threshold:int=None, 
                                             distance_threshold:float = None,
                                             ):
        

        start_date_index = len(self.dataframe) - self.m
        filter_df = self.dataframe[dimension_cols].reset_index(drop = True)
        mps, indices = stumpy.mstump(filter_df, self.m)
        start_date_index = np.repeat(start_date_index, len(filter_df.columns))
        nn_idx = indices[np.arange(len(filter_df.columns)), start_date_index]
        # Check if the absolute difference between element 1 and element 2 is less than threshold
        if nn_idx_threshold is not None and np.abs(nn_idx[1] - nn_idx[0]) > nn_idx_threshold:
            return None, None, None
        
        distance = []
        for k in range(len(filter_df.columns)):
            distance.append(mps[k, nn_idx[k]])
        mean_distance = np.mean(distance)

        if distance_threshold is not None and mean_distance > distance_threshold:
            return None, None, None
        
        result  = np.concatenate((nn_idx, np.array([mean_distance])))

        print(f'm: {self.m}')
        print(f'From {self.dataframe.iloc[nn_idx[0]].time} to {self.dataframe.iloc[nn_idx[0] + self.m].time}')
        print(f'nn_idx: {nn_idx}')

        return result, mps, start_date_index

    def plot_and_save_find_matching_series_multi_dim_with_date(self,
                                              dimension_cols:list=['close', 'volume'],
                                             nn_idx_threshold:int=None, 
                                             distance_threshold:float = None,
                                             save_fig:bool=False):
        
        filter_df = self.dataframe[dimension_cols].reset_index(drop = True)
        result, mps, start_date_index = self.find_matching_series_multi_dim_with_date(dimension_cols=dimension_cols,
                                                                    nn_idx_threshold=nn_idx_threshold,
                                                                    distance_threshold=distance_threshold,)

        if result is None:
            return None
        
        nn_idx = result[:-1].astype(int)
        
        fig, axs = plt.subplots(mps.shape[0] * 2, sharex=True, gridspec_kw={'hspace': 0}, figsize=(14, 3 * mps.shape[0] * 2))
        fig.suptitle(f"Multi-dimension pattern for {self.symbol}", fontsize=20)
        fig.text(0.5, 0.92, f"Window Size = {self.m} | Distance = {result[-1]:.2f}", ha='center', fontsize=17)

        for k, dim_name in enumerate(filter_df.columns):
            axs[k].set_ylabel(dim_name, fontsize='20')
            axs[k].plot(filter_df[dim_name])
            axs[k].set_xlabel('Time', fontsize ='20')

            axs[k + mps.shape[0]].set_ylabel(dim_name.replace('T', 'P'), fontsize='20')
            axs[k + mps.shape[0]].plot(mps[k], c='orange')
            axs[k + mps.shape[0]].set_xlabel('Time', fontsize ='20')

            axs[k].axvline(x=start_date_index[1], linestyle="dashed", c='black')
            axs[k].axvline(x=nn_idx[1], linestyle="dashed", c='black')
            axs[k + mps.shape[0]].axvline(x=start_date_index[1], linestyle="dashed", c='black')
            axs[k + mps.shape[0]].axvline(x=nn_idx[1], linestyle="dashed", c='black')

            axs[k].plot(range(start_date_index[k], start_date_index[k] + self.m), filter_df[dim_name].iloc[start_date_index[k] : start_date_index[k] + self.m], c='red', linewidth=4)
            axs[k].plot(range(nn_idx[k], nn_idx[k] + self.m), filter_df[dim_name].iloc[nn_idx[k] : nn_idx[k] + self.m], c='red', linewidth=4)
            axs[k + mps.shape[0]].plot(start_date_index[k], mps[k, start_date_index[k]] + 1, marker="v", markersize=10, color='red')
            axs[k + mps.shape[0]].plot(nn_idx[k], mps[k, nn_idx[k]] + 1, marker="v", markersize=10, color='red')

        plt.tight_layout()
        if save_fig:
            check_path('data')
            save_path = f'data/{self.symbol}_multi_dimension_pattern.png'
            plt.savefig(save_path)
            print(f'Plot saved in {save_path}')
            return save_path
        else:
            plt.show()
            return None


class BestMarketMotifSearch:
    '''Search best motif on all stocks on the market'''

    def __init__(self, start_date:str=None, 
                 end_date:str=None, 
                 dimension_cols:list=['close', 'volume'], 
                 nn_idx_threshold:int=5, 
                 distance_threshold:float=5, 
                 motif_data_path:str='memory/motif.csv'):
        
        self.end_date = end_date
        if self.end_date is None:
            self.end_date = datetime.now().strftime("%Y-%m-%d")

        self.start_date = start_date
        if self.start_date is None:
            self.start_date = (
                datetime.strptime(self.end_date, "%Y-%m-%d") - timedelta(days=30)
            ).strftime("%Y-%m-%d")

        self.dimension_cols = dimension_cols
        self.nn_idx_threshold = nn_idx_threshold
        self.distance_threshold = distance_threshold
        self.motif_data_path = motif_data_path
        self.date_format ='%Y-%m-%d'
        self.current_date = datetime.today().strftime(self.date_format)

    def motif_pre_filter(self) -> list[str]:
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

    def calculate_best_motif_from_scratch(self):
        '''Prefilter the stock on the market that match teh standard then find best motif among them'''
        unique_tickers = self.motif_pre_filter()
        results_dict = {}

        for ticker in unique_tickers:
            motif = MotifMatching(symbol=ticker, start_date=self.start_date, end_date=self.end_date)

            result, mps, start_date_index = motif.find_matching_series_multi_dim_with_date(
                                                dimension_cols=self.dimension_cols,
                                                nn_idx_threshold=self.nn_idx_threshold,
                                                distance_threshold=self.distance_threshold,
                                            )

            if result is not None:
                pattern_start_date = motif.dataframe.iloc[int(result[0])].time.strftime(self.date_format)
                pattern_end_date = motif.dataframe.iloc[int(result[0]) + motif.m].time.strftime(self.date_format)
                distance = result[2]
                results_dict[ticker] = [pattern_start_date, pattern_end_date, distance]

        return results_dict

    def load_motif_csv(self):
        '''Check if there is database'''
        check_path(self.motif_data_path)
        if os.path.exists(self.motif_data_path):
            return pd.read_csv(self.motif_data_path)
        return None

    def save_motif_csv(self, results_dict):
        '''Save result from calculate_best_motif_from_scratch to database'''
        df = pd.DataFrame(results_dict.items(), columns=['stock_name', 'info'])
        df[['pattern_start_date', 'pattern_end_date', 'distance']] = pd.DataFrame(df['info'].tolist(), index=df.index)
        df.drop(columns=['info'], inplace=True)
        df['backup_date'] = self.current_date
        df.to_csv(self.motif_data_path, index=False)

    def find_best_motifs(self):
        '''
        1. Search best motif on database
        2. If there is no database, create one
        3. If the data is not today data. Recalculate best motif
        '''
        motif_df = self.load_motif_csv()

        if motif_df is not None and self.current_date in motif_df["backup_date"].values:
            today_data = motif_df[motif_df['backup_date'] == self.current_date] 
            results_dict = dict(zip(today_data['stock_name'], 
                                    today_data[['pattern_start_date', 'pattern_end_date', 'distance']].values.tolist()))
            return results_dict
        else:
            results_dict = self.calculate_best_motif_from_scratch()
            self.save_motif_csv(results_dict)
            return results_dict

    def plot_best_motifs(self):
        results_dict = self.find_best_motifs()
        for ticker, values in results_dict.items():
            motif = MotifMatching(symbol=ticker, start_date=self.start_date, end_date=self.end_date)
            _ = motif.plot_and_save_find_matching_series_multi_dim_with_date(
                                                dimension_cols=self.dimension_cols,
                                                nn_idx_threshold=self.nn_idx_threshold,
                                                distance_threshold=self.distance_threshold,
                                            )
        return

def main():
    motif_matching = MotifMatching('VNINDEX', start_date="2023-09-11")
    # motif_matching.find_top_pattern()
    market_motif_search = BestMarketMotifSearch(motif_data_path='memory/motif.csv')
    results_dict = market_motif_search.find_best_motifs()
    print(json.dumps(results_dict, indent=4))

if __name__ =="__main__":
    main()