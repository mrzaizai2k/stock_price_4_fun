import sys
sys.path.append("")
import os
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths


class WinLossAnalyzer:
    def __init__(self, win_loss_df_path:str = 'data/BaoCaoLaiLo_058C647873.csv',
                  start_date = None, end_date = None, time_col:str = 'time'):
        """
        Initialize the StrategyAnalyzer with trading data.

        Parameters:
        - df (pd.DataFrame): DataFrame containing trading data.
        """
        self.df_path = win_loss_df_path
        self.df = self.load_data()
        self.time_col = time_col

        self.end_date = end_date
        if self.end_date is None:
            self.end_date = self.df[self.time_col].max()

        self.start_date = start_date
        if self.start_date is None:
            self.start_date = self.df[self.time_col].min()
        
        self.df = self.filter_data()


    def load_data(self):
        df = pd.read_csv(self.df_path, encoding='latin1', skiprows=5)
        column_order = ['time', 'sell_vol', 'sell_price', 'sell_value', 'capital_price', 'capital_value', 'dividend', 'win_loss_value', 'win_loss_percent']
        df.columns = column_order
        df = df.drop(df.index[-1])
        df = self.preprocess_data(df) 
        return df

    def preprocess_data(self, df):
        df['win_loss_percent'] = df['win_loss_percent'].str.rstrip('%')
        df['ticker'] = df['time'].where(df['time'].str.len() == 3)
        df['ticker'] = df['ticker'].ffill().astype('category')
        df = df[df['time'].str.len() != 3]
        df['time'] = pd.to_datetime(df['time'], format='%d/%m/%Y')
        df = df.drop('dividend', axis=1)
        num_cols = ['sell_vol', 'sell_price', 'sell_value', 'capital_price',
                'capital_value', 'win_loss_value', 'win_loss_percent']
        df[num_cols] = df[num_cols].replace(',', '', regex=True).astype(float)

        return df

        

    def filter_data(self):
        filtered_data = self.df[
            (self.df[self.time_col] >= self.start_date) & (self.df[self.time_col] <= self.end_date)
        ]
        return filtered_data
    

    def calculate_win_ratio(self):
        total_trades_amount = self.df['sell_value'].sum()
        winning_trades = self.df[self.df['win_loss_value'] > 0]
        winning_trades_amount = winning_trades['sell_value'].sum()
        win_ratio = winning_trades_amount / total_trades_amount
        return win_ratio*100

    def calculate_payoff_ratio(self):
        average_win = self.df[self.df['win_loss_value'] > 0]['win_loss_value'].mean()
        average_loss = self.df[self.df['win_loss_value'] < 0]['win_loss_value'].mean()
        payoff_ratio = abs(average_win / average_loss)
        return payoff_ratio

    def calculate_largest_winning_trade(self):
        return self.df['win_loss_value'].max()

    def calculate_largest_losing_trade(self):
        return self.df['win_loss_value'].min()

    def calculate_largest_winning_trade_percent(self):
        return 100*self.df['win_loss_percent'].max()

    def calculate_largest_losing_trade_percent(self):
        return 100*self.df['win_loss_percent'].min()

    def calculate_average_winning_trade(self):
        return self.df[self.df['win_loss_value'] > 0]['win_loss_value'].mean()

    def calculate_average_losing_trade(self):
        return self.df[self.df['win_loss_value'] < 0]['win_loss_value'].mean()

    def calculate_largest_drawdown(self):
        cumulative_returns = (1 + self.df['win_loss_percent'] / 100).cumprod()
        largest_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        return 100*largest_drawdown

    def calculate_average_drawdown(self):
        cumulative_returns = (1 + self.df['win_loss_percent'] / 100).cumprod()
        average_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).mean()
        return 100*average_drawdown

    def calculate_total_days(self):
        return (self.df[self.time_col].max() - self.df[self.time_col].min()).days

    def calculate_total_sell_value(self):
        return self.df['sell_value'].sum()

    def calculate_total_capital(self):
        return self.df['capital_value'].sum()

    def calculate_delta_sell_capital(self):
        total_sell_value = self.calculate_total_sell_value()
        total_capital = self.calculate_total_capital()
        return total_sell_value - total_capital

    def calculate_delta_sell_capital_percent(self):
        delta_sell_capital = self.calculate_delta_sell_capital()
        total_capital = self.calculate_total_capital()
        return (delta_sell_capital / total_capital) * 100

    def calculate_average_trading_frequency(self):
        self.df['year_month'] = self.df[self.time_col].dt.to_period('M')
        average_trading_frequency = self.df.groupby('year_month').size().mean()
        self.df.drop(['year_month'], axis=1, inplace=True) 
        return average_trading_frequency

    def calculate_stock_fee(self):
        stock_fee = ((0.088 + 0.1) / 100) * self.df['capital_value'].sum()
        return stock_fee

    def plot_return(self, col='win_loss_value', figsize=(800, 600)):

        daily_aggregated = self.df.groupby(self.time_col).agg({col: 'sum'}).reset_index()

        # Step 3: Plot the Sell Value and Capital Value using Plotly
        fig = px.line(daily_aggregated, x=self.time_col, y=col,  # Use col directly here
                      title=f'Daily Aggregated {col}',
                      labels={'value': 'Value', self.time_col: 'Date'},  # Corrected the date label
                      line_shape='linear', render_mode='svg')

        # Step 4: Customize the layout if needed
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Value',
            showlegend=True,
        )

        # Step 5: Set the figsize
        fig.update_layout(width=figsize[0], height=figsize[1])

        # Step 6: Show the plot
        fig.show()

    def top_best_stock(self, top_k: int = 3):
        profitable_stocks = self.df[self.df['win_loss_value'] > 0].groupby('ticker')['win_loss_value'].sum()
        top_profitable_stocks = profitable_stocks.nlargest(top_k).reset_index()
        return list(zip(top_profitable_stocks['ticker'], top_profitable_stocks['win_loss_value']))


    def top_worst_stock(self, top_k: int = 3):
        losing_stocks = self.df[self.df['win_loss_value'] < 0].groupby('ticker')['win_loss_value'].sum()
        top_losing_stocks = losing_stocks.nsmallest(top_k).reset_index()
        return list(zip(top_losing_stocks['ticker'], top_losing_stocks['win_loss_value']))


    def analyze_strategy(self) ->dict:
        metrics = {
            'Win Ratio': self.calculate_win_ratio(),
            'Payoff Ratio': self.calculate_payoff_ratio(),
            'Largest Winning Trade': self.calculate_largest_winning_trade(),
            'Largest Losing Trade': self.calculate_largest_losing_trade(),
            'Largest Winning Trade percent': self.calculate_largest_winning_trade_percent(),
            'Largest Losing Trade percent': self.calculate_largest_losing_trade_percent(),
            'Average Winning Trade': self.calculate_average_winning_trade(),
            'Average Losing Trade': self.calculate_average_losing_trade(),
            'Largest % Drawdown': self.calculate_largest_drawdown(),  # Convert to percentage
            'Average % Drawdown': self.calculate_average_drawdown(),  # Convert to percentage
            'Total Days': self.calculate_total_days(),
            'total_sell_value': self.calculate_total_sell_value(),
            'total_capital': self.calculate_total_capital(),
            'delta_sell_capital': self.calculate_delta_sell_capital(),
            'delta_sell_capital_percent': self.calculate_delta_sell_capital_percent(),
            'Average Trading Frequency per Month': self.calculate_average_trading_frequency(),
            'stock_fee': self.calculate_stock_fee(),
            'Top Best Stock': self.top_best_stock(1),  # Adjust the parameter based on your requirements
            'Top Worst Stock': self.top_worst_stock(1),
        }

        return metrics

    def get_report(self) -> None:
        metrics = self.analyze_strategy()

        print("Strategy Analysis Report:")
        print("-" * 50)
        
        for key, value in metrics.items():
            formatted_value = f"{value:.2f}" if isinstance(value, float) else value
            print(f"{key}: {formatted_value}")
