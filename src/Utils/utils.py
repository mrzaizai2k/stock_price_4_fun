import sys
sys.path.append("")
import pandas as pd
import os
from vnstock import *
from dotenv import load_dotenv
load_dotenv()

import subprocess
import schedule
import time
import yaml
from functools import wraps
import torch
from src.Microsofttodo import *

def create_env_file():
    env_file_path = '.env'
    if not os.path.isfile(env_file_path):
        with open(env_file_path, 'w') as env_file:
            env_file.write('TELEBOT_API=\n')
            env_file.write('TRADE_USER=\n')
            env_file.write('TRADE_PASS=\n')
            env_file.write('YOUR_TELEGRAM_ID=\n')


def sync_task_to_todo(tasks_list:list[dict]):
    todo = MicrosoftToDo()
    # Split the text by '\n' and create a list
    for tmp in tasks_list:
        todo.create_task(task_name=tmp["title"], list_name='Tasks', 
                         importance=tmp["important"], dueDateTime=tmp["dueDateTime"])
    return

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
    return (symbol in listing_companies(live=False).ticker.tolist()) or (symbol in ['VNINDEX','VN30'])

def schedule_checker():
    while True:
        schedule.run_pending()
        time.sleep(1)

def take_device():
    # Check for GPU availability
    gpu_available = torch.cuda.is_available()

    # Set the device based on availability
    device = torch.device("cuda" if gpu_available else "cpu")

    # Print the selected device
    print(f"Selected device: {device}")

    return device

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


def is_file(path: str):
    return '.' in path


def check_path(path):
    # Extract the last element from the path
    last_element = os.path.basename(path)
    if is_file(last_element):
        # If it's a file, get the directory part of the path
        folder_path = os.path.dirname(path)

        # Check if the directory exists, create it if not
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Create new folder path: {folder_path}")
    else:
        # If it's not a file, it's a directory path
        # Check if the directory exists, create it if not
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Create new path: {path}")

def config_parser(data_config_path = 'config/config.yaml'):
    with open(data_config_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def validate_mrzaizai2k_user(user_id):
    MRZAIZAI2K_ID = os.getenv('MRZAIZAI2K_ID')
    if str(user_id) != MRZAIZAI2K_ID:
        return False
    return True

def read_commands_from_file(filename):
    with open(filename, 'r') as file:
        commands_string = file.read()
    return commands_string

class UserDatabase:
    def __init__(self, user_data_path:str='data/user_db.csv'):
        self.user_data_path = user_data_path
        self.load_user_database()
        self.user_df.reset_index(inplace=True)
        

    def is_user_in_database(self, user_id):
        return str(user_id) in self.user_df['user_ID'].astype(str).values

    def save_user_to_database(self, user_id):
        new_data = pd.DataFrame({'user_ID': [str(user_id)]})
        self.user_df = pd.concat([self.user_df, new_data], ignore_index=True)
        self.user_df.to_csv(self.user_data_path, index=False)

    def convert_data_type(self):
        self.user_df = self.user_df[['user_ID', 'watch_list']].astype(str)

    def load_user_database(self):
        try:
            self.user_df = pd.read_csv(self.user_data_path)
            self.convert_data_type()
        except FileNotFoundError:
            self.create_empty_database()

    def create_empty_database(self):
        check_path(self.user_data_path)
        self.user_df = pd.DataFrame(columns=['user_ID', 'watch_list', 'user_step'])
        self.user_df.to_csv(self.user_data_path, index=False)

    def save_watch_list(self, user_id, watch_list:list):
        user_id = str(user_id)
        if not self.is_user_in_database(user_id):
            self.save_user_to_database(user_id)

        watch_list_str = ", ".join(watch_list)
        user_row = self.user_df[self.user_df['user_ID'] == user_id]
        self.user_df.loc[user_row.index, 'watch_list'] = watch_list_str
        self.user_df.to_csv(self.user_data_path, index=False)

    def get_watch_list(self, user_id) -> list:
        user_id = str(user_id)
        if self.is_user_in_database(user_id):
            watch_list_str = self.user_df.loc[self.user_df['user_ID'] == user_id, 'watch_list'].values[0]
            if watch_list_str == 'nan':
                return []
            return watch_list_str.split(', ') if watch_list_str else []
        else:
            self.save_user_to_database(user_id)
            return []
    def get_users_for_warning(self) -> list:
        user_list = self.user_df[self.user_df['watch_list'].notna()]['user_ID'].tolist()
        return user_list
    
    def get_all_watchlist(self)-> list:
        '''Get all stocks of all users'''
        
        all_watchlists = []

        # Iterate through each user and append their watchlist to the combined list
        for index, row in self.user_df.iterrows():
            watch_list_str = row['watch_list']
            if watch_list_str and watch_list_str != 'nan':
                all_watchlists.extend(watch_list_str.split(', '))

        # Remove duplicates by converting to a set and then back to a list
        all_watchlists = list(set(all_watchlists))

        return all_watchlists

def main():
    print('Hi')
    # check_path("data/data1")
    # check_path("data/data2/note.txt")
    user_db = UserDatabase()
    data_config_path = 'config/config.yaml'
    with open(data_config_path, 'r') as file:
        data = yaml.safe_load(file)

    watchlist = data.get('my_watchlist', [])    
    USER_ID = os.getenv('USER_ID')
    user_db.save_watch_list(user_id=USER_ID, watch_list=watchlist)
    watch_list = user_db.get_watch_list(user_id=USER_ID)


if __name__ == "__main__":
    main()
