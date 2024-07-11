import sys
sys.path.append("")
import pandas as pd
import os
from vnstock import *
from dotenv import load_dotenv
load_dotenv()

import cv2
import subprocess
import schedule
import time
import yaml
from functools import wraps
import torch
from pydub import AudioSegment

import re
import phunspell
from rank_bm25 import BM25Okapi
import editdistance
from typing import Literal

from src.Microsofttodo import *

from src.Utils.logger import create_logger
logger = create_logger()


class SpellCheck:
    def __init__(self, history_tasks_path: str, n_grams: int):
        self.history_tasks_path = history_tasks_path
        self.n_grams = n_grams
        self.history_tasks = self.load_history_tasks(history_tasks_path)

    def preprocess_text(self, input_string):
        return re.sub(r'[^\w\s\']', '', input_string)

    def load_history_tasks(self, file_path:str):
        with open(file_path, 'r') as file:
            tasks = file.readlines()
            tasks = [task.strip() for task in tasks]
        return tasks

    def generate_ngrams(self, text:str, n_grams:int=1):
        words = text.split()
        ngrams = []
        for i in range(1, n_grams + 1):
            ngrams.extend([' '.join(words[j:j + i]) for j in range(len(words) - i + 1)])
        return ngrams

    def get_best_match_bm25(self, token, history_tokens_flat, bm25, 
                            verbose:bool = False):
        token_candidates = bm25.get_top_n(token, history_tokens_flat, n=5)
        if verbose:
            print('wrong token', token)
            print('token candidates', token_candidates[:5])
        return token_candidates[0] if token_candidates else token

    def get_best_match_editdistance(self, token, history_tokens_flat, 
                                    verbose:bool = False):
        distances = [(history_token, editdistance.eval(token, history_token)) for history_token in history_tokens_flat]
        distances.sort(key=lambda x: x[1])
        if verbose:
            print('wrong token', token)
            print('token candidates', distances[:5])  # Print top 5 candidates based on edit distance
        return distances[0][0] if distances else token

    def get_best_match_bm25_editdistance(self, token, history_tokens_flat, bm25, 
                                         verbose:bool = False, top_n:int = 10):
        token_candidates = bm25.get_top_n(token, history_tokens_flat, top_n)
        if verbose:
            print('wrong token', token)
            print('BM25 top 10 candidates', token_candidates)

        if not token_candidates:
            return token

        distances = [(candidate, editdistance.eval(token, candidate)) for candidate in token_candidates]
        distances.sort(key=lambda x: x[1])
        if verbose:
            print('Edit Distance candidates', distances[:5])  # Print top 5 candidates based on edit distance

        return distances[0][0] if distances else token

    def spell_check_and_correct(self, input_string, 
                                method: Literal["BM25", "editdistance", "BM25_EditDistance"], 
                                loc_lang: Literal['en_US', 'vi_VN'] = 'en_US', 
                                verbose: bool = False):
        
        input_tokens = self.preprocess_text(input_string.lower())
        corrected_tokens = []

        input_ngrams = self.generate_ngrams(input_tokens, self.n_grams)

        history_tasks_tokens = [self.generate_ngrams(self.preprocess_text(task.lower()), self.n_grams) for task in self.history_tasks]
        history_tokens_flat = [token for sublist in history_tasks_tokens for token in sublist]

        pspell = phunspell.Phunspell(loc_lang=loc_lang)

        if method == "BM25" or method == "BM25_EditDistance":
            bm25 = BM25Okapi(history_tokens_flat)

        for token in input_ngrams:
            if token in pspell.lookup_list(token.split(" ")):
                if method == "BM25":
                    best_match = self.get_best_match_bm25(token, history_tokens_flat, bm25, verbose)
                elif method == "editdistance":
                    best_match = self.get_best_match_editdistance(token, history_tokens_flat, verbose)
                elif method == "BM25_EditDistance":
                    best_match = self.get_best_match_bm25_editdistance(token, history_tokens_flat, bm25, verbose)
                corrected_tokens.append(best_match)
            else:
                corrected_tokens.append(token)

        corrected_string = input_string
        for original_token, corrected_token in zip(input_tokens.split(), corrected_tokens):
            if original_token != corrected_token:
                corrected_string = re.sub(r'\b{}\b'.format(re.escape(original_token)), corrected_token, corrected_string, count=1, flags=re.IGNORECASE)

        corrected_string = corrected_string.capitalize()
        return corrected_string
    

def convert_m4a_to_mp3(m4a_file_path:str, mp3_file_path:str):
    # Load the .m4a file
    audio = AudioSegment.from_file(m4a_file_path, format="m4a")
    
    # Export as .mp3 file
    audio.export(mp3_file_path, format="mp3")
    print(f"Conversion complete: {m4a_file_path} to {mp3_file_path}")


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.2f} seconds to execute.")
        return result

    return wrapper

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
    for task in tasks_list:
        todo.create_task(task_name=task["title"], list_name='Tasks', 
                         importance=task["important"], dueDateTime=task["dueDateTime"])
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
    capture_image_from_camera()
    # check_path("data/data1")
    # check_path("data/data2/note.txt")
    # user_db = UserDatabase()
    # data_config_path = 'config/config.yaml'
    # with open(data_config_path, 'r') as file:
    #     data = yaml.safe_load(file)

    # watchlist = data.get('my_watchlist', [])    
    # USER_ID = os.getenv('USER_ID')
    # user_db.save_watch_list(user_id=USER_ID, watch_list=watchlist)
    # watch_list = user_db.get_watch_list(user_id=USER_ID)


if __name__ == "__main__":
    main()
