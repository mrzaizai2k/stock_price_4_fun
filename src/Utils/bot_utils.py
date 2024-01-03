import sys
sys.path.append("")
import os 
import yaml
import schedule
import threading
from threading import Thread
from time import sleep
import subprocess

import telebot
from telebot import types
from dotenv import load_dotenv
load_dotenv()

# from telegram.ext.filters import Filters 
from src.Utils.utils import *
from src.PayBackTime import PayBackTime, find_PBT_stocks
from src.motif import MotifMatching, BestMarketMotifSearch
from src.Indicators import MACD, BigDayWarning, PricevsMA
from src.support_resist import SupportResistFinding
from src.trading_record import BuySellAnalyzer, WinLossAnalyzer, TradeScraper
from functools import wraps


# data = config_parser(data_config_path = 'config/config.yaml')

def toggle_button_state(button_text, data_config_path:str = 'config/config.yaml'):
    data = config_parser(data_config_path = data_config_path)
    special_symbols = data.get('special_symbols')
    CHECK_CHAR = special_symbols['CHECK_CHAR']
    UNCHECK_CHAR = special_symbols['UNCHECK_CHAR']
    return f'{UNCHECK_CHAR} {button_text.split(" ", 1)[-1]}' if button_text.startswith(CHECK_CHAR) else f'{CHECK_CHAR} {button_text.split(" ", 1)[-1]}'


def handle_checklist(text, data_config_path:str = 'config/config.yaml'):

    data = config_parser(data_config_path = data_config_path)
    special_symbols = data.get('special_symbols')
    CHECK_CHAR = special_symbols['CHECK_CHAR']
    UNCHECK_CHAR = special_symbols['UNCHECK_CHAR']
    
    lines = text.split('\n')
    keyboard = []
    index = 0
    previous_state = {}
    for line in lines:
        line_strip = line.strip()
        if line_strip == '':
            continue

        keyboard.append([types.InlineKeyboardButton(
            f"{CHECK_CHAR if previous_state.get(line_strip, False) else UNCHECK_CHAR} {line_strip}",
            callback_data=f"checklist_toggle__{index}",
        )])
        index += 1

    reply_markup = types.InlineKeyboardMarkup(keyboard)
    return reply_markup

def validate_symbol_decorator(bot):
    def decorator(func):
        @wraps(func)
        def wrapper(message, command=None):
            symbol = message.text.upper()
            if not validate_symbol(symbol):
                bot.send_message(message.chat.id, f'Sorry! There is no stock {symbol}')
                return
            return func(message, command) if command else func(message)
        
        return wrapper
    
    return decorator



def run_vscode_tunnel(bot, message):
    command = ['./code', 'tunnel']
    VSCODE_TUNNEL_LINK = os.getenv('VSCODE_TUNNEL_LINK')
    bot.reply_to(message, f"VS Code remote tunnel opened!: {VSCODE_TUNNEL_LINK}")
    result = subprocess.run(command, check=True, text=True)

def warning_macd(bot, watchlist, user_id):  # Pass the message parameter

    warning_report = []  # Initialize an empty list to store warning reports
    for symbol in watchlist:
        macd = MACD(symbol)
        if macd.is_cross_up(offset=3):
            warning_report.append(f'{symbol}: Crossed up 🔼')
        elif macd.is_cross_down(offset=3):
            warning_report.append(f'{symbol}: Crossed down 🔻')
    if warning_report:
        # If there are warnings, send a report
        report_message = '\n'.join(warning_report)
        return bot.send_message(user_id, f'Report for stocks with warnings:\n{report_message}')
    else:
        print("There's no warning")


def warningbigday(bot, watchlist, user_id):  # Pass the message parameter

    warning_report = []  # Initialize an empty list to store warning reports

    for symbol in watchlist:
        bigday = BigDayWarning(symbol, percent_diff=3)
        if bigday.is_big_increase():
            warning_report.append(f'Powerful UP for {symbol} 💹')
        if bigday.is_big_decrease():
            warning_report.append(f'Powerful DOWN for {symbol} 🆘')

    if warning_report:
        # If there are FTD warnings, send a report
        report_message = '\n'.join(warning_report)
        return bot.send_message(user_id, f'Report for stocks with big day warnings:\n{report_message}')
    else:
        print("There's no warning")

def warningpricevsma(bot,watchlist, user_id, data_config):  # Pass the message parameter
    warning_report = []  # Initialize an empty list to store warning reports

    for symbol in watchlist:
        pvma = PricevsMA(symbol)
        pvma_offset = data_config.get('pvma_offset')

        if pvma.is_cross_up(offset=pvma_offset):
            warning_report.append(f'{symbol}: Crossed up')
        elif pvma.is_cross_down(offset=pvma_offset):
            warning_report.append(f'{symbol}: Crossed down')

    if warning_report:
        # If there are warnings, send a report
        report_message = '\n'.join(warning_report)
        return bot.send_message(user_id, f'Report for stocks with PricevsMA warnings:\n{report_message}')
    else:
        print("There's no warning")

def warningsnr(bot, watchlist, user_id, data_config):  # Pass the message parameter
    warning_report = []  # Initialize an empty list to store warning reports
    tolerance_percent = data_config.get('tolerance_percent') 
    for symbol in watchlist:
        sr_finding = SupportResistFinding(symbol=symbol)
        current_price = sr_finding.get_current_price()

        # Find the closest support and resistance levels
        support, resistance = sr_finding.find_closest_support_resist(current_price=current_price)

        # Set the tolerance percentage
        tolerance_percent = tolerance_percent / 100  # 1 percent

        # Check if the current price is within the tolerance range of support or resistance
        if support <= current_price <= support * (1 + tolerance_percent):
            warning_report.append(f'{symbol}: Meeting Support')
        elif resistance is not None and (resistance * (1 - tolerance_percent) <= current_price <= resistance):
            warning_report.append(f'{symbol}: Meeting Resistance')


    if warning_report:
        # If there are warnings, send a report
        report_message = '\n'.join(warning_report)
        return bot.send_message(user_id, f'Support/Resistance warning for stocks:\n{report_message}')
    else:
        print("There's no warning")