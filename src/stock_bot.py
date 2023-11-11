import sys
sys.path.append("")
import os 
import telebot
from telebot import types
from dotenv import load_dotenv
load_dotenv()

# from telegram.ext.filters import Filters 
from src.PayBackTime import PayBackTime, find_PBT_stocks
from src.utils import *
from src.motif import MotifMatching, find_best_motifs


TELEBOT_API= os.getenv('TELEBOT_API')
# print('key', TELEBOT_API)
bot = telebot.TeleBot(TELEBOT_API)

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Welcome to the Mrzaizai2k Stock Assistant bot! Type /help to see the available commands.")

@bot.message_handler(commands=['help'])
def help(message):
    bot.send_message(message.chat.id, "Available commands:\n/help - Show this help message")
    bot.send_message(message.chat.id, "\n/pbt + symbol: Calculate the payback time for a stock")
    bot.send_message(message.chat.id, "\n/findpbt: find payback time stocks right now")
    bot.send_message(message.chat.id, "\n/findmyfav: find my_param stocks right now")
    bot.send_message(message.chat.id, "\n/risk + symbol: calculate how much stocks u can buy with loss/trade = 6% with max loss/capital = 2%")
    bot.send_message(message.chat.id, "\n/rate + symbol: general rating the stock")
    bot.send_message(message.chat.id, "\n/mulpattern + symbol + date (YYYY-mm-dd): find pattern of the stock on multi-dimension ['close', 'volume']")
    bot.send_message(message.chat.id, "\n/pattern + symbol + date (YYYY-mm-dd): find pattern of the stock ['close']")
    bot.send_message(message.chat.id, "\n/findbestmotif: Find the best motif on all the stocks")

@bot.message_handler(commands=['rate', 'risk', 'pbt','mulpattern', 'pattern'])
def ask_for_symbol(message):
    # Ask for the stock symbol
    markup = types.ForceReply(selective = False)
    bot.reply_to(message, "Please enter the stock symbol:", reply_markup = markup)
    if message.text == '/rate':
        bot.register_next_step_handler(message, rate)
    elif message.text == '/risk':
        bot.register_next_step_handler(message, calculate_risk)
    elif message.text == '/pbt':
        bot.register_next_step_handler(message, get_paybacktime)
    else: # message.text in ['/mulpattern', '/pattern']:
        bot.register_next_step_handler(message, ask_pattern_stock, message.text)


def ask_pattern_stock(message, command):
    symbol = message.text.upper()
    if not validate_symbol(symbol):
        bot.send_message(message.chat.id, f'Sorry! There is no stock {symbol}')
        return 
    markup = types.ForceReply(selective = False)
    bot.reply_to(message, "Please enter the start_date (YYYY-mm-dd):", reply_markup = markup)

    if command == '/mulpattern':
        bot.register_next_step_handler(message, find_similar_pattern_multi_dimension, symbol)
    else: # command == '/pattern':
        bot.register_next_step_handler(message, find_similar_pattern, symbol)


def find_similar_pattern(message, symbol):

    start_date = message.text # %Y-%m-%d
    bot.send_message(message.chat.id, "Please wait. This process can takes several minutes")

    motif_matching = MotifMatching(symbol=symbol, start_date=start_date)
    pattern_start_date, pattern_end_date, distance = motif_matching.find_similar_subseries_with_date()

    report = ""
    report += f"The similar pattern for {symbol} from {start_date} to current day\n"
    report += f"- Indices: from {pattern_start_date} to {pattern_end_date} (Window_size m = {motif_matching.m})\n"
    report += f"- Distance: {distance:.2f}\n"
    # Send the report to the user
    bot.send_message(message.chat.id, report)

# @bot.message_handler(commands=['mulpattern'])
def find_similar_pattern_multi_dimension(message, symbol):

    start_date = message.text # %Y-%m-%d
    bot.send_message(message.chat.id, "Please wait. This process can takes several minutes")

    motif_matching = MotifMatching(symbol=symbol, start_date=start_date)
    ticker, result = motif_matching.find_matching_series_multi_dim_with_date()
    pattern_start_date = motif_matching.dataframe.iloc[int(result[0])].time.strftime('%Y-%m-%d')
    pattern_end_date= motif_matching.dataframe.iloc[int(result[0]) + motif_matching.m].time.strftime('%Y-%m-%d')
    distance = result[2]

    report = ""
    report += f"The similar pattern for {ticker} from {start_date} to current day in multi dimension ['close','volume']\n"
    report += f"- Indices: from {pattern_start_date} to {pattern_end_date} (Window_size m = {motif_matching.m})\n"
    report += f"- Distance: {distance:.2f}\n"
    # Send the report to the user
    bot.send_message(message.chat.id, report)

def get_paybacktime(message):
    # Get the symbol from the user's message
    symbol = message.text.upper()
    if not validate_symbol(symbol):
        bot.send_message(message.chat.id, f'Sorry! There is no stock {symbol}')
        return 
    
    bot.send_message(message.chat.id, "Please wait. This process can takes several minutes")
    # Create the PayBackTime object and get the report
    pbt_generator = PayBackTime(symbol=symbol, report_range='yearly', window_size=10)
    report = pbt_generator.get_report()
    
    # Send the report to the user
    bot.send_message(message.chat.id, report)

def calculate_risk(message):
    symbol = message.text.upper()
    if not validate_symbol(symbol):
        bot.send_message(message.chat.id, f'Sorry! There is no stock {symbol}')
        return 

    pbt_generator = PayBackTime(symbol=symbol, report_range='yearly', window_size=10)
    stock_price = pbt_generator.get_current_price()
    num_stocks = calculate_stocks_to_buy(stock_price)
    mess = ""
    mess += f"You can buy {num_stocks} stocks at the price of {stock_price} each\n"
    mess += f'Total price: {stock_price*num_stocks/1_000_000} (triệu VND)\n'
    bot.send_message(message.chat.id, mess)


def rate(message):
    symbol = message.text.upper()
    if not validate_symbol(symbol):
        bot.send_message(message.chat.id, f'Sorry! There is no stock {symbol}')
        return 
    
    if len(symbol) > 3: # Its not for Index
        bot.send_message(message.chat.id, 'This function can just be used for stocks, not index')
        return
    
    rating = general_rating(symbol)
    report = f"The general rating for {symbol}:\n"
    report += "\n".join([f"{col}: {rating[col].values[0]}" for col in rating.columns])

    # Send the report to the user
    bot.send_message(message.chat.id, report)

@bot.message_handler(commands=['findpbt'])
def findpbt(message):
    bot.send_message(message.chat.id, "Please wait. This process can takes several minutes")
    pass_ticker = find_PBT_stocks()
    pass_ticker_string = ", ".join(pass_ticker)
    print('pass_ticker_string',pass_ticker_string)

    # Send the report to the user
    bot.send_message(message.chat.id, f"The Paybacktime stocks are {pass_ticker_string}")

@bot.message_handler(commands=['findmyfav'])
def findmyfav(message):
    my_params = {
        "exchangeName": "HOSE,HNX",
        "marketCap": (1_000, 200_000),
        "roe": (10, 100),  # Minimum Return on Equity (ROE)
        'pe': (10,20),
        'priceNearRealtime': (10,100),
        # "avgTradingValue20Day": (100, 1000),  # Minimum 20-day average trading value
    #     'uptrend': 'buy-signal',
        'macdHistogram': 'macdHistLT0Increase',
        'strongBuyPercentage': (20,100),
        'relativeStrength3Day': (50,100),
    }
    
    pass_ticker = filter_stocks(param=my_params)
    pass_ticker_string = ", ".join(pass_ticker.ticker.unique())

    # Send the report to the user
    bot.send_message(message.chat.id, f"The stocks most suitable for u are: {pass_ticker_string}")


@bot.message_handler(commands=['findbestmotif'])
def findpbt(message):
    bot.send_message(message.chat.id, "Please wait. This process can takes several minutes")
    result_dict = find_best_motifs()
    report = ""
    for stock, values in result_dict.items():
        start_date, end_date, distance = values
        report += f"Stock: {stock}\n"
        report += f"- Date: {start_date} to {end_date}\n"
        report += f"- Distance: {distance:.3f}\n\n"
    # Send the report to the user
    bot.send_message(message.chat.id, report)

# Define the function to handle all other messages
@bot.message_handler(func=lambda message: True)
def echo(message):
    response_message = "Apologies, I didn't understand that command. 😕\nPlease type /help to see the list of available commands."
    bot.send_message(message.chat.id, response_message)

def main():
    bot.polling()

if __name__ == "__main__":
    main()