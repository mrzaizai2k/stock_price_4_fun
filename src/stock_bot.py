import sys
sys.path.append("")
import os 
import yaml
import schedule
from threading import Thread
from time import sleep

import telebot
from telebot import types
from dotenv import load_dotenv
load_dotenv()

# from telegram.ext.filters import Filters 
from src.PayBackTime import PayBackTime, find_PBT_stocks
from src.utils import *
from src.motif import MotifMatching, find_best_motifs
from src.Indicators import *
from src.support_resist import SupportResistFinding



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
    bot.send_message(message.chat.id, "\n/snr + symbol: Find a closest support and resistance for a stock")
    bot.send_message(message.chat.id, "\n/findpbt: find payback time stocks right now")
    bot.send_message(message.chat.id, "\n/findmyfav: find my_param stocks right now")
    bot.send_message(message.chat.id, "\n/risk + symbol: calculate how much stocks u can buy with loss/trade = 6% with max loss/capital = 2%")
    bot.send_message(message.chat.id, "\n/rate + symbol: general rating the stock")
    bot.send_message(message.chat.id, "\n/mulpattern + symbol + date (YYYY-mm-dd): find pattern of the stock on multi-dimension ['close', 'volume']")
    bot.send_message(message.chat.id, "\n/pattern + symbol + date (YYYY-mm-dd): find pattern of the stock ['close']")
    bot.send_message(message.chat.id, "\n/findbestmotif: Find the best motif on all the stocks")
    # bot.send_message(message.chat.id, "\n/warningmacd: Check macd")
    # bot.send_message(message.chat.id, "\n/warningpricevsma: check if price cross EMA")
    # bot.send_message(message.chat.id, "\n/warningbigday: Check Follow Through Day or Distribution Day")
    # bot.send_message(message.chat.id, "\n/warningsnr: Check if price is in Support or Resistance Range")

@bot.message_handler(commands=['rate', 'risk', 'pbt','mulpattern', 'pattern','snr'])
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
    elif message.text == '/snr':
        bot.register_next_step_handler(message, get_support_resistance)
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

def get_support_resistance(message):
    # Get the symbol from the user's message
    symbol = message.text.upper()
    if not validate_symbol(symbol):
        bot.send_message(message.chat.id, f'Sorry! There is no stock {symbol}')
        return 
  
    # Create the PayBackTime object and get the report
    sr_finding = SupportResistFinding(symbol=symbol)
    result = sr_finding.find_closest_support_resist(current_price=sr_finding.get_current_price())
    report = f'The current price for {symbol} is {sr_finding.get_current_price()}\n'
    report += f'- The closest support is {result[0]}\n'
    report += f'- The closest resistance is {result[1]}\n'
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
    pass_ticker = find_PBT_stocks(file_path="memory/paybacktime.csv")
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

# @bot.message_handler(commands=['warningmacd'])
# def warning_macd(message):  # Pass the message parameter
#     bot.send_message(message.chat.id, "Please wait. This process can takes several minutes")
#     with open('config/config.yaml', 'r') as file:
#         data = yaml.safe_load(file)
#     watchlist = data.get('my_watchlist', [])

#     warning_report = []  # Initialize an empty list to store warning reports
#     for symbol in watchlist:
#         macd = MACD(symbol)
#         if macd.is_cross_up(offset=3):
#             warning_report.append(f'{symbol}: Crossed up')
#         elif macd.is_cross_down(offset=3):
#             warning_report.append(f'{symbol}: Crossed down')
#     if warning_report:
#         # If there are warnings, send a report
#         report_message = '\n'.join(warning_report)
#         bot.send_message(message.chat.id, f'Report for stocks with warnings:\n{report_message}')
#     else:
#         # If no warnings, send a message indicating that
#         bot.send_message(message.chat.id, 'There is no warning for any stock in your watchlist')
        


# @bot.message_handler(commands=['warningbigday'])
# def warningbigday(message):  # Pass the message parameter
#     with open('config/config.yaml', 'r') as file:
#         data = yaml.safe_load(file)
#     watchlist = data.get('my_watchlist', [])
#     warning_report = []  # Initialize an empty list to store warning reports

#     for symbol in watchlist:
#         bigday = BigDayWarning(symbol, percent_diff=3)
#         if bigday.is_big_increase():
#             warning_report.append(f'Powerful UP for {symbol}')
#         if bigday.is_big_decrease():
#             warning_report.append(f'Powerful DOWN for {symbol}')

#     if warning_report:
#         # If there are FTD warnings, send a report
#         report_message = '\n'.join(warning_report)
#         bot.send_message(message.chat.id, f'Report for stocks with big day warnings:\n{report_message}')
#     else:
#         # If no FTD warnings, send a message indicating that
#         bot.send_message(message.chat.id, 'There is no big day warning for any stock in your watchlist')

# @bot.message_handler(commands=['warningpricevsma'])
# def warningpricevsma(message):  # Pass the message parameter
#     bot.send_message(message.chat.id, "Please wait. This process can takes several minutes")
#     with open('config/config.yaml', 'r') as file:
#         data = yaml.safe_load(file)
#     watchlist = data.get('my_watchlist', [])
#     warning_report = []  # Initialize an empty list to store warning reports

#     for symbol in watchlist:
#         pvma = PricevsMA(symbol)
#         if pvma.is_cross_up(offset=3):
#             warning_report.append(f'{symbol}: Crossed up')
#         elif pvma.is_cross_down(offset=3):
#             warning_report.append(f'{symbol}: Crossed down')

#     if warning_report:
#         # If there are warnings, send a report
#         report_message = '\n'.join(warning_report)
#         bot.send_message(message.chat.id, f'Report for stocks with PricevsMA warnings:\n{report_message}')
#     else:
#         # If no warnings, send a message indicating that
#         bot.send_message(message.chat.id, 'There is no PricevsMA warning for any stock in your watchlist')


# @bot.message_handler(commands=['warningsnr'])
# def warningsnr(message, tolerance_percent:float = 1.0):  # Pass the message parameter
#     bot.send_message(message.chat.id, "Please wait. This process can takes several minutes")
#     with open('config/config.yaml', 'r') as file:
#         data = yaml.safe_load(file)
#     watchlist = data.get('my_watchlist', [])

#     warning_report = []  # Initialize an empty list to store warning reports
#     for symbol in watchlist:
#         sr_finding = SupportResistFinding(symbol=symbol)
#         current_price = sr_finding.get_current_price()

#         # Find the closest support and resistance levels
#         support, resistance = sr_finding.find_closest_support_resist(current_price=current_price)

#         # Set the tolerance percentage
#         tolerance_percent = tolerance_percent / 100  # 1 percent

#         # Check if the current price is within the tolerance range of support or resistance
#         if support <= current_price <= support * (1 + tolerance_percent):
#             warning_report.append(f'{symbol}: Meeting Support')
#         elif resistance * (1 - tolerance_percent) <= current_price <= resistance:
#             warning_report.append(f'{symbol}: Meeting Resistance')

#     if warning_report:
#         # If there are warnings, send a report
#         report_message = '\n'.join(warning_report)
#         bot.send_message(message.chat.id, f'Support/Resistance warning for stocks:\n{report_message}')
#     else:
#         # If no warnings, send a message indicating that
#         bot.send_message(message.chat.id, 'There is no Support/Resistance warning for any stock in your watchlist')

def warning_macd(watchlist, user_id):  # Pass the message parameter

    warning_report = []  # Initialize an empty list to store warning reports
    for symbol in watchlist:
        macd = MACD(symbol)
        if macd.is_cross_up(offset=3):
            warning_report.append(f'{symbol}: Crossed up')
        elif macd.is_cross_down(offset=3):
            warning_report.append(f'{symbol}: Crossed down')
    if warning_report:
        # If there are warnings, send a report
        report_message = '\n'.join(warning_report)
        return bot.send_message(user_id, f'Report for stocks with warnings:\n{report_message}')
    else:
        # If no warnings, send a message indicating that
        return bot.send_message(user_id, 'There is no warning for any stock in your watchlist')


def warningbigday(watchlist, user_id):  # Pass the message parameter

    warning_report = []  # Initialize an empty list to store warning reports

    for symbol in watchlist:
        bigday = BigDayWarning(symbol, percent_diff=3)
        if bigday.is_big_increase():
            warning_report.append(f'Powerful UP for {symbol}')
        if bigday.is_big_decrease():
            warning_report.append(f'Powerful DOWN for {symbol}')

    if warning_report:
        # If there are FTD warnings, send a report
        report_message = '\n'.join(warning_report)
        return bot.send_message(user_id, f'Report for stocks with big day warnings:\n{report_message}')
    else:
        # If no FTD warnings, send a message indicating that
        return bot.send_message(user_id, 'There is no big day warning for any stock in your watchlist')


def warningpricevsma(watchlist, user_id):  # Pass the message parameter
    warning_report = []  # Initialize an empty list to store warning reports

    for symbol in watchlist:
        pvma = PricevsMA(symbol)
        if pvma.is_cross_up(offset=3):
            warning_report.append(f'{symbol}: Crossed up')
        elif pvma.is_cross_down(offset=3):
            warning_report.append(f'{symbol}: Crossed down')

    if warning_report:
        # If there are warnings, send a report
        report_message = '\n'.join(warning_report)
        return bot.send_message(user_id, f'Report for stocks with PricevsMA warnings:\n{report_message}')
    else:
        # If no warnings, send a message indicating that
        return bot.send_message(user_id, 'There is no PricevsMA warning for any stock in your watchlist')

def warningsnr(watchlist, user_id, tolerance_percent:float = 1.0):  # Pass the message parameter
    warning_report = []  # Initialize an empty list to store warning reports
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
        elif resistance * (1 - tolerance_percent) <= current_price <= resistance:
            warning_report.append(f'{symbol}: Meeting Resistance')

    if warning_report:
        # If there are warnings, send a report
        report_message = '\n'.join(warning_report)
        return bot.send_message(user_id, f'Support/Resistance warning for stocks:\n{report_message}')
    else:
        # If no warnings, send a message indicating that
        return bot.send_message(user_id, 'There is no Support/Resistance warning for any stock in your watchlist')


# Define the function to handle all other messages
@bot.message_handler(func=lambda message: True)
def echo(message):
    response_message = "Apologies, I didn't understand that command. 😕\nPlease type /help to see the list of available commands."
    bot.send_message(message.chat.id, response_message)


def schedule_checker():
    while True:
        schedule.run_pending()
        sleep(1)


def main():

    data_config_path = 'config/config.yaml'
    with open(data_config_path, 'r') as file:
        data = yaml.safe_load(file)

    watchlist = data.get('my_watchlist', [])
    USER_ID = os.getenv('USER_ID')

    schedule.every(1).minutes.do(warning_macd, watchlist=watchlist, user_id=USER_ID)
    schedule.every(1).minutes.do(warningbigday, watchlist=watchlist, user_id=USER_ID)
    schedule.every(1).minutes.do(warningpricevsma, watchlist=watchlist, user_id=USER_ID)
    schedule.every(1).minutes.do(warningsnr, watchlist=watchlist, user_id=USER_ID)


    # Spin up a thread to run the schedule check so it doesn't block your bot.
    # This will take the function schedule_checker which will check every second
    # to see if the scheduled job needs to be ran.
    Thread(target=schedule_checker).start() 

    while True:
        try:
            bot.infinity_polling()
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Resetting the bot in 3 seconds...")
            time.sleep(3)  # Pause for 3 seconds before restarting
            main()  # Restart the main function
        
        finally:
            # Clear scheduled jobs to avoid duplication on restart
            schedule.clear()

if __name__ == "__main__":
    main()
