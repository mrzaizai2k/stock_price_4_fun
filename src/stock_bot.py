import sys
sys.path.append("")
import os 
import telebot
from dotenv import load_dotenv
load_dotenv()

# from telegram.ext.filters import Filters 
from src.PayBackTime import PayBackTime, find_PBT_stocks
from src.utils import *
from src.motif import MotifMatching

TELEBOT_API= os.getenv('TELEBOT_API')
print('key', TELEBOT_API)
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


@bot.message_handler(commands=['pbt'])
def get_paybacktime(message):
    # Get the symbol from the user's message
    bot.send_message(message.chat.id, "Please wait. This process can takes several minutes")
    symbol = message.text.split()[1]
    # Create the PayBackTime object and get the report
    pbt_generator = PayBackTime(symbol=symbol, report_range='yearly', window_size=10)
    report = pbt_generator.get_string_report()
    
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

@bot.message_handler(commands=['risk'])
def calculate_risk(message):
    symbol = message.text.split()[1]
    pbt_generator = PayBackTime(symbol=symbol, report_range='yearly', window_size=10)
    stock_price = pbt_generator.get_current_price()
    num_stocks = calculate_stocks_to_buy(stock_price)
    mess = ""
    mess += f"You can buy {num_stocks} stocks at the price of {stock_price} each\n"
    mess += f'Total price: {stock_price*num_stocks/1_000_000} (triệu VND)\n'
    bot.send_message(message.chat.id, mess)

@bot.message_handler(commands=['rate'])
def findpbt(message):
    symbol = message.text.split()[1].upper()
    rating = general_rating(symbol).to_string()
    # Send the report to the user
    bot.send_message(message.chat.id, rating)

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

@bot.message_handler(commands=['pattern'])
def find_similar_pattern(message):
    symbol = message.text.split()[1].upper()
    start_date = message.text.split()[2] # %Y-%m-%d
    motif_matching = MotifMatching(symbol=symbol, start_date=start_date)
    pattern_start_date, pattern_end_date, distance = motif_matching.find_similar_subseries_with_date()

    report = ""
    report += f"The similar pattern for {symbol} from {start_date} to current day\n"
    report += f"Indices: from {pattern_start_date} to {pattern_end_date} (Window_size m = {motif_matching.m})\n"
    report += f"Distance: {distance}\n"
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
