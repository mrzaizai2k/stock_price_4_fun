import sys
sys.path.append("")
import os 
import telebot
from dotenv import load_dotenv
load_dotenv()

# from telegram.ext.filters import Filters 
from src.PayBackTime import PayBackTime

TELEBOT_API= os.getenv('TELEBOT_API')
print('key', TELEBOT_API)
bot = telebot.TeleBot(TELEBOT_API)

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Welcome to the Mrzaizai2k Stock Assistant bot! Type /help to see the available commands.")

@bot.message_handler(commands=['help'])
def help(message):
    bot.send_message(message.chat.id, "Available commands:\n/paybacktime + symbol - Calculate the payback time for a stock\n/help - Show this help message")

@bot.message_handler(commands=['paybacktime'])
def paybacktime(message):
    # Get the symbol from the user's message
    symbol = message.text.split()[1].upper()
    
    # Create the PayBackTime object and get the report
    pbt_generator = PayBackTime(symbol=symbol, report_range='yearly', window_size=10)
    report = pbt_generator.get_string_report()
    
    # Send the report to the user
    bot.send_message(message.chat.id, report)


# Define the function to handle all other messages
def echo(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, 
                             text="Sorry, I didn't understand that command. Type /help to see the available commands.")

def main():
    bot.polling()

if __name__ == "__main__":
    main()