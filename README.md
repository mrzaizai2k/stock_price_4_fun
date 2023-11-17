# Mrzaizai2k Stock Assistant Bot

## Table of Contents
1. [Introduction](#introduction)
2. [Available Commands](#available-commands)
3. [How to Use](#how-to-use)
3. [How to Set Up the Bot](#how-to-set-up-the-bot)

## Introduction
Welcome to the Mrzaizai2k Stock Assistant bot! This bot is designed to assist you with various stock-related tasks and analyses. Whether you want to calculate the payback time for a stock, find support and resistance levels, or receive warnings about specific market conditions, this bot has you covered.

## Available Commands
1. `/start`: Initializes the bot and provides a welcome message.
2. `/help`: Displays a list of available commands and their descriptions.
3. `/pbt + symbol`: Calculates the payback time for a specific stock.
4. `/snr + symbol`: Finds the closest support and resistance levels for a stock.
5. `/findpbt`: Identifies stocks with favorable payback times.
6. `/findmyfav`: Recommends stocks based on specified criteria.
7. `/risk + symbol`: Calculates the number of stocks you can buy with a given loss/trade percentage and maximum loss/capital percentage.
8. `/rate + symbol`: Provides a general rating for a stock.
9. `/mulpattern + symbol + date (YYYY-mm-dd)`: Finds patterns of a stock on multiple dimensions (close, volume).
10. `/pattern + symbol + date (YYYY-mm-dd)`: Finds patterns of a stock based on closing prices.
11. `/findbestmotif`: Identifies the best motifs across all stocks.
12. `/warningmacd`: Checks for MACD (Moving Average Convergence Divergence) signals.
13. `/warningpricevsma`: Checks if the stock price crosses the Exponential Moving Average (EMA).
14. `/warningbigday`: Checks for Follow Through Day (FTD) or Distribution Day signals.
15. `/warningsnr`: Checks if the stock price is within the support or resistance range.

## How to Use
1. Start a command with a forward slash ("/") followed by the desired command.
   - Example: `/pbt AAPL`
2. Follow the bot's prompts to enter additional information, such as stock symbols or dates.
3. Wait for the bot to process your request. Some commands may take several minutes.
4. Review the bot's responses for the calculated results, warnings, or recommended stocks.


## How to Set Up the Bot

### Prerequisites
Before running the Mrzaizai2k Stock Assistant Bot, follow these steps to set up the environment:

1. **Clone the Repository:**
   - Close the Git repository to your local machine:
     ```bash
     git clone [repository_url]
     ```

2. **Install Dependencies:**
   - Navigate to the project directory and install the required packages using the provided `setup_win.txt` file:
     ```bash
     pip install -r setup_win.txt
     ```

3. **Get TELEBOT_API Key:**
   - Go to [BotFather](https://t.me/botfather) on Telegram and create a new bot.
   - Copy the generated `TELEBOT_API` key.

4. **Create .env File:**
   - Create a new file named `.env` in the project root directory.
   - Add the following line to the file, replacing `YOUR_TELEBOT_API_KEY` with the key obtained from BotFather:
     ```env
     TELEBOT_API=YOUR_TELEBOT_API_KEY
     ```

5. **Run the Bot:**
   - Execute the following command to run the Mrzaizai2k Stock Assistant Bot:
     ```bash
     python src/stock_bot.py
     ```

6. **Interact with the Bot:**
   - Once the bot is running, open Telegram and start interacting with the bot using the available commands.

Note: Ensure that you have Python installed on your machine and the necessary permissions to install packages and run scripts.


Feel free to explore the various commands and leverage the bot for your stock-related analyses. If you have any questions or encounter issues, type `/help` for assistance. Happy investing!
