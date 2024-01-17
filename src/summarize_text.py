import sys
sys.path.append("")
import torch
import whisper
import requests
import re
import time
import random
import json
import urllib3
import ast
from dotenv import load_dotenv
load_dotenv()
import os

urllib3.disable_warnings()
from typing import Literal
from transformers import pipeline
from bs4 import BeautifulSoup
from src.Utils.utils import check_path, take_device

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    NewsURLLoader,
)
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.text_splitter import TokenTextSplitter

from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from ctransformers import AutoModelForCausalLM, AutoTokenizer
from langchain_openai import OpenAI
from datetime import datetime


class SeperateTaskPrompt:
    def __init__(self, template_path:str = 'config/seperate_task_template.txt', ):

        self.template_path = template_path
        with open(self.template_path, 'r') as file:
            self.template = file.read()
        self.load_llm()
        # self.prompt = PromptTemplate(template=self.template, input_variables=['date', 'question'])
        # self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm)

    def load_llm(self):
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.llm = OpenAI(openai_api_key=OPENAI_API_KEY, 
                          max_tokens = 512,
                          temperature=0.7,
                          
                          )
        
    def get_response(self, text) -> list:

        tmp_prompt = f"{self.template}\n Now at {datetime.now()}, please break down this text: {text}"
        response = self.llm(tmp_prompt)
        # print('raw', response)
        response = self.get_tasks_from_string(response)
        return response

    def get_tasks_from_string(self, text:str) -> list:
        # Find the substring between '[' and ']'
        start_index = text.find('[')
        end_index = text.find(']') + 1
        json_string = text[start_index:end_index]
        tasks_list = eval(json_string)
        return tasks_list


class GoogleTranslator:
    def __init__(self):
        pass

    def translate(self, text, to_lang):
        url = 'https://translate.googleapis.com/translate_a/single'

        params = {
        'client': 'gtx',
        'sl': 'auto',
        'tl': to_lang,
        # 'hl': from_lang,
        'dt': ['t', 'bd'],
        'dj': '1',
        'source': 'popup5',
        'q': text
        }
        translated_text = ""
        data = requests.get(url, params=params, verify=False).json()
        sentences = data['sentences']
        for sentence in sentences:
            translated_text += f"{sentence['trans']}\n"
        return translated_text

class SpeechSummaryProcessor:
    '''
    Capture Ideas with Whisper and Translate them to English
    https://colab.research.google.com/github/AndreDalwin/Whisper2Summarize/blob/main/Whisper2Summarize_Colab_Edition.ipynb#scrollTo=y3CCY-m4Wbo6
    
    audio =  # Make sure you upload the audio file (mp3,wav,m4a) into the session storage!
    model = "base" #possible options are 'tiny', 'base', 'small', 'medium', and 'large'
    '''
    def __init__(self, audio_path: str, 
                 whisper_model: Literal['base', 'small'] = 'base', 
                 translator = GoogleTranslator(), 
                 task_seperator = SeperateTaskPrompt()):
        
        # Step 1: Initialize the SpeechToTextProcessor
        self.device = take_device()
        self.whisper_model = whisper.load_model(whisper_model, device=self.device)

        # Step 2: Load and preprocess the audio
        self.audio_path = audio_path
        self.audio = whisper.load_audio(audio_path)
        self.translator = translator
        self.task_seperator = task_seperator

    def transcibe_text_from_sound(self):
        # Step 3: Perform speech-to-text conversion
        audio_copy = self.audio.copy()
        self.result = self.whisper_model.transcribe(audio_copy, verbose=None, 
                                               fp16=False, temperature=0.5)
        self.language = self.result['language']
        return self.result, self.language

    def segment_text(self, result) -> str:
        # Step 4: Segment the transcribed text
        segments = result['segments']
        # Join the segment texts into one string
        segmented_text = '.'.join(segment['text'] for segment in segments)
        try:
            # Send the joined text to self.task_seperator.get_response() to get a list
            self.response_list = self.task_seperator.get_response(text=segmented_text)
            # Separate the received list by newline
            titles = [task['title'] for task in self.response_list]
            # Join titles into a single string with newline separator
            segmented_text = '\n'.join(titles)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            segmented_text = re.sub(r'[\.,\n]', '\n', segmented_text)  # Replace dots, commas, and newlines with newlines
            segmented_text = segmented_text.replace(' and ', '\n')  # Replace 'and' with newlines separate
            
        return segmented_text

    def translate_to_english(self, text, to_lang='en'):
        # Step 5: Translate the text to English
        translated_text = self.translator.translate(text,to_lang=to_lang)
        return translated_text

    def generate_speech_to_text(self):
        # Step 6: Perform the overall processing and translation
        result, language = self.transcibe_text_from_sound()
        segmented_text = self.segment_text(result)

        if language == 'en':
            return segmented_text
        else:
            return self.translate_to_english(segmented_text)
        
    def get_task_list(self):
        return self.response_list



class NewsScraper:
    '''
    Scape News from https://vnexpress.net/ 
    How to run selenium on linux
    https://cloudbytes.dev/snippets/run-selenium-and-chrome-on-wsl2#:~:text=With%20Selenium%20libraries%2C%20Python%20can,using%20Python%20and%20Selenium%20webdriver.
    
    '''
    def __init__(self):
        pass

    def search_stock_news(self, symbol:str = "SSI",
                          date_format:Literal['day', 'week', 'month', 'year']='day')-> list : 
        symbol = symbol.upper()
        url = f"https://timkiem.vnexpress.net/?search_f=&q={symbol}&date_format={date_format}&"

        # Send a GET request to the webpage
        response = requests.get(url)
        # Check if the request was successful (status code 200)
        attemp = 0
        max_attemps = 3
        news_urls = []

        while attemp <= max_attemps:
            try:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Assuming the news articles are wrapped in <article> elements
                articles = soup.find_all('article', class_='item-news-common')
                # Iterate through each article and extract the URL
                for article in articles:
                    url = article.get('data-url')
                    if url:
                        news_urls.append(url)

                break
                
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                print("Resetting the Scraper in 10 seconds...")
                time.sleep(10)  
                attemp += 1
                if attemp > max_attemps:
                    print("Max attempts reached. Exiting.")
                    break

        return news_urls

    def search_top_news_cafef(self)-> list:

        url = 'https://cafef.vn/'
        # Send a GET request to the webpage
        response = requests.get(url)
        # Check if the request was successful (status code 200)
        attemp = 0
        max_attemps = 3
        news_urls = []

        while attemp <= max_attemps:
            try:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find top news elements
                top_news_elements = soup.find_all('div', class_='top_noibat')
                top_news = top_news_elements[0] if top_news_elements else None

                top_news_links = top_news.find_all('a')
                # Extract and print the links
                for link in top_news_links:
                    news_link = link.get('href')
                    news_link = f'{url}{news_link}'
                    news_urls.append(news_link)

                    # Convert set to list
                    news_urls = list(set(news_urls))
                break
                
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                print("Resetting the Scraper in 10 seconds...")
                time.sleep(10)  
                attemp += 1
                if attemp > max_attemps:
                    print("Max attempts reached. Exiting.")
                    break

        return news_urls
    
    def search_top_news_vnexpress(self)-> list:

        url = 'https://vnexpress.net/kinh-doanh'
        # Send a GET request to the webpage
        response = requests.get(url)
        # Check if the request was successful (status code 200)
        attemp = 0
        max_attemps = 3
        news_urls = []

        while attemp <= max_attemps:
            try:
                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find news elements
                news_elements = soup.find_all('h3', class_='title_news')
                news_elements.append(soup.find('section', class_='section_topstory_folder'))

                # Extract and return the URLs
                news_urls = [element.find('a').get('href') for element in news_elements if element.find('a')]
                break
                
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                print("Resetting the Scraper in 10 seconds...")
                time.sleep(10)  
                attemp += 1
                if attemp > max_attemps:
                    print("Max attempts reached. Exiting.")
                    break

        return news_urls
    
    def search_top_news(self)-> list:
        news_urls = self.search_top_news_cafef()
        news_urls.extend(self.search_top_news_vnexpress())
        return news_urls
    
    def take_text_from_link(self, news_url:str) -> str :
        news_url = [news_url]
        loader = NewsURLLoader(urls=news_url, 
                            post_processors=[clean_extra_whitespace],)
        news_text = loader.load()
        news_text = news_text[0].page_content
        return  news_text
    

class NewsSummarizer:
    def __init__(self, summarizer = pipeline("summarization", 
                                             model="Falconsai/text_summarization", device = take_device()),
                 translator = GoogleTranslator(),
                 max_length:int=230, 
                 min_length:int=30,
                 ):
        self.summarizer = summarizer
        self.translator = translator
        self.max_length = max_length
        self.min_length = min_length
        
    def summary_text(self,text:str)->str:
        '''Summary short text'''
        sum_text = self.summarizer(text, max_length=self.max_length, 
                                   min_length=self.min_length, do_sample=False)[0]['summary_text']
        return sum_text
    
    def summary_news(self, news:str, chunk_overlap:str = 0)->str:

        text_splitter = TokenTextSplitter(chunk_size=self.max_length * 2,
                                           chunk_overlap=chunk_overlap)
        
        trans_news = self.translator.translate(text=news, to_lang='en')
        text_chunks = text_splitter.split_text(trans_news)
        summary_documents = [self.summary_text(chunk) for chunk in text_chunks]
        summary_text = '\n'.join(summary_documents)

        summary_text = self.translator.translate(text=summary_text, to_lang='vi')
        return summary_text
    


class StockNewsDatabase:
    def __init__(self, summary_news_data_path='data/summary_stock_news.json'):
        self.summary_news_data_path = summary_news_data_path
        self.news_scraper = NewsScraper()
        self.new_summarizer = NewsSummarizer()

    def update_top_news(self):
        '''Update the top news on schedule'''
        summary_data = []

        news_list = self.news_scraper.search_top_news()[:5]
        for news_url in news_list:
            news = self.news_scraper.take_text_from_link(news_url=news_url)
            sum_text = self.new_summarizer.summary_news(news=news)

            # Append data to summary_data list
            summary_data.append({
                "type": "top_news",
                "stock": None,
                "summary_text": sum_text,
                "news_url": news_url
            })

        return summary_data


    def update_stock_news(self, watch_list):
        '''Update the news summary on schedule'''
        summary_data = []

        for stock in watch_list:
            news_list = self.news_scraper.search_stock_news(symbol=stock, date_format='month')[:1]
            for news_url in news_list:
                news = self.news_scraper.take_text_from_link(news_url=news_url)
                sum_text = self.new_summarizer.summary_news(news=news)

                # Append data to summary_data list
                summary_data.append({
                    "type": "stock_news",
                    "stock": stock,
                    "summary_text": sum_text,
                    "news_url": news_url
                })
        return summary_data
    
    def update_news(self, watch_list):
        summary_data = self.update_top_news()
        summary_data.extend(self.update_stock_news(watch_list=watch_list))
        # Save the summary data to a JSON file
        self._save_summary_data(summary_data)


    def read_summary_stock_news(self):
        try:
            with open(self.summary_news_data_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                return data
        except FileNotFoundError:
            print(f"Error: File {self.summary_news_data_path} not found.")
            return None

    def extract_text_for_stock(self, stock_symbol):
        summary_data = self.read_summary_stock_news()
        if summary_data is None:
            return None, None
        
        for entry in summary_data:
            if entry["stock"] == stock_symbol:
                return entry["summary_text"], entry["news_url"]

        print(f"No summary text found for stock: {stock_symbol}")
        return None, None

    def get_top_news(self)-> str:
        summary_data = self.read_summary_stock_news()

        if summary_data is None:
            return None
        
        top_news = [entry for entry in summary_data if entry["type"] == "top_news"]
        
        report = ""
        for entry in top_news:
            report += entry["summary_text"] + "\n"
            report += entry["news_url"] + "\n"
            report += "----------\n"

        return report
        
    def get_all_stocks(self):
        summary_data = self.read_summary_stock_news()

        if summary_data:
            # Use a set comprehension to filter out None values
            all_stocks = {entry["stock"] for entry in summary_data if entry["stock"] is not None}
            return list(all_stocks)
        else:
            return []


    def _save_summary_data(self, summary_data):
        check_path(self.summary_news_data_path)
        with open(self.summary_news_data_path, 'w', encoding='utf-8') as json_file:
            json.dump(summary_data, json_file, ensure_ascii=False, indent=2)
        print(f"Summary data saved to {self.summary_news_data_path}")
        
if __name__ == "__main__":
    speech_to_text = SpeechSummaryProcessor(audio_path='sample_voice.m4a')
    text = speech_to_text.generate_speech_to_text()
    print ('Text', text)

    symbol = 'SSI'
    date_format='year'
    news_scraper = NewsScraper()
    news_list = news_scraper.search_stock_news(symbol=symbol, date_format=date_format)
    news = news_scraper.take_text_from_link(news_url=news_list[0])
    new_summarizer = NewsSummarizer()
    sum_text = new_summarizer.summary_news(news= news)
    print('sum_text', sum_text)
    news_db = StockNewsDatabase()
    print(news_db.get_all_stocks())

    text = "I have to go to the gym at 9, and figure out the power of 2, I have to call my parent at 6 in my house, and do my homework with james"
    seperator = SeperateTaskPrompt()
    task_lists = seperator.get_response(text)
    print(task_lists)
