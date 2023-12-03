import torch
import whisper
import requests
import re
import time
import random
import urllib3

urllib3.disable_warnings()
from typing import Literal

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
    def __init__(self, audio_path: str, whisper_model: Literal['base', 'small'] = 'base', 
                 translator = GoogleTranslator()):
        # Step 1: Initialize the SpeechToTextProcessor
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device', self.device)
        self.whisper_model = whisper.load_model(whisper_model, device=self.device)

        # Step 2: Load and preprocess the audio
        self.audio_path = audio_path
        self.audio = whisper.load_audio(audio_path)
        self.translator = translator

    def transcibe_text_from_sound(self):
        # Step 3: Perform speech-to-text conversion
        audio_copy = self.audio.copy()
        self.result = self.whisper_model.transcribe(audio_copy, verbose=None, 
                                               fp16=False, temperature=0.5)
        self.language = self.result['language']
        return self.result, self.language

    def segment_text(self, result):
        # Step 4: Segment the transcribed text
        segments = result['segments']
        segmented_text = ''

        for idx, segment in enumerate(segments):
            segment_text = segment['text']
            segmented_text += f'{idx + 1}: {segment_text}\n'
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


if __name__ == "__main__":
    speech_to_text = SpeechSummaryProcessor(audio_path='sample_voice.m4a')
    text = speech_to_text.generate_speech_to_text()
    print ('Text', text)