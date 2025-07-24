# Text Preprocessing of Url Content
import re
from bs4 import BeautifulSoup
import unicodedata
from duckduckgo_search import DDGS

import requests
from typing import List, Union, Optional
from readability import Document

# for progress bars, requires !pip install tqdm and run pip install -U jupyter ipywidgets
from tqdm.auto import tqdm
import json

import nltk
from spacy.lang.en import English
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd

nltk.download('punkt')  # Only needed once
nltk.download('punkt_tab')


# Defining it globally
nlp = English()                                  
# Add a sentencizer pipeline           
nlp.add_pipe("sentencizer")                   
# Page-wise stats 


from agentic_rag.get_url_content.fetch_cleaned_url_content import FetchURLContent


from fpdf import FPDF
import os

from agentic_rag.rag_input_preprocessor.text_preprocessing import TextPreprocessing
from agentic_rag.constants.constants import URL_pages_and_chunks_df_filepath, weburl_filename_path



# Create a class to handle PDF generation
class TextToPDF:
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir

        if self.output_dir:
            # Create the output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)

    def save_to_pdf(self, text: str, filename: str = "output.pdf"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Break long text into lines for PDF
        lines = text.split('\n')
        for line in lines:
            pdf.multi_cell(0, 10, txt=line.strip())

        if self.output_dir : 
            filepath = os.path.join(self.output_dir, filename)
        else : 
            filepath = os.path.join(filename)
            
        pdf.output(filepath)
        print(f"[✅] PDF saved at: {filepath}")

        return filepath




class AskUrl:
    
    def __init__(self,
                 web_url,
                 save_pages_and_chunks_df=True,
                 sentences_per_chunk=5,
                 min_token_length=None,
                 filename=weburl_filename_path,
                 ):
        
        self.web_url = web_url
        self.filename = filename
        #self.filepath = filepath
        self.sentences_per_chunk = sentences_per_chunk
        self.min_token_length = min_token_length,
        self.save_pages_and_chunks_df = save_pages_and_chunks_df
  
        
    
    
    def save_file_as_pfd(self):
        
        # create objet 
        fetch_url_Content = FetchURLContent()
        
        cleaned_text = fetch_url_Content.get_url_cleaned_context(web_url=self.web_url)
        
        # STEP 2: Save cleaned content to PDF
        pdf_saver = TextToPDF(output_dir = None)  # optional: specify output dir
        filepath = pdf_saver.save_to_pdf(text=cleaned_text, filename=self.filename)
        
        return filepath
    
        
    def web_text_preprocessing(self):
        
        
        filepath = self.save_file_as_pfd()
        
        text_preprocessing_object = TextPreprocessing(filepath = filepath,
                                                      sentences_per_chunk = 5,
                                                      min_token_length = None,
                                                      pages_and_chunks_df_filepath = URL_pages_and_chunks_df_filepath,
                                                      save_pages_and_chunks_df = True,
                                                        
                                                      )
        
        
        
        
        url_pages_and_chunks = text_preprocessing_object.run_pipeline()

        return url_pages_and_chunks


