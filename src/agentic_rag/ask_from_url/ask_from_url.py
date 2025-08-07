# Text Preprocessing of Url Content

import re
import unicodedata
import json
import os
import requests
from typing import List, Union, Optional
from duckduckgo_search import DDGS


from bs4 import BeautifulSoup
from readability import Document
# for progress bars, requires !pip install tqdm and run pip install -U jupyter ipywidgets
from tqdm.auto import tqdm
import pandas as pd
from fpdf import FPDF



import nltk
from spacy.lang.en import English


# Defining it globally
nlp = English()                                  
# Add a sentencizer pipeline           
nlp.add_pipe("sentencizer")                   



from agentic_rag.get_url_content.fetch_cleaned_url_content import FetchURLContent


from agentic_rag.rag_input_preprocessor.text_preprocessing import TextPreprocessing
from agentic_rag.constants.constants import (URL_pages_and_chunks_df_filepath,
                                             weburl_filename_path,
                                             is_min_token_length_required,
                                             URL_sentences_per_chunk
                                            )



# Create a class to handle PDF generation
class TextToPDF:
    
    """
    Converts plain text into a PDF document and optionally saves it to a specified output directory.
    """
    
    def __init__(self, output_dir: str = None):
        
        """
        Parameters : 
            output_dir (str, optional): Directory to save the generated PDF. Defaults to current working directory.
        """
        
        self.output_dir = output_dir

        if self.output_dir:
            # Create the output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)


    def save_to_pdf(self, text: str, filename: str = "output.pdf")-> str:
        
        """
        Saves given text content to a PDF file.

        parameters : 
            (a) text (str): The cleaned text content to be saved.
            (b) filename (str): Output PDF filename.

        Returns:
            str: The full path of the saved PDF file.
        """
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Break long text into lines for PDF
        for line in text.split('\n'):
            pdf.multi_cell(0, 10, txt=line.strip())

        if self.output_dir : 
            filepath = os.path.join(self.output_dir, filename)
        else : 
            filepath = os.path.join(filename)
            
        pdf.output(filepath)
        print(f"[✅] PDF saved at: {filepath}")

        return filepath




class AskUrl:
    
    """
    Handles end-to-end processing of a web URL for RAG pipelines:
        - Cleans and fetches article text from a URL.
        - Saves it as a PDF file.
        - Runs a sentence-chunking and token-length filtering pipeline.
    """
    
    def __init__(self,
                 web_url: str,
                 save_pages_and_chunks_df: bool = True,
                 sentences_per_chunk: int = URL_sentences_per_chunk,
                 is_min_token_length_required: bool = is_min_token_length_required,
                 filename: str = weburl_filename_path,
                 ):
        
        """
        parameters : 
            (a) web_url (str): The target URL to scrape and process.
            (b) save_pages_and_chunks_df (bool): Whether to save the pages/chunks DataFrame to disk.
            (c) sentences_per_chunk (int): Number of sentences per chunk.
            (d) is_min_token_length_required (bool): Minimum number of tokens required to keep a chunk.
            (e) filename (str): PDF filename to save the cleaned content.
        """
        
        self.web_url = web_url
        self.filename = filename
        #self.filepath = filepath
        self.sentences_per_chunk = sentences_per_chunk
        self.is_min_token_length_required = is_min_token_length_required
        self.save_pages_and_chunks_df = save_pages_and_chunks_df
  
        
    
    
    def save_file_as_pdf(self)-> str:
        
        """
        Fetches and cleans content from the URL and saves it as a PDF.

        Returns:
            str: The full path of the saved PDF file.
        """
        
        # create object 
        fetch_url_Content = FetchURLContent()
        
        cleaned_text = fetch_url_Content.get_url_cleaned_context(web_url=self.web_url)
        
        # STEP 2: Save cleaned content to PDF
        pdf_saver = TextToPDF(output_dir = None)  # optional: specify output dir
        filepath = pdf_saver.save_to_pdf(text=cleaned_text, filename=self.filename)
        
        return filepath
    
        
    def web_text_preprocessing(self)-> List[dict]:
        
        """
        Executes the full preprocessing pipeline on the web URL content:
            - Fetch and clean content.
            - Save as PDF.
            - Run text chunking and page-wise tokenization.

        Returns:
            List[dict]: A list of structured text chunks with metadata.
        """
        
        filepath = self.save_file_as_pdf()
        
        text_preprocessing_object = TextPreprocessing(filepath=filepath,
                                                      sentences_per_chunk=URL_sentences_per_chunk,
                                                      is_min_token_length_required=self.is_min_token_length_required,
                                                      pages_and_chunks_df_filepath=URL_pages_and_chunks_df_filepath,
                                                      save_pages_and_chunks_df=True,
                                                        
                                                      )
        
        url_pages_and_chunks = text_preprocessing_object.run_pipeline()

        return url_pages_and_chunks


