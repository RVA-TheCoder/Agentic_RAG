import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import random


import os
import requests
import fitz
# for progress bars, requires !pip install tqdm and run pip install -U jupyter ipywidgets
from tqdm.auto import tqdm

import re

import torch

# Open and Read the file
import fitz  # PyMuPDF
import docx

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')  # Only needed once
nltk.download('punkt_tab')

from spacy.lang.en import English
import platform
import uuid
import tempfile
import subprocess
from fpdf import FPDF

from agentic_rag.utils_methods.basic_methods import text_formatter
from agentic_rag.constants.constants import pages_and_chunks_df_filepath

import pythoncom
import win32com.client


# OS detection
IS_WINDOWS = platform.system() == "Windows"

# Defining it globally
nlp = English()                                  
# Add a sentencizer pipeline           
nlp.add_pipe("sentencizer")                   
# Page-wise stats                     

class TextPreprocessing:
    
    def __init__(self, 
                 filepath,
                 sentences_per_chunk, 
                 pages_and_chunks_df_filepath, 
                 min_token_length=None, 
                 save_pages_and_chunks_df :bool=True
                 ):
        
        self.filepath = filepath
        self.sentences_per_chunk = sentences_per_chunk
        self.min_token_length = min_token_length
        self.pages_and_chunks_df_filepath = pages_and_chunks_df_filepath
        self.save_pages_and_chunks_df = save_pages_and_chunks_df
     
    
    
    # Convert TXT → PDF using FPDF
    def txt_to_pdf(self, txt_path, output_pdf_path):
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        with open(txt_path, 'r', encoding='utf-8') as file:
            for line in file:
                pdf.multi_cell(0, 10, txt=line.strip())

        # Writes the PDF content to disk at the given path.
        pdf.output(output_pdf_path)


    # Convert DOCX → PDF (platform-specific)
    def docx_to_pdf(self, docx_path, output_pdf_path):
        
        if IS_WINDOWS:
            #from docx2pdf import convert
            #convert(docx_path, output_pdf_path)
            
            pythoncom.CoInitialize()  # <- Fixes the COM error
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            word.DisplayAlerts = False
            doc = word.Documents.Open(docx_path)
            doc.SaveAs(output_pdf_path, FileFormat=17)
            doc.Close()
            word.Quit()
            
        else:
            subprocess.run([
                "libreoffice", "--headless", "--convert-to", "pdf",
                "--outdir", os.path.dirname(output_pdf_path), docx_path
            ])


    def process_page(self, text, page_number):
        
        words = word_tokenize(text)
        #sentences = sent_tokenize(text)

        #nlp = English()
        # Add a sentencizer pipeline
        #nlp.add_pipe("sentencizer")
        
        sentences = list(nlp(text=text).sents)
        # Make sure all sentences are strings
        sentences = [str(sentence) for sentence in sentences]

        return {
                    "page_number": page_number + 1, 
                    "page_char_count": len(text),
                    "page_word_count": len(words),
                    "page_sentence_count_raw" : len( text.split(". ") ),
                    "page_sentence_count_spacy" : len(sentences),
                    "page_token_count": len(text) / 4,  # Here token = word for simplicity
                    "text": text,
                    "sentences" : sentences
                }


    # Main unified reader
    def read_file_details(self, filepath):
        
        ext = os.path.splitext(filepath)[1].lower()
        temp_pdf_path = None

        if ext == ".pdf":
            pdf_path = filepath

        elif ext == ".txt":
            temp_pdf_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pdf")
            self.txt_to_pdf(txt_path=filepath, output_pdf_path=temp_pdf_path)
            pdf_path = temp_pdf_path

        elif ext == ".docx":
            temp_pdf_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.pdf")
            self.docx_to_pdf(docx_path=filepath, output_pdf_path=temp_pdf_path)
            pdf_path = temp_pdf_path

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Read PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        pages_and_texts = []

        #nlp = English()
        # Add a sentencizer pipeline
        #nlp.add_pipe("sentencizer")
        
        for i, page in tqdm(enumerate(doc)):
            
            text = page.get_text()
            text = text_formatter(text)
            
            pages_and_texts.append( self.process_page(text=text, page_number=i) )
        doc.close()

        # Cleanup
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

        return pages_and_texts



    # Chunking our sentences together
    # we will have to take into account the maximum input token limit of embedding model.
    def chunk_sentences(self, Pages_and_Texts, sentences_per_chunk):
        
        # Function that recursively splits a list into desired sizes
        def split_list(input_list: list,
                       slice_size: int) -> list[list[str]]:
            
            """
            Splits the input_list into sublists of size slice_size (or as close as possible).

            For example, a list of 16 sentences would be split into two lists of [[10], [6]] if slice_size = 10
            """

            return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

        # Loop through pages and texts and split sentences into chunks
        for item in tqdm(Pages_and_Texts):
            item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                                 slice_size=sentences_per_chunk
                                                )

            item["num_chunks"] = len(item["sentence_chunks"])

        return Pages_and_Texts



    # Splitting each chunk into its own item
    # to embed each chunk of sentences into its own numerical representation.
    def pages_n_chunks(self,Pages_and_Texts, min_token_length) :
        
        """
        Convert sentence_chunks key in the Pages_and_Texts into its own chunk item.
        min_token_length : a filter that will remove the chunk that has a length less than min_token_length.
        
                            Note : There is high probability that these chunks represent : Url, Chapter heading ,
                                figure description etc.,
        """

        # Split each chunk into its own item
        pages_and_chunks = []

        for item in tqdm(Pages_and_Texts):

            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]

                # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
                # joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                # " ".join() instead of  "".join()
                joined_sentence_chunk = " ".join(sentence_chunk).replace("  ", " ").strip()

                # ".A" -> ". A" for any full-stop/capital letter combo
                joined_sentence_chunk = re.sub(pattern=r'\.([A-Z])',
                                            repl=r'. \1',
                                            string=joined_sentence_chunk
                                            )

                chunk_dict["sentence_chunk"] = joined_sentence_chunk

                # Get stats about the chunk
                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])

                # 1 token = ~4 characters
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4

                pages_and_chunks.append(chunk_dict)

        if min_token_length:
            # PreProcessing
            df = pd.DataFrame(pages_and_chunks)

            # There is high probability that these chunks represent : Url, Chapter heading , fig. description etc.,
            min_token_length = 30

            pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")

            # returning a dictionary
            return pages_and_chunks_over_min_token_len

        else:

            return pages_and_chunks



    def run_pipeline(self):
        
        pages_and_texts = self.read_file_details(filepath=self.filepath)
        
        pages_and_texts = self.chunk_sentences(Pages_and_Texts=pages_and_texts, 
                                               sentences_per_chunk=self.sentences_per_chunk    
                                               )
        

        pages_and_chunks = self.pages_n_chunks(Pages_and_Texts=pages_and_texts, 
                                               min_token_length=self.min_token_length
                                               )
        
        if self.save_pages_and_chunks_df :
            
            pages_and_chunks_df = pd.DataFrame(pages_and_chunks)
            
            pages_and_chunks_df.to_csv(self.pages_and_chunks_df_filepath, index=False)
            
            print(f"file {self.pages_and_chunks_df_filepath} has been saved.")
            
        
        
        return pages_and_chunks















