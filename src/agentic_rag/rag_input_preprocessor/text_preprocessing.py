import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import random, os, requests, re
# for progress bars, requires !pip install tqdm and run pip install -U jupyter ipywidgets
from tqdm.auto import tqdm

# Open and Read the file
import fitz  # PyMuPDF
import docx

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize


try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download('punkt_tab', quiet=True)



from spacy.lang.en import English
import platform, uuid,  tempfile, subprocess

from fpdf import FPDF
#import pythoncom, win32com.client

from agentic_rag.utils_methods.basic_methods import text_formatter
from agentic_rag.constants.constants import (pages_and_chunks_df_filepath,
                                             is_min_token_length_required,
                                             min_token_length)

#import torch

# OS detection
IS_WINDOWS = platform.system() == "Windows"

               

class TextPreprocessing:
    
    def __init__(self, 
                 filepath,
                 sentences_per_chunk, 
                 pages_and_chunks_df_filepath, 
                 is_min_token_length_required=is_min_token_length_required, 
                 save_pages_and_chunks_df :bool=True
                 ):
        
        self.filepath = filepath
        self.sentences_per_chunk = sentences_per_chunk
        self.is_min_token_length_required = is_min_token_length_required
        self.pages_and_chunks_df_filepath = pages_and_chunks_df_filepath
        self.save_pages_and_chunks_df = save_pages_and_chunks_df
        self.nlp = English()
        # Add a sentencizer pipeline 
        self.nlp.add_pipe("sentencizer")
    
    
    # Convert TXT → PDF using FPDF
    def txt_to_pdf(self, txt_path, output_pdf_path):
        
        """
        Converts a plain text (.txt) file into a PDF using the FPDF library.

        Each line from the input text file is added as a separate line in the PDF, 
        with automatic page breaks and basic formatting.

        Parameters : 
            (a) txt_path (str): Path to the input '.txt' file.
            (b) output_pdf_path (str): Path where the resulting PDF will be saved.

        Returns:
            None. The generated PDF is saved to 'output_pdf_path'.
        """
        
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            with open(txt_path, 'r', encoding='utf-8') as file:
                for line in file:
                    pdf.multi_cell(0, 10, txt=line.strip())

            pdf.output(output_pdf_path)
            
        except Exception as e:
            raise RuntimeError(f"[txt_to_pdf] Failed to convert TXT to PDF: {e}")


    # Convert DOCX → PDF (platform-specific)
    def docx_to_pdf(self, docx_path, output_pdf_path):
        
        """
        Converts a DOCX file into a PDF file. Uses platform-specific methods depending on the OS.

            - On "Windows" :  it uses the Microsoft Word COM interface via 'win32com' to open and save the document as PDF.
            - On "Linux/macOS" : it uses 'libreoffice' in headless mode for conversion.

        Parameters : 
            (a) docx_path (str): Path to the input '.docx' file.
            (b) output_pdf_path (str): Path where the converted '.pdf' will be saved.

        Returns:
            None. The resulting PDF is saved to 'output_pdf_path'.

        Raises:
            RuntimeError: If the conversion fails or if required dependencies (Word or LibreOffice) are not available.
        
        Notes:
            - Requires Microsoft Word installed on Windows.
            - Requires LibreOffice installed and available in PATH on Linux/macOS.
            - Automatically suppresses Word UI and alerts during conversion on Windows.
        """
        
        try:
            
            if IS_WINDOWS:
                import pythoncom
                import win32com.client

                pythoncom.CoInitialize()
                word = win32com.client.Dispatch("Word.Application")
                word.Visible = False
                word.DisplayAlerts = False
                doc = word.Documents.Open(docx_path)
                doc.SaveAs(output_pdf_path, FileFormat=17)
                doc.Close()
                word.Quit()
                
            else:
                result = subprocess.run([
                    "libreoffice", "--headless", "--convert-to", "pdf",
                    "--outdir", os.path.dirname(output_pdf_path), docx_path
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                if result.returncode != 0:
                    raise RuntimeError(result.stderr.decode())
                
        except Exception as e:
            raise RuntimeError(f"[docx_to_pdf] Failed to convert DOCX to PDF: {e}")


    def process_page(self, text, page_number):
        
        """
        Processes the raw text content of a page to extract sentence and word-level information.

        This method performs the following:
            - Tokenizes the input text into words using 'word_tokenize'.
            - Segments the text into sentences using a spaCy pipeline ('nlp' with sentencizer).
            - Computes basic statistics for the page such as character count, word count, 
              sentence counts (raw vs. spaCy), and approximate token count.
            - Packages all this information along with the original text and page number into a dictionary.

        Parameters : 
            (a) text (str): The raw extracted text of the page.
            (b) page_number (int): The 0-based index of the page being processed.

        Returns:
            dict: A dictionary containing processed information about the page with the following keys:
                - "page_number": Page number (1-indexed)
                - "page_char_count": Total number of characters on the page
                - "page_word_count": Total number of words
                - "page_sentence_count_raw": Number of sentences based on '.split(". ")'
                - "page_sentence_count_spacy": Number of sentences detected by spaCy
                - "page_token_count": Approximate number of tokens (estimated as len(text) / 4)
                - "text": The original raw text of the page
                - "sentences": A list of sentence strings extracted by spaCy
        """
        try : 
            words = word_tokenize(text)
          
            
            sentences = list(self.nlp(text).sents)
            # Make sure all sentences are strings
            sentences = [str(sentence).strip() for sentence in sentences]


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
            
        except Exception as e:
            raise RuntimeError(f"[process_page] Failed processing page {page_number + 1}: {e}")




    # Main unified reader
    def read_file_details(self, filepath):
        
        """
        Unified file reader that processes '.pdf', '.txt', and '.docx' files into cleaned, page-wise text chunks.

        This method converts supported file types into PDFs if necessary (for '.txt' and '.docx'), then extracts 
        and formats the text from each page using PyMuPDF. 
        
        Each page's content is passed through a custom page processing function ('self.process_page') and 
        returned as a list of structured dictionaries.
        
        Temporary PDF files created during conversion are automatically deleted after processing.

        Parameters:
            (a) filepath (str): Path to the input file. Supported extensions are '.pdf', '.txt', and '.docx'.

        Returns:
            list[dict]: A list of dictionaries where each dictionary corresponds to a page, typically with keys like:
                        - "page_number"
                        - "sentences"
                        - Other metadata from `self.process_page()`

        Raises:
            ValueError: If the file extension is not supported (i.e., not '.pdf', '.txt', or '.docx').
        """
        try : 
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
            
            for i, page in tqdm(enumerate(doc)):
                
                text = page.get_text()
                text = text_formatter(text)
                
                pages_and_texts.append( self.process_page(text=text, page_number=i) )
            doc.close()

            # Cleanup
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

            return pages_and_texts
        
        except Exception as e:
            raise RuntimeError(f"[read_file_details] Error reading file: {e}")



    # Chunking our sentences together
    # we will have to take into account the maximum input token limit of embedding model.
    def chunk_sentences(self, Pages_and_Texts: list[dict], sentences_per_chunk: int) -> list[dict]:
        
        """
        Splits sentence lists for each page into smaller chunks based on a specified sentence limit.

        This method iterates over a list of dictionaries, where each dictionary represents a page or document 
        containing a list of preprocessed sentences. It chunks each list of sentences into sublists of 
        size 'sentences_per_chunk', while preserving the original structure by appending the following keys:

            - sentence_chunks : A list of sentence sublists (chunks)
            - num_chunks : The number of chunks created

        This is particularly useful when preparing input for embedding models that have a maximum input sequence 
        length limit 384. Chunking sentences helps ensure each chunk stays within the model's limit 
        when used downstream in a RAG or embedding pipeline.

        Parameters : 
            
            (a) Pages_and_Texts (list[dict]): A list of dictionaries, each containing a "sentences" key 
                                             with a list of sentence strings.
            (b) sentences_per_chunk (int): The number of sentences to include in each chunk.

        Returns:
        
            list[dict]: The input list with each dictionary updated to include:
                        - sentence_chunks: a list of sentence chunks (sublists)
                        - num_chunks: the total number of chunks per item
    
        """
        
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
    def pages_n_chunks(self, Pages_and_Texts, min_token_length) :
        
        """ 
        Flattens "sentence_chunks" from each page (Pages_and_Texts) into individual chunk records and filters out short chunks.

        This method takes a list of page-level dictionaries (each with sentence-level chunks) and flattens them
        so each sentence chunk becomes a separate dictionary item. 
        
        It also cleans and normalizes each chunk (e.g., fixing punctuation spacing) and calculates basic statistics such as 
        character count, word count, and approximate token count.
        
        
        Optionally, it filters out chunks whose estimated token count is below the 'min_token_length' threshold.
        These are often headings, figure captions, or noisy content with limited semantic value.

        Parameters : 
            (a) Pages_and_Texts (list[dict]): List of dictionaries containing "sentence_chunks" (list of sentence sublists)
                                              and a "page_number" field.
            (b) min_token_length (int): Minimum token length threshold. Chunks with fewer estimated tokens are filtered out.
                                        If None or 0, no filtering is applied.

        Returns:
        
            list[dict]: A flattened list of dictionaries, where each dictionary represents a single cleaned and 
                        preprocessed chunk with the following keys:
                            - "page_number"
                            - "sentence_chunk"
                            - "chunk_char_count"
                            - "chunk_word_count"
                            - "chunk_token_count"
        
        """

        # Split each chunk into its own item
        pages_and_chunks = []

        for item in tqdm(Pages_and_Texts):

            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]

                # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
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

        if self.is_min_token_length_required:
            
            # PreProcessing
            df = pd.DataFrame(pages_and_chunks)

            pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")

            # returning a dictionary
            return pages_and_chunks_over_min_token_len

        else:

            return pages_and_chunks



    def run_pipeline(self):
        
        """
        Runs the complete text processing pipeline on a given file.

        This unified method performs the following sequential steps:
            1. Reads the input file (PDF, DOCX, or TXT) and extracts page-wise text content.
            2. Splits each page's text into sentence-level chunks based on a specified 'number of sentences per chunk'.
            3. Flattens the sentence chunks into standalone records (one per chunk), while filtering out short chunks 
               below a minimum token length threshold.
            4. Optionally saves the processed data to a CSV file for inspection or downstream use.

        Configuration values such as 'filepath', 'sentences_per_chunk', 'min_token_length', and 
        'pages_and_chunks_df_filepath' are expected to be set as instance attributes prior to execution.

        Returns:
            list[dict]: A list of dictionaries, each representing an individual chunk with metadata fields such as:
                - "page_number"
                - "sentence_chunk"
                - "chunk_char_count"
                - "chunk_word_count"
                - "chunk_token_count"
        
        """
        
        try:
            pages_and_texts = self.read_file_details(filepath=self.filepath)

            pages_and_texts = self.chunk_sentences(
                Pages_and_Texts=pages_and_texts,
                sentences_per_chunk=self.sentences_per_chunk
            )

            pages_and_chunks = self.pages_n_chunks(
                Pages_and_Texts=pages_and_texts,
                min_token_length=min_token_length
            )

            if self.save_pages_and_chunks_df:
                try:
                    pages_and_chunks_df = pd.DataFrame(pages_and_chunks)
                    pages_and_chunks_df.to_csv(self.pages_and_chunks_df_filepath, index=False)
                    print(f"file {self.pages_and_chunks_df_filepath} has been saved.")
                except Exception as e:
                    print(f"[run_pipeline] Failed to save CSV: {e}")

            return pages_and_chunks

        except Exception as e:
            raise RuntimeError(f"[run_pipeline] Pipeline failed: {e}")






