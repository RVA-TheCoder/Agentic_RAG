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
import pandas as pd

import nltk
from spacy.lang.en import English
from nltk.tokenize import word_tokenize, sent_tokenize

try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download('punkt_tab', quiet=True)

from agentic_rag.constants.constants import *




# Create a class then add the class methods if required
class FetchURLContent:
    
    """
    A utility class to fetch and clean webpage content for downstream NLP tasks 
    such as embedding and retrieval in RAG pipelines.
    """
    
    def __init__(self):
        
        """Initialize NLP pipeline and output file path."""
        
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        self.output_path = weburl_text_output_json

    # Source :  utils_methods\stage03_utils_method_url_content.py
    def preprocess_web_content(self, content: str) -> str:
        
        """
        Cleans raw web content (HTML or markdown) into plain text.

        Parameters : 
            (a) content (str): Raw content extracted from a web page.

        Returns:
            str: Cleaned plain text.
        """

        # Remove image markdown tags
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)

        # Remove markdown links but keep the text
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)

        # Remove HTML tags if any (optional, as Jina gives markdown)
        content = BeautifulSoup(content, "html.parser").get_text()

        # Replace multiple newlines with one
        content = re.sub(r'\n+', '\n', content)

        # Optionally remove headers like "###", "##", etc.
        content = re.sub(r'^#+\s+', '', content, flags=re.MULTILINE)

        # Remove stray unicode symbols or extra spaces
        content = content.replace('\u200b', '').strip()

        """
        Performs minor formatting on text.
        """

        # Note: This might be different for each doc (best to experiment)
        # strip() : Return a copy of the string with leading and trailing whitespace removed.
        content = content.replace("\n", " ").strip()

        # step : remove non-ascii characters
        content = unicodedata.normalize('NFKD', content)
        content = content.encode('ascii', 'ignore').decode('ascii')
        
        # Step : Normalize whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        # step : remove Visual emphasis
        cleaned_content = re.sub(r"\*+", "", content)

        # Step : General-purpose divider remover: Remove sequences of 3 or more below symbols in a appearing continuously
        #cleaned_text = re.sub(r"[\*\-\=\_#]{3,}", "", cleaned_text)
        
        return cleaned_content



    def fetch_url_content_with_fallback(self, url: str) -> Optional[str]:
        
        """
        Fetches web content using Jina proxy or fallback to raw scraping.

        Parameters : 
            (a) url (str): The URL to fetch.

        Returns:
            Optional[str]: Raw content if successful, else None.
        """
        
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
            )
        }

        # Try r.jina.ai proxy first
        jina_url = f"https://r.jina.ai/{url}"
        
        try:
            
            response = requests.get(jina_url, headers=headers, timeout=20)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 451:
                print(f"[⚠️] Jina blocked ({url}). Falling back...")
            else:
                print(f"[⚠️] Jina failed ({response.status_code}) for {url}")
                
        except requests.RequestException as e:
            print(f"[⚠️] Jina error for {url}: {e}")

        # Fallback to direct scraping using readability
        try:
            
            raw_response = requests.get(url, headers=headers, timeout=20)
            if raw_response.status_code == 200:
                doc = Document(raw_response.text)
                html = doc.summary()
                soup = BeautifulSoup(html, "html.parser")
                return soup.get_text(separator="\n", strip=True)
            
            else:
                print(f"[❌] Fallback failed ({raw_response.status_code}) for {url}")
                
        except requests.RequestException as e:
            print(f"[❌] Fallback scraping failed for {url}: {e}")
        
        return None



    def get_url_cleaned_context(self, web_url):
        
        """
        Fetches and cleans the content from a single URL.

        Parameters : 
            (a) web_url (str): The target web URL.

        Returns:
            str: Cleaned text or empty string if failed.
        """
        
        # STEP 1 : Get the Web raw content
        raw_context = self.fetch_url_content_with_fallback(url=web_url)
        
        # STEP2 : Clean the raw text
        return self.preprocess_web_content(content=raw_context) if raw_context else " "

    
    def process_page(self, text, source_url):
        
        """
        Tokenizes and sentence-splits the given text using spaCy.

        Parameters : 
            (a) text (str): Cleaned text.
            (b) source_url (str): The originating URL.

        Returns:
            dict: Contains sentence list and page-level metadata.
        """
        
        sentences = [str(sentence) for sentence in self.nlp(text=text).sents]

        return {
                    "source_url": source_url, 
                    "text": text,
                    "sentences" : sentences,
                    "webpage_sentence_count_spacy" : len(sentences),
 
                }
    
    
    # Chunking our sentences together
    # we will have to take into account the maximum input token limit of embedding model.
    def chunk_sentences(self, item , sentences_per_chunk):
        
        """
        Breaks the sentence list into smaller chunks of fixed size.

        Parameters : 
            (a) item (dict): A dictionary with a "sentences" key (list of strings).
            (b) sentences_per_chunk (int): Number of sentences per chunk.

        Returns:
            dict: Original item with added 'sentence_chunks' and 'num_chunks'.
        """
        
        # Function that recursively splits a list into desired sizes
        def split_list(input_list: list,
                       slice_size: int) -> list[list[str]]:
            
            """
            Splits the input_list into sublists of size slice_size (or as close as possible).

            For example, a list of 16 sentences would be split into two lists of [[10], [6]] if slice_size = 10
            """

            return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

    
        item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                            slice_size=sentences_per_chunk
                                            )

        item["num_chunks"] = len(item["sentence_chunks"])

        return item
    
    
    
    # Splitting each chunk into its own item
    # to embed each chunk of sentences into its own numerical representation.
    def pages_n_chunks(self, item) :
        
        """
        Converts each sentence chunk into its own dictionary item with metadata.

        Parameters : 
            (a) item (dict): Dictionary with "sentence_chunks" key.

        Returns:
            List[dict]: A list of chunks with chunk-level metadata.
        """

        # Split each chunk into its own item
        Pages_and_Chunks = []
        
        for sentence_chunk in item["sentence_chunks"] :  
            
            joined_sentence_chunk = " ".join(sentence_chunk).replace("  ", " ").strip()

            # ".A" -> ". A" for any full-stop/capital letter combo
            joined_sentence_chunk = re.sub(pattern=r'\.([A-Z])',
                                        repl=r'. \1',
                                        string=joined_sentence_chunk
                                        )
            
            chunk_dict = {
                "source_url": item["source_url"],
                "sentence_chunk": joined_sentence_chunk,
                "chunk_char_count": len(joined_sentence_chunk),
                "chunk_word_count": len(joined_sentence_chunk.split(" ")),
                "chunk_token_count": len(joined_sentence_chunk) / 4  # Approx.
            }

            Pages_and_Chunks.append(chunk_dict)

        return Pages_and_Chunks
    
    
    
    def run_fetch_cleaned_url_content_pipeline(self, urls):
        
        """
        Main pipeline to fetch, clean, split, and chunk content from web URLs.

        Parameters : 
            (a) urls (Union[str, List[str]]): Single URL or list of URLs.

        Returns:
            List[dict]: Final list of flattened sentence chunks with metadata.
        """

        # Type validation
        if isinstance(urls, str):
            urls = [urls]  # Wrap single URL in list
            
        elif isinstance(urls, list):
            if not all(isinstance(u, str) for u in urls):
                raise TypeError("All elements in the URL list must be strings.")
            
        else:
            raise TypeError("`urls` must be a string or a list of strings.")


        cleaned_url_contexts = []
        for url in urls:
            cleaned_context = self.get_url_cleaned_context(web_url=url)
            if cleaned_context:
                cleaned_url_contexts.append({"web_url": url , "cleaned_text" : cleaned_context})

        
        # Save to a file
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_url_contexts, f, indent=4, ensure_ascii=True)
        

        pages_and_texts_list = [self.process_page(text=item["cleaned_text"], source_url=item["web_url"]) for item in cleaned_url_contexts]
        
        
        pages_and_texts_chunks_list = [self.chunk_sentences(item=p, 
                                                            sentences_per_chunk=URL_sentences_per_chunk) for p in pages_and_texts_list]
        
        
        pages_and_chunks_list = [self.pages_n_chunks(item=p) for p in pages_and_texts_chunks_list]
        
        # Flattened list
        Pages_and_Chunks = [d for sublist in pages_and_chunks_list for d in sublist]
        
        return Pages_and_Chunks
           





   
    
    