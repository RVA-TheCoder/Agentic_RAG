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



# Create a class then add the class methods if required
class FetchURLContent:
    
    
    def __init__(self):
        
        pass
    

    # Source :  utils_methods\stage03_utils_method_url_content.py
    def preprocess_web_content(self, content: str) -> str:
        
        """
        Cleans raw web content from Jina or any HTML/Markdown-like source.
        
        Returns cleaned plain text suitable for RAG chunking.
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
        
        #url_contexts = []
        # STEP 1 : Get the Web raw content
        raw_context = self.fetch_url_content_with_fallback(url=web_url)
        
        # STEP2 : Clean the raw text
        if raw_context:
            
            cleaned_context = self.preprocess_web_content(content=raw_context)
            
            return cleaned_context
        
        #url_contexts.append(cleaned_context)
        
        return " "

    
    def process_page(self, text, source_url):
        
        words = word_tokenize(text)
        #sentences = sent_tokenize(text)

        #nlp = English()
        # Add a sentencizer pipeline
        #nlp.add_pipe("sentencizer")
        
        sentences = list(nlp(text=text).sents)
        # Make sure all sentences are strings
        sentences = [str(sentence) for sentence in sentences]

        return {
                    "source_url": source_url, 
                    "text": text,
                    "sentences" : sentences,
                    "webpage_sentence_count_spacy" : len(sentences),
 
                }
    
    # Chunking our sentences together
    # we will have to take into account the maximum input token limit of embedding model.
    def chunk_sentences(self, item , sentences_per_chunk):
        
        # Function that recursively splits a list into desired sizes
        def split_list(input_list: list,
                       slice_size: int) -> list[list[str]]:
            
            """
            Splits the input_list into sublists of size slice_size (or as close as possible).

            For example, a list of 16 sentences would be split into two lists of [[10], [6]] if slice_size = 10
            """

            return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

        # Loop through pages and texts and split sentences into chunks
        #for item in tqdm(Pages_and_Texts):
        item["sentence_chunks"] = split_list(input_list=item["sentences"],
                                            slice_size=sentences_per_chunk
                                            )

        item["num_chunks"] = len(item["sentence_chunks"])

        return item
    
    
    
    
    # Splitting each chunk into its own item
    # to embed each chunk of sentences into its own numerical representation.
    def pages_n_chunks(self, item) :
        
        """
        Convert sentence_chunks key in the Pages_and_Texts into its own chunk item.
       
        """

        # Split each chunk into its own item
        Pages_and_Chunks = []
        
        for sentence_chunk in item["sentence_chunks"] :  
            
            chunk_dict = {}
            chunk_dict["source_url"] = item["source_url"]
            
            # For original text
            #chunk_dict["text"] = item["text"]

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

            Pages_and_Chunks.append(chunk_dict)

        

        return Pages_and_Chunks
    
    
    

    def run_fetch_cleaned_url_content_pipeline(self, urls):
        
        """
        Accepts a single URL (str) or a list of URLs (List[str]).
        Fetches, cleans, and returns the combined cleaned content from the URLs.
        
        returns :  list of dictionaries
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

        #final_cleaned_url_text = "- " + "\n- ".join(cleaned_url_contexts_dict)
        
        # Save to a file
        with open("weburl_text_output.json", "w", encoding="utf-8") as f:
            json.dump(cleaned_url_contexts, f, indent=4, ensure_ascii=True)
        
        
        
        pages_and_texts_list = []
        for item in cleaned_url_contexts:
            
            pages_and_texts = self.process_page(text=item["cleaned_text"], source_url=item["web_url"])
            
            pages_and_texts_list.append(pages_and_texts)
        
        #print("pages_and_texts_list : ", pages_and_texts_list)
        
        pages_and_texts_chunks_list = []
        for item in pages_and_texts_list:
            
            pages_and_texts_chunks = self.chunk_sentences(item=item, sentences_per_chunk=3)
            
            pages_and_texts_chunks_list.append(pages_and_texts_chunks)
            
        #print("pages_and_texts_chunks_list : ", pages_and_texts_chunks_list)
        
        pages_and_chunks_list = []    
        for item in pages_and_texts_chunks_list:
            
            text_chunks = self.pages_n_chunks(item=item)
            
            pages_and_chunks_list.append(text_chunks)
            
            #print("text_chunks : ", text_chunks)
        
        # Flattened list
        Pages_and_Chunks = [d for sublist in pages_and_chunks_list for d in sublist]
        
        return Pages_and_Chunks
           
        #return cleaned_url_contexts 
































# source  : utils_methods\stage03_utils_method_url_content.py
# update this code from notebook
# def fetch_url_content(url:str) -> Optional[str]:
    
#     """
#     fetches content from a URL by performing an HTTP Get request.

#     parameters : 
#          url (str) : the endpoint or url to fetch content from.

#     returns :
#          Optional[str] : The content retrieved from the URL as a string or None if the request fails.

#     """
#     prefix_url : str = "https://r.jina.ai/"

#     # Concatenate the prefix url with the provided url
#     full_url :str = prefix_url + url 

#     try :
        
#         response = requests.get(full_url)
#         if response.status_code == 200:
#             # return the content of the response as a string
#             return response.content.decode("utf-8")

#         else : 
#             print(f"Error : HTTP Get request failed with status code {response.status_code}")
#             return None
            
#     except requests.RequestException as e:
        
#         print(f"Error : failed to fetch URL {full_url}. Exception : {e}")

#         return None
    
    
    
# add the functionality the if user input the "web_url" then the class pipeline method will return the 
# cleaned url_texts_chunks_and_embeddings.df , url_embeddings, etc.,     
    
    

    
    
    
    
    