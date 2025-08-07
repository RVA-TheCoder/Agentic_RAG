import pandas as pd
import numpy as np

import random
import re
import os
import requests
import fitz
# for progress bars, requires !pip install tqdm and run pip install -U jupyter ipywidgets
from tqdm.auto import tqdm

import torch

from spacy.lang.en import English
from duckduckgo_search import DDGS

from agentic_rag.constants.constants import METAPHOR_API_KEY,ONFLY_topk_URLs_to_return





# Check device Availability
def device_availabilty():
    
    """
    Detects and returns the available processing device (CUDA GPU or CPU).
    
    This function checks if a CUDA-enabled GPU is available. If so, it sets the 
    device to "cuda", retrieves and prints the total GPU memory in GB. If CUDA is 
    not available, it defaults to using the CPU and prints the selected device.

    Returns: 
        str: "cuda" if a GPU is available, otherwise "cpu".
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #print("torch version : ",torch.__version__)                 # 2.5.1
    #print("cuda version : " ,torch.version.cuda)               # 12.1
    #print("cudnn version : ",torch.backends.cudnn.version())   # not None
    #print("Is Cuda available : ",torch.cuda.is_available())        # True
    #print("GPU name : " ,torch.cuda.get_device_name(0))    # Should show RTX 4060 or your GPU

    if device == "cuda":

        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory_bytes / (1024 * 10 ** 6)
        print(f"GPU memory available: {gpu_memory_gb:.2f} GB")

    else:
        print(f"You are using {device} as your Processing device.")

    return device




# Import or Download a Document
def read_or_download_file(filepath, file_url):

    """
    Checks if a file exists locally, and downloads it from the given URL if not.

    This function verifies the presence of a file at the specified 'filepath'. 
    If the file is missing, it sends a GET request to the provided 'file_url' and saves the file 
    locally using the filename derived from 'filepath'.
    
    parameters : 
        (a) filepath (str): The local file path to check or save the downloaded file.
        (b) file_url (str): The URL from which to download the file if it doesn't exist.

    Returns:
        None
    """

    # Download PDF if it doesn't already exist
    if not os.path.exists(filepath):

        print("File doesn't exist, downloading...")
        
        # The local filename to save the downloaded file
        # Get the directory path
        directory_path = os.path.dirname(filepath)
        # Get the filename : to save the downloaded file in the local system
        filename = os.path.basename(filepath)
        
        #filename = pdf_filepath

        # Send a GET request to the URL
        response = requests.get(file_url)

        # Check if the request was successful
        if response.status_code == 200:

            # Open a file in binary write mode and save the content to it
            with open(filename, "wb") as file:
                file.write(response.content)
                print(f"The file has been downloaded and saved as {filename}")

        else:
            print(f"Failed to download the file. Status code: {response.status_code}")

    else:
        print(f"File {filepath} exists.")




# Formatting the text
import unicodedata
def text_formatter(text: str) -> str:

    """
    Performs minor formatting on text.
    """

    # Note: This might be different for each doc (best to experiment)
    # strip() : Return a copy of the string with leading and trailing whitespace removed.
    cleaned_text = text.replace("\n", " ").strip()

    # step : remove non-ascii characters
    cleaned_text = unicodedata.normalize('NFKD', cleaned_text)
    cleaned_text = cleaned_text.encode('ascii', 'ignore').decode('ascii')
    
    # Step : Normalize whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # step : remove Visual emphasis
    cleaned_text = re.sub(r"\*+", "", cleaned_text)

    # Step : General-purpose divider remover: Remove sequences of 3 or more below symbols in a appearing continuously
    cleaned_text = re.sub(r"[\*\-\=\_#]{3,}", "", cleaned_text)
    
    
    return cleaned_text



# Reading the embedding file
def load_embeddings_and_embedding_df(filepath):

    """
    Loads a saved CSV file containing text chunks and their corresponding embeddings.

    If the file exists at the specified path, this function reads the CSV, converts the
    string-encoded embedding column into NumPy arrays, and then converts them into a 
    PyTorch tensor of dtype float16 for efficient GPU processing.

    Parameters :
        (a) filepath (str): Path to the CSV file containing the saved embeddings.

    Returns:
        tuple:
            - embeddings (torch.Tensor): Tensor of embeddings loaded from the file.
            - text_chunks_and_embeddings_df (pd.DataFrame): DataFrame with original text chunks and their embeddings.

    Raises:
        Prints a warning if the file does not exist and returns None.
    """
    
    
    if os.path.exists(filepath) : 
        # read saved .csv file
        text_chunks_and_embeddings_df = pd.read_csv(filepath)
        
        text_chunks_and_embeddings_df['embedding'] = text_chunks_and_embeddings_df['embedding'].apply(
                                                                                                     lambda x: np.fromstring(string=x.strip("[]"),
                                                                                                                             dtype="float32",
                                                                                                                             sep=" ")
                                                                                                    )


        # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
        embeddings = torch.tensor(np.array(text_chunks_and_embeddings_df["embedding"].tolist()),
                                  dtype=torch.float16
                                  )

        return  embeddings, text_chunks_and_embeddings_df
        
    else:
        print(f"Path {filepath} doen not exist!")



# Get the LLM model size  
def get_model_mem_size(model: torch.nn.Module):
    
    """
    Get how much memory a PyTorch model takes up.

    See: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
    """
    # Get model parameters and buffer sizes
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    # Calculate various model sizes
    model_mem_bytes = mem_params + mem_buffers # in bytes
    model_mem_mb = model_mem_bytes / (1024**2) # in megabytes
    model_mem_gb = model_mem_bytes / (1024**3) # in gigabytes

    return {"model_mem_bytes": model_mem_bytes,
            "model_mem_mb": round(model_mem_mb, 2),
            "model_mem_gb": round(model_mem_gb, 2)}



def online_search_duckduckgo(user_query, topk_results=2):
    
    """
    Performs an online search using the DuckDuckGo search engine and returns a list of result URLs.

    Utilizes the DuckDuckGo Search API via the 'duckduckgo_search' library to fetch web search results 
    based on the user query.

    Parameters : 
        (a) user_query (str): The search query entered by the user.
        (b) topk_results (int, optional): Number of top search results to retrieve. Defaults to 2.

    Returns:
        list[str]: A list of URLs extracted from the top search results.
    """
    
    results = DDGS().text(user_query, max_results=topk_results)
    url_list = []
    for item in results:

        url_list.append(item['href'])
        #print(item['href'])
        #print()
    
    return url_list


def online_search_metaphor(user_query, topk_results=ONFLY_topk_URLs_to_return):
    
    """
    Performs a semantic web search using the Metaphor API and returns a list of relevant URLs.

    Sends a POST request to Metaphor's semantic search API with the user query and specified parameters 
    like result count and publish date filters.

    Parameters : 
        (a) user_query (str): The search query to be processed semantically.
        (b) topk_results (int): Number of top search results to return.

    Returns:
        list[str]: A list of URLs from the top search results.

    Note:
        Requires a valid 'METAPHOR_API_KEY' to be defined in the environment or globally.
    """

    headers = {
        "x-api-key": METAPHOR_API_KEY,
        "Content-Type": "application/json"
    }

    query = {
        "query": user_query,
        "numResults": topk_results,
        "startPublishedDate": "2022-01-01",
        "useAutoprompt": True
    }

    response = requests.post(
        "https://api.metaphor.systems/search",
        headers=headers,
        json=query
    )

    urls_list = []
    if response.status_code == 200:
        data = response.json()
        for result in data.get("results", []):
            urls_list.append(result["url"])
    else:
        print(f"[ERROR] Metaphor API request failed with status {response.status_code}: {response.text}")

    return urls_list




def format_llm_output(answer: str, deduplicate: bool = True):
    
    # Step 1: Convert escaped newlines to real newlines
    if "\\n" in answer:
        answer = answer.replace("\\n", "\n")

    # Step 2: Remove code block markers if any
    answer = answer.replace("```", "")

    # Step 3: Normalize bullet points - add space if missing after dash
    answer = re.sub(r'^[ \t]*-(\S)', r'- \1', answer, flags=re.MULTILINE)

    # Step 4: Convert dash bullets to asterisk bullets
    answer = re.sub(r'^[ \t]*- ', '* ', answer, flags=re.MULTILINE)

    # Step 5: Bold bullet labels like: * Label: something
    answer = re.sub(r'^(\* )([A-Z][\w\s\-]+?):', r'\1**\2:**', answer, flags=re.MULTILINE)

    # Step 6: Strip leading spaces
    answer = re.sub(r'^[ \t]+', '', answer, flags=re.MULTILINE)

    # Step 7 (optional): Collapse excessive blank lines
    answer = re.sub(r'\n{3,}', '\n\n', answer)

    # Step 8 (optional): Deduplicate identical lines
    if deduplicate:
        """
        This code removes duplicate non-empty lines from a multi-line string (answer) while 
        preserving blank lines for formatting. It improves the readability of LLM-generated 
        responses by eliminating repeated statements when the deduplicate flag is enabled.
        """
        lines_seen = set()
        unique_lines = []
        for line in answer.split('\n'):
            if line.strip() and line.strip() not in lines_seen:
                lines_seen.add(line.strip())
                unique_lines.append(line)
            elif not line.strip():  # Preserve blank lines
                unique_lines.append('')
        answer = '\n'.join(unique_lines)

    # Step 9: Final cleanup
    cleaned_answer = answer.strip()

    return cleaned_answer



def convert_markdown_to_html(answer: str) -> str:
    
    lines = answer.strip().split("\n")
    html_lines = []
    inside_list = False 

    for line in lines:
        line = line.strip()

        # Check if it's a bullet line (starts with "* " or "- ")
        if line.startswith("* ") or line.startswith("- "):
            if not inside_list:
                html_lines.append("<ul>")
                inside_list = True

            # Remove markdown asterisks and wrap content in <strong>
            content = line.lstrip("*- ").replace("**", "").strip()
            html_lines.append(f"<li><strong>{content}</strong></li>")
        else:
            if inside_list:
                html_lines.append("</ul>")
                inside_list = False

            # Convert normal lines to paragraphs
            html_lines.append(f"<p>{line}</p>")

    if inside_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


