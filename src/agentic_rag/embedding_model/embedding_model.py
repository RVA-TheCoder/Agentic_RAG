import os
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import pandas as pd
import torch
import numpy as np



import os
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import pandas as pd
import torch
import numpy as np

from agentic_rag.constants.constants import *



class EmbeddingModel:
    
    """
    A class to manage downloading/loading SentenceTransformer embedding models
    and generating vector embeddings for sentence chunks.

    Attributes:
        embedding_model_name (str): Name of the model from HuggingFace or local directory.
        local_model_dir (str): Directory where downloaded models are stored locally.
        device (str): Device to run the embedding model on ('cpu' or 'cuda').
    """

    def __init__(self,
                 embeddings_df_save_path=embeddings_df_save_path,
                 embedding_model_name="all-mpnet-base-v2",
                 local_model_dir="HF_Embedding_Models",
                 device="cpu"):
        """
        Initializes the EmbeddingModel object.

        Args:
            embeddings_df_save_path (str): Path where the generated embeddings DataFrame should be saved.
            embedding_model_name (str): Name of the SentenceTransformer model.
            local_model_dir (str): Directory to save/download embedding models.
            device (str): Device to run the embedding model on ('cpu' or 'cuda').
        """
        self.embedding_model_name = embedding_model_name
        self.local_model_dir = local_model_dir
        self.device = device
        self.embeddings_df_save_path = embeddings_df_save_path
        
        

    def load_or_download_embedding_model(self):
        """
        Loads the embedding model from local storage if available; otherwise downloads it from HuggingFace.

        Returns:
            SentenceTransformer: Loaded embedding model instance.
        """
        local_model_path = os.path.join(self.local_model_dir, self.embedding_model_name)

        if os.path.exists(local_model_path):
            print(f"[INFO] Loading Embedding model from local directory: {local_model_path}")
            embedding_model = SentenceTransformer(local_model_path, device=self.device)
        else:
            print(f"[INFO] Downloading Embedding model from Hugging Face: {self.embedding_model_name}")
            embedding_model = SentenceTransformer(model_name_or_path=self.embedding_model_name,
                                                  device=self.device)
            os.makedirs(local_model_path, exist_ok=True)
            embedding_model.save(local_model_path)
            print(f"[INFO] Embedding Model saved locally to: {local_model_path}")
        
        return embedding_model
    
    

    def embedding_chunks(self,
                         embedding_model,
                         device,
                         pages_and_chunks,
                         embeddings_df_save_path=None,
                         pages_and_chunks_df_replace: bool = False):
        
        """
        Creates embeddings for sentence chunks and saves or loads them from disk.

        Args:
            embedding_model (SentenceTransformer): Preloaded embedding model.
            device (str): Device to run the embedding model on.
            pages_and_chunks_df (pd.DataFrame): DataFrame with a 'sentence_chunk' column to embed.
            embeddings_df_save_path (str, optional): Path to save or load the embeddings DataFrame.
                Defaults to the one provided during initialization.
            replace (bool): If True, regenerate and overwrite embeddings even if file exists.

        Returns:
            tuple:
                - embeddings (torch.Tensor): Tensor of shape [n_chunks, embedding_dim].
                - text_chunks_and_embeddings (list): List of dicts containing text and embeddings.
                - text_chunks_and_embeddings_df (pd.DataFrame): DataFrame with embedded chunks.
        """
        
        # Used fallback for embeddings_df_save_path for flexibility.
        if embeddings_df_save_path is None:
            embeddings_df_save_path = self.embeddings_df_save_path

        embedding_model = embedding_model.to(device)
        

        if not os.path.exists(embeddings_df_save_path) or pages_and_chunks_df_replace:
            print(f"[INFO] Embeddings file {embeddings_df_save_path} not found or replace=True. Generating embeddings...")

            #pages_and_chunks = pages_and_chunks_df.to_dict(orient="records")
            for item in tqdm(pages_and_chunks):
                item["embedding"] = embedding_model.encode(item["sentence_chunk"], device=device)

            text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks)
            text_chunks_and_embeddings = text_chunks_and_embeddings_df.to_dict(orient="records")
            text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

        else:
            print(f"[INFO] Embeddings loaded from: {embeddings_df_save_path}")
            text_chunks_and_embeddings_df = pd.read_csv(embeddings_df_save_path)
            text_chunks_and_embeddings_df['embedding'] = text_chunks_and_embeddings_df['embedding'].apply(
                lambda x: np.fromstring(string=x.strip("[]"), dtype="float32", sep=" ")
            )

        text_chunks_and_embeddings = text_chunks_and_embeddings_df.to_dict(orient="records")
        embeddings = torch.tensor(np.array(text_chunks_and_embeddings_df["embedding"].tolist()),
                                  dtype=torch.float16)

        return embeddings, text_chunks_and_embeddings, text_chunks_and_embeddings_df
    
    

    def run_embedding_pipeline(self, pages_and_chunks, replace: bool):
        
        """
        Full pipeline to generate or load embeddings from sentence chunks.

        Args:
            pages_and_chunks_df (pd.DataFrame): DataFrame with a 'sentence_chunk' column.
            replace (bool): Whether to regenerate embeddings even if cached file exists.

        Returns:
            tuple:
                - embeddings (torch.Tensor): Tensor of shape [n_chunks, embedding_dim].
                - text_chunks_and_embeddings (list): List of dicts with text and embeddings.
                - text_chunks_and_embeddings_df (pd.DataFrame): DataFrame with embedded chunks.
        """
        
        embedding_model = self.load_or_download_embedding_model()

        (embeddings, 
         text_chunks_and_embeddings,
         text_chunks_and_embeddings_df) = self.embedding_chunks(
                                                                embedding_model=embedding_model,
                                                                device=self.device,
                                                                embeddings_df_save_path=self.embeddings_df_save_path,
                                                                pages_and_chunks=pages_and_chunks,
                                                                pages_and_chunks_df_replace=replace
                                                              )

        return embedding_model, embeddings, text_chunks_and_embeddings, text_chunks_and_embeddings_df



























# class EmbeddingModel:
    
#     """
    
    
#     """
    
#     def __init__(self,
#                  embeddings_df_save_path,
#                  embedding_model_name="all-mpnet-base-v2",
#                  local_model_dir="embedding_models",
#                  device="cpu"):
        
#         self.embedding_model_name = embedding_model_name
#         self.local_model_dir = local_model_dir
#         self.device = device
#         #self.embeddings_df_save_path = embeddings_df_save_path
       
    
    
#     def load_or_download_embedding_model(self):
        
#         """
#         Loads a SentenceTransformer model from local directory if available.
#         If not, downloads and saves it for future reuse.
        
#         Args:
#             model_name (str): HuggingFace model ID or local subdirectory name.
#             local_dir (str): Directory to store/reuse models.
#             device (str): 'cpu' or 'cuda' (e.g., from torch.cuda.is_available()).

#         Returns:
#             SentenceTransformer model object.
#         """
#         local_model_path = os.path.join(self.local_model_dir, self.embedding_model_name)

#         if os.path.exists(local_model_path):
#             print(f"[INFO] Loading Embedding model from local directory: {local_model_path}")
#             embedding_model = SentenceTransformer(local_model_path, device=self.device)
#         else:
#             print(f"[INFO] Downloading Embedding model from Hugging Face: {self.embedding_model_name}")
#             embedding_model = SentenceTransformer(model_name_or_path=self.embedding_model_name,
#                                         device=self.device
#                                         )
            
#             os.makedirs(local_model_path, exist_ok=True)
#             embedding_model.save(local_model_path)
#             print(f"[INFO] Embedding Model saved locally to: {local_model_path}")
        
#         return embedding_model
        
    
#     # Create Embeddings
#     def embedding_chunks(self,
#                          embedding_model,
#                          device, 
#                          pages_and_chunks_df:pd.DataFrame,  # we need its value, do something about it 
#                          embeddings_df_save_path=embeddings_df_save_path,
#                          replace : bool = False 
#                         ):

#         """
#         Embed the text chunks into a 768-dimensional dense vector space.

#         Parameters : 
#         (a) embedding_model : embedding model to be used to create embeddings
#         (b) device :  device on which embedding model will run ("cpu" or "cuda")
#         (c) pages_and_chunks_df : pandas DF that has required 'sentence_chunk' column which is to be embedded using embedding model.
#         (d) embeddings_df_save_path : local System Path where embeddings pandas DF will be saved.
#         (e) replace : Boolean : create the new embeddings and replace the existing embedding Pandas DF file

#         return : embedding_model , embeddings, text_chunks_and_embeddings dictionary object, text_chunks_and_embeddings_df Pandas DF
#         """
    
#         # It maps sentences & paragraphs to a 768-dimensional dense vector space, and
#         # can be used for tasks like clustering or semantic search.

#         # Make sure the model is on the correct device
#         embedding_model = embedding_model.to(device)
        
#         if not os.path.exists(embeddings_df_save_path) or replace :

#             print(f"[INFO] Embeddings file {embeddings_df_save_path} not found or replace=True. Generating embeddings...")

#             # Embed each chunk one by one
#             pages_and_chunks = pages_and_chunks_df.to_dict(orient="records")
#             for item in tqdm(pages_and_chunks):
#                 item["embedding"] = embedding_model.encode(item["sentence_chunk"], device=device)

            
#             text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks)
#             text_chunks_and_embeddings = text_chunks_and_embeddings_df.to_dict(orient="records")
#             #embeddings = text_chunks_and_embeddings_df['embedding']

#             text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

            
#         else : 
            
#             # read saved CSV file
#             print(f"[INFO] Embeddings loaded from: {embeddings_df_save_path}")
#             text_chunks_and_embeddings_df = pd.read_csv(embeddings_df_save_path)
        
#             text_chunks_and_embeddings_df['embedding'] = text_chunks_and_embeddings_df['embedding'].apply(
#                                                                                                         lambda x: np.fromstring(string=x.strip("[]"),
#                                                                                                                                 dtype="float32",
#                                                                                                                                 sep=" ")
#                                                                                                         )

#         text_chunks_and_embeddings = text_chunks_and_embeddings_df.to_dict(orient="records")
#         # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
#         embeddings = torch.tensor(np.array(text_chunks_and_embeddings_df["embedding"].tolist()),
#                                 dtype=torch.float16
#                                 )
        
#         #return embedding_model, embeddings, text_chunks_and_embeddings_df
#         #return embedding_model, embeddings, text_chunks_and_embeddings, text_chunks_and_embeddings_df
#         return embeddings, text_chunks_and_embeddings, text_chunks_and_embeddings_df
    
    
#     def run_embedding_pipeline(self, pages_and_chunks_df:pd.DataFrame, replace):
        
#         # STEP 1 :  Get the Embedding model
#         embedding_model = self.load_or_download_embedding_model()
        
#         # STEP2 : Embedding Chunks
#         (embeddings, 
#          text_chunks_and_embeddings,
#          text_chunks_and_embeddings_df) = self.embedding_chunks( embedding_model=embedding_model,
#                                                                  device=self.device,
#                                                                  embeddings_df_save_path=embeddings_df_save_path,
#                                                                  pages_and_chunks_df=pages_and_chunks_df,
#                                                                  replace=replace)
    
#         return embeddings, text_chunks_and_embeddings, text_chunks_and_embeddings_df
    
    
    
    
    
       
        


# Embedding our text chunks

# def load_or_download_embedding_model(model_name="all-mpnet-base-v2", 
#                                      local_dir="embedding_models", 
#                                      device="cpu"
#                                     ):
#     """
#     Loads a SentenceTransformer model from local directory if available.
#     If not, downloads and saves it for future reuse.
    
#     Args:
#         model_name (str): HuggingFace model ID or local subdirectory name.
#         local_dir (str): Directory to store/reuse models.
#         device (str): 'cpu' or 'cuda' (e.g., from torch.cuda.is_available()).

#     Returns:
#         SentenceTransformer model object.
#     """
#     local_model_path = os.path.join(local_dir, model_name)

#     if os.path.exists(local_model_path):
#         print(f"[INFO] Loading Embedding model from local directory: {local_model_path}")
#         model = SentenceTransformer(local_model_path, device=device)
#     else:
#         print(f"[INFO] Downloading Embedding model from Hugging Face: {model_name}")
#         model = SentenceTransformer(model_name_or_path=model_name, device=device)
        
#         os.makedirs(local_model_path, exist_ok=True)
#         model.save(local_model_path)
#         print(f"[INFO] Embedding Model saved locally to: {local_model_path}")
    
#     return model



# Create Embeddings
# def embedding_chunks(embedding_model,
#                      device, 
#                      pages_and_chunks_df,
#                      embeddings_df_save_path,
#                      replace = False 
#                      ):

#     """
#     Embed the text chunks into a 768-dimensional dense vector space.

#     Parameters : 
#        (a) embedding_model : embedding model to be used to create embeddings
#        (b) device :  device on which embedding model will run ("cpu" or "cuda")
#        (c) pages_and_chunks_df : pandas DF that has required 'sentence_chunk' column which is to be embedded using embedding model.
#        (d) embeddings_df_save_path : local System Path where embeddings pandas DF will be saved.
#        (e) replace : Boolean : create the new embeddings and replace the existing embedding Pandas DF file

#     return : embedding_model , embeddings, text_chunks_and_embeddings dictionary object, text_chunks_and_embeddings_df Pandas DF
#     """
    
#     # It maps sentences & paragraphs to a 768-dimensional dense vector space, and
#     # can be used for tasks like clustering or semantic search.

#     # Make sure the model is on the correct device
#     embedding_model = embedding_model.to(device)
    
#     if not os.path.exists(embeddings_df_save_path) or replace :

#         print(f"[INFO] Embeddings file {embeddings_df_save_path} not found or replace=True. Generating embeddings...")

#         # Embed each chunk one by one
#         pages_and_chunks = pages_and_chunks_df.to_dict(orient="records")
#         for item in tqdm(pages_and_chunks):
#             item["embedding"] = embedding_model.encode(item["sentence_chunk"], device=device)

        
#         text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks)
#         text_chunks_and_embeddings = text_chunks_and_embeddings_df.to_dict(orient="records")
#         #embeddings = text_chunks_and_embeddings_df['embedding']

#         text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)

        
#     else : 
        
#         # read saved CSV file
#         print(f"[INFO] Embeddings loaded from: {embeddings_df_save_path}")
#         text_chunks_and_embeddings_df = pd.read_csv(embeddings_df_save_path)
    
#         text_chunks_and_embeddings_df['embedding'] = text_chunks_and_embeddings_df['embedding'].apply(
#                                                                                                      lambda x: np.fromstring(string=x.strip("[]"),
#                                                                                                                              dtype="float32",
#                                                                                                                              sep=" ")
#                                                                                                     )

#     text_chunks_and_embeddings = text_chunks_and_embeddings_df.to_dict(orient="records")
#     # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
#     embeddings = torch.tensor(np.array(text_chunks_and_embeddings_df["embedding"].tolist()),
#                               dtype=torch.float16
#                               )
    
#     #return embedding_model, embeddings, text_chunks_and_embeddings_df
#     return embedding_model, embeddings, text_chunks_and_embeddings, text_chunks_and_embeddings_df


