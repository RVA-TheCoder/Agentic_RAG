import os
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from tqdm.auto import tqdm


import torch
from sentence_transformers import SentenceTransformer
from agentic_rag.constants.constants import *



class EmbeddingModel:
    
    """
    A class to handle loading/downloading a SentenceTransformer Embedding model from HuggingFace and generating
    embeddings for a list of sentence chunks.

    Attributes:
        embedding_model_name (str): Name of the Embedding model from HuggingFace or local directory.
        local_embed_model_dir (str): Local directory to store or retrieve the embedding model.
        embeddings_df_save_path (str): Path to save the generated embeddings DataFrame.
        device (str): Device to run the embedding model on ('cpu' or 'cuda').
    """

    def __init__(self,
                 embeddings_df_save_path=embeddings_df_save_path,
                 embedding_model_name=embedding_model_name,
                 local_embed_model_dir=local_embed_model_dir,
                 device="cpu"
                 ):
        """
        Initializes the EmbeddingModel instance.

        Parameters : 
           (a) embeddings_df_save_path (str): Path to save the embeddings DataFrame.
           (b) embedding_model_name (str): Name or path of the SentenceTransformer Embedding model.
           (c) local_embed_model_dir (str): Directory to store downloaded models.
           (d) device (str): Device to run model on ('cpu' or 'cuda').
        """
        
        self.embedding_model_name = embedding_model_name
        self.local_embed_model_dir = local_embed_model_dir
        self.device = device
        self.embeddings_df_save_path = embeddings_df_save_path
        
        

    def load_or_download_embedding_model(self):
        
        
        """
        Loads the embedding model from the local directory. If not found, downloads it 
        from HuggingFace and saves it locally.
        
        Returns:
            SentenceTransformer: The loaded or downloaded model instance.
        """
        
        local_model_path = os.path.join(self.local_embed_model_dir, self.embedding_model_name)

        try : 
            
            if os.path.exists(local_model_path):
                print(f"[INFO] Loading embedding model from local path: {local_model_path}")
                embedding_model = SentenceTransformer(local_model_path, device=self.device)
            
            else:
                print(f"[INFO] Downloading Embedding model from Hugging Face: {self.embedding_model_name}")
                embedding_model = SentenceTransformer(model_name_or_path=self.embedding_model_name,
                                                    device=self.device)
                os.makedirs(local_model_path, exist_ok=True)
                embedding_model.save(local_model_path)
                print(f"[INFO] Embedding Model saved locally at: {local_model_path}")
            
            return embedding_model
    
        except : 
            print(f"[ERROR] Failed to load or download embedding model: {e}")
            raise
    

    def embedding_chunks(self,
                         embedding_model,
                         device,
                         pages_and_chunks,
                         embeddings_df_save_path=None,
                         pages_and_chunks_df_replace: bool = False):
        
        """
        Generates or loads embeddings for the given sentence chunks.

        Parameters : 
            (a) embedding_model (SentenceTransformer): Pre-loaded embedding model.
            (b) device (str): Device to run the embedding model on.
            (c) pages_and_chunks : (list): List of dicts, each with a 'sentence_chunk' key.
            (d) embeddings_df_save_path (str, optional): Path to save or load the embeddings DataFrame.
                Defaults to the one provided during initialization.
            (e) pages_and_chunks_df_replace (bool): Whether to regenerate and overwrite existing embeddings.

        Returns:
            tuple:
                - embeddings (torch.Tensor): Tensor of shape [n_chunks, embedding_dim].
                - text_chunks_and_embeddings (list): List of dicts containing with sentence chunk(Text) and its embedding.
                - text_chunks_and_embeddings_df (pd.DataFrame): DataFrame containing all chunks and embeddings.
        """
        
        # Used fallback for embeddings_df_save_path for flexibility.
        if embeddings_df_save_path is None:
            embeddings_df_save_path = self.embeddings_df_save_path

        embedding_model = embedding_model.to(device)
        

        if not os.path.exists(embeddings_df_save_path) or pages_and_chunks_df_replace:
            print(f"[INFO] Embeddings file {embeddings_df_save_path} not found or replace=True. Generating new embeddings...")

            try : 
                #pages_and_chunks = pages_and_chunks_df.to_dict(orient="records")
                for item in tqdm(pages_and_chunks, desc="Embedding chunks"):
                    item["embedding"] = embedding_model.encode(item["sentence_chunk"], device=device)

                text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks)
                #text_chunks_and_embeddings = text_chunks_and_embeddings_df.to_dict(orient="records")
                text_chunks_and_embeddings_df.to_csv(embeddings_df_save_path, index=False)
                
            except :
                print(f"[ERROR] Failed during embedding or saving: {e}")
                raise

        else:
            print(f"[INFO]  Loading embeddings from: {embeddings_df_save_path}")
            
            try : 
                text_chunks_and_embeddings_df = pd.read_csv(embeddings_df_save_path)
                text_chunks_and_embeddings_df['embedding'] = text_chunks_and_embeddings_df['embedding'].apply(
                    lambda x: np.fromstring(string=x.strip("[]"), dtype="float32", sep=" ")
                )
                
            except:
                print(f"[ERROR] Failed to load or parse existing embeddings file: {e}")
                raise
        
        try:
            
            text_chunks_and_embeddings = text_chunks_and_embeddings_df.to_dict(orient="records")
            embeddings_tensors = torch.tensor(np.array(text_chunks_and_embeddings_df["embedding"].tolist()),
                                    dtype=torch.float16)

        except Exception as e:
            print(f"[ERROR] Failed to convert embeddings to tensor: {e}")
            raise
        
        return embeddings_tensors, text_chunks_and_embeddings, text_chunks_and_embeddings_df
    
    

    def run_embedding_pipeline(self, pages_and_chunks, replace: bool):
        
        """
        Full pipeline: load/download model, generate/load embeddings, return all data.

        Parameters : 
            (a) pages_and_chunks : (list): List of dicts with a 'sentence_chunk' key.
            (b) replace (bool): Whether to force regeneration of embeddings.

        Returns:
            tuple:
                - embedding_model : SentenceTransformer : The loaded embedding model.
                - embeddings (torch.Tensor): Tensor of shape [n_chunks, embedding_dim].
                - text_chunks_and_embeddings (list): List of dicts with text and embeddings.
                - text_chunks_and_embeddings_df (pd.DataFrame): DataFrame of sentence chunks and their embeddings.
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







