import textwrap
from time import perf_counter as timer
import numpy as np
from typing import Tuple , List
import pandas as pd

import torch
from sentence_transformers import util, SentenceTransformer

from agentic_rag.constants.constants import *
from agentic_rag.reranker_model.reranker import BGEReranker






class SemanticSearch:
    
    """
    A class to perform semantic search and reranking over precomputed sentence embeddings.

    Core Capabilities:
        - Encodes user query and computes top-k relevant text chunks using dot-product similarity.
        - Uses BGE reranker to refine ranking of top results.
        - Returns relevant context and metadata for downstream applications.
    
    """
    
    def __init__(self):
        
        pass
    
    
    # Define helper function to print wrapped text 
    def print_wrapped(self, text, wrap_length=text_wrap_length):
        
        # Utility to print wrapped text for better readability.  
        print(textwrap.fill(text, wrap_length))
        
        
    def topk_scores_and_indices(self, 
                                query: str,
                                embeddings: torch.Tensor,
                                embedding_model: SentenceTransformer,
                                device : str,
                                n_resources_to_return: int = topk_semantic_results,
                                print_time: bool = True
                                ) -> Tuple[torch.Tensor, np.ndarray] :
    
        """
        Encodes the query and computes top-k similar document chunks using dot-product similarity.


        Parameters : 
            (a) query (str):  User input-query string.
            (b) embeddings (torch.Tensor): Precomputed document embeddings [n_chunks, dim].
            (c) embedding_model (SentenceTransformer): The model used to embed both query and docs.
            (d) device (str or torch.device): Device to use ("cuda" or "cpu").
            (e) n_resources_to_return (int): Number of top results to return.
            (f) print_time (bool): Whether to print time taken for similarity calculation.

        Returns:
            Tuple[torch.Tensor, np.ndarray]: 
                - Similarity scores (sorted).
                - Indices of top-k most relevant chunks.
   
        """

        # Step 1: Encode and move query embedding to device
        query_embedding = embedding_model.encode(query, convert_to_tensor=True, device=device)
        query_embedding = query_embedding.to(dtype=torch.float16, device=device)

        embeddings = embeddings.to(dtype=torch.float16, device=device)
        
        # Step 2: Move embeddings to device and compute dot scores
        start_time = timer()
        dot_scores = util.dot_score(query_embedding, embeddings)[0]
        end_time = timer()

        if print_time:
            print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time - start_time:.4f} seconds.")


        # Step 3: Get top-k results
        n_resources_to_return = min(n_resources_to_return, len(dot_scores))
        scores, indices = torch.topk(dot_scores, k=n_resources_to_return)
        
        #print("indices : ", indices)
        #print("dot-Scores : ", scores)
        return scores, indices.cpu().numpy()

        
    
    def get_context(self, 
                    user_query:str, 
                    embeddings: torch.Tensor, 
                    embedding_model: SentenceTransformer, 
                    pages_n_chunks_embeddings_df: pd.DataFrame, 
                    device: str,
                    print_relevant_resources: bool = True,
                    topk_semantic_results: int = topk_semantic_results
                    )-> Tuple[List[dict], torch.Tensor, np.ndarray]:
        
        
        """
        Retrieves the top-k most relevant sentence chunks for a given query using semantic similarity.

        This method computes similarity between the query and all precomputed document embeddings, 
        selects the top-k most similar chunks, and optionally prints the scores, chunk content, and page numbers.


        Returns:
            Tuple:
                - context_items (List[dict]): List of top-k relevant document chunks and their metadata.
                - scores (torch.Tensor): Similarity scores for top-k chunks.
                - indices (np.ndarray): Index positions of top-k chunks in the original DataFrame.
        """
        
        
        scores, indices = self.topk_scores_and_indices(query=user_query,
                                                        embeddings=embeddings,
                                                        embedding_model=embedding_model ,
                                                        device=device,
                                                        n_resources_to_return=topk_semantic_results,
                                                        print_time=True,
                                                        
                                                        )
        
        # Convert texts and embedding df to list of dicts
        pages_and_chunks_dict = pages_n_chunks_embeddings_df.to_dict(orient="records")

        # Create a list of context items
        context_items = [pages_and_chunks_dict[i] for i in indices]
        
        if print_relevant_resources:
            
            print("Score : ", scores, "\n", "Indices : ", indices, "\n")

            print("Results:")
            # Loop through zipped together scores and indicies from torch.topk
            for score, idx in zip(scores, indices):
                print(f"Idx: {idx} | Score: {score:.4f}")
                print(f"Page: {pages_n_chunks_embeddings_df.iloc[idx]['page_number']}")
                print("Text:")
                self.print_wrapped(pages_n_chunks_embeddings_df.iloc[idx]["sentence_chunk"])
                print("-" * 60) 
            
              
        return context_items, scores, indices
        
                  
    def run_pipeline(self, 
                    user_query: str, 
                    embeddings: torch.Tensor, 
                    embedding_model: SentenceTransformer, 
                    pages_n_chunks_embeddings_df: pd.DataFrame,
                    device: str,
                    print_relevant_resources: bool,
                    topk_semantic_results: int = topk_semantic_results
                    ) -> Tuple[str, List[dict], torch.Tensor, np.ndarray]:
        
        """
        Executes full semantic search and reranking pipeline.

        Returns:
            - context: final reranked joined paragraph string.
            - context_items: initial top-k chunks.
            - scores: similarity scores.
            - indices: indices of top-k chunks.
        """
        
        context_items , scores, indices = self.get_context(user_query=user_query, 
                                                           embeddings=embeddings, 
                                                           embedding_model=embedding_model, 
                                                           pages_n_chunks_embeddings_df=pages_n_chunks_embeddings_df, 
                                                           device=device,
                                                           print_relevant_resources=print_relevant_resources,
                                                           topk_semantic_results = topk_semantic_results
                                                          )
        
        
        retrieve_chunks = [item["sentence_chunk"] for item in context_items]
        
        ###########################   Re-ranker Model    #############################
        reranker = BGEReranker()
        if topk_semantic_results < topk_reranker_results:
            raise ValueError("topk_results must be >= topk_reranker_results")
        
        top_docs = reranker.rerank(query=user_query, passages=retrieve_chunks, top_k=topk_reranker_results)
        
        context = "\n\n".join([doc for doc, _ in top_docs])
        
        return context, context_items , scores, indices
        
    



