import textwrap
from sentence_transformers import util, SentenceTransformer
from time import perf_counter as timer


from agentic_rag.constants.constants import *
import torch

from agentic_rag.reranker_model.reranker import BGEReranker



class SemanticSearch:
    
    def __init__(self):
        
        pass
    
    
    # Define helper function to print wrapped text 
    def print_wrapped(self, text, wrap_length=text_wrap_length):
        
        wrapped_text = textwrap.fill(text, wrap_length)
        print(wrapped_text)
        
        
    def topk_scores_and_indices(self, 
                                query: str,
                                embeddings: torch.Tensor,
                                embedding_model: SentenceTransformer,
                                device,
                                n_resources_to_return: int = topk_resources_to_return,
                                print_time: bool = True
                                ):
    
        """
        Embeds a query, computes dot similarity against document embeddings,
        and returns the top-k most similar results.

        Args:
            query (str): The input query.
            embeddings (torch.Tensor): Tensor of precomputed document embeddings.
            embedding_model (SentenceTransformer): The model used to embed both query and docs.
            device (str or torch.device): Device to use ("cuda" or "cpu").
            n_resources_to_return (int): Number of top results to return.
            print_time (bool): Whether to print elapsed time.

        Returns:
            Tuple[torch.Tensor, np.ndarray]: similarity scores, and top indices
        """

        # Step 1: Encode and move query embedding to device
        query_embedding = embedding_model.encode(query, convert_to_tensor=True, device=device)
        query_embedding = query_embedding.to(dtype=torch.float16, device=device)

        # Step 2: Move embeddings to device and compute dot scores
        start_time = timer()
        embeddings = embeddings.to(dtype=torch.float16, device=device)
        dot_scores = util.dot_score(query_embedding, embeddings)[0]
        end_time = timer()

        if print_time:
            print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time - start_time:.5f} seconds.")

        # Step 3: Get top-k results
        #print("dot_scores : " , dot_scores)
        print("length of dot scores  :", len(dot_scores))
        if n_resources_to_return > len(dot_scores):
            
            n_resources_to_return = len(dot_scores)
            
        scores, indices = torch.topk(dot_scores, k=n_resources_to_return)
        indices = indices.cpu().numpy()

        #print("indices : ", indices)
        #print("Scores : ", scores)
        
        return scores, indices
        
     
       
    def get_context(self, 
                    user_query, 
                    embeddings, 
                    embedding_model, 
                    pages_n_chunks_embeddings_df, 
                    device,
                    print_relevant_resources=True,
                    topk_results = topk_resources_to_return
                    ):
        
        scores, indices = self.topk_scores_and_indices(query=user_query,
                                                        embeddings=embeddings,
                                                        embedding_model=embedding_model ,
                                                        device=device,
                                                        n_resources_to_return=topk_results,
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
                print("Idx : ", idx)
                print(f"Score: {score:.4f}")

                # Print the page number too so we can reference the textbook further (and check the results)
                # print(f"Page number: {pages_and_chunks[idx]['page_number']}")
                print(f"Page number: {pages_n_chunks_embeddings_df.iloc[idx]['page_number']}")

                # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
                # print_wrapped(pages_and_chunks[idx]["sentence_chunk"])
                print("Text:")
                self.print_wrapped(pages_n_chunks_embeddings_df.iloc[idx]["sentence_chunk"])

                print("\n") 
            
            
            
        return context_items, scores, indices
        
        
        
              
    def run_pipeline(self, 
                    user_query, 
                    embeddings, 
                    embedding_model, 
                    pages_n_chunks_embeddings_df,
                    device,
                    print_relevant_resources,
                    topk_results
                    ):
        
        
        context_items , scores, indices = self.get_context(user_query=user_query, 
                                                           embeddings=embeddings, 
                                                           embedding_model=embedding_model, 
                                                           pages_n_chunks_embeddings_df=pages_n_chunks_embeddings_df, 
                                                           device=device,
                                                           print_relevant_resources=print_relevant_resources,
                                                           topk_results = topk_results
                                                          )
        
        
        # Join context items into one dotted paragraph
        #context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
        
        #context  = "\n\n".join([item["sentence_chunk"] for item in context_items])
        
        retrieve_chunks = []
        for sent_chunk in context_items:
            
            retrieve_chunks.append(sent_chunk['sentence_chunk'])
            #print(sent_chunk['sentence_chunk'],"\n\n","-"*100)
        
        reranker = BGEReranker()
        
        top_docs = reranker.rerank(query=user_query, passages=retrieve_chunks, top_k=topk_results)
        
        
        top_docs_list = []
        for item in top_docs:
            top_docs_list.append(item[0])

        #print(top_docs_list)
        context  = "\n\n".join([item for item in top_docs_list])
        
    
        return context, context_items , scores, indices
        
    






















